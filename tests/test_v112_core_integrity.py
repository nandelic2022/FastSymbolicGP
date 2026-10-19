import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from fastsymbolicgp import (
    FastSymbolicRegressor, FastSymbolicClassifier, SymbolicProgram,
    FastSymbolicNetworkClassifier, DistilledSymbolicClassifier,
    feature_stability, subexpression_stability,
)
from fastsymbolicgp.program import Program, Node
from fastsymbolicgp.constraints import unit_constraint_penalty


def small_reg(**kwargs):
    p=dict(population_size=14,generations=3,max_depth=3,max_nodes=31,random_state=12,verbose=0,
           uncertainty="off",constant_optimization="off",evaluation_backend="numpy")
    p.update(kwargs); return FastSymbolicRegressor(**p)


def test_v112_version_and_v100_public_surface():
    import fastsymbolicgp
    assert fastsymbolicgp.__version__ == "1.2.8"
    assert SymbolicProgram is Program
    assert FastSymbolicNetworkClassifier is not None
    assert DistilledSymbolicClassifier is not None
    assert callable(feature_stability) and callable(subexpression_stability)


def test_preset_does_not_mutate_constructor_params():
    X,y=make_regression(n_samples=80,n_features=3,random_state=1)
    m=FastSymbolicRegressor(preset="balanced",population_size=500,generations=50,max_depth=12,
                            uncertainty="off",constant_optimization="off",evaluation_backend="numpy",random_state=1,verbose=0)
    before=m.get_params(deep=False).copy()
    m.set_params(population_size=12,generations=2)
    explicit=m.get_params(deep=False).copy()
    m.fit(X,y)
    after=m.get_params(deep=False)
    assert after["population_size"]==explicit["population_size"]==12
    assert after["generations"]==explicit["generations"]==2
    assert after["max_depth"]==before["max_depth"]
    assert m.resolved_config_["constructor_params_mutated"] is False


def test_hard_depth_and_nodes_hold_for_all_retained_programs():
    X,y=make_regression(n_samples=90,n_features=4,random_state=2)
    m=small_reg(population_size=20,generations=4,max_depth=2,max_nodes=5).fit(X,y)
    programs=[m.best_program_]+m.get_pareto_front()+list(getattr(m,"hall_of_fame_",[]))
    assert programs
    assert all(p.depth<=2 and p.size<=5 for p in programs)


def test_unit_enforcement_survives_variation():
    rng=np.random.default_rng(3);X=rng.uniform(.2,2,(100,2));y=X[:,0]*X[:,1]
    m=small_reg(population_size=24,generations=4,max_depth=3,function_set=("mul","div"),
                feature_units=["m","s"],target_unit="m*s",unit_policy="enforce").fit(X,y)
    programs=[m.best_program_]+m.get_pareto_front()+list(getattr(m,"hall_of_fame_",[]))
    for p in programs:
        pen,_=unit_constraint_penalty(p,["m","s"],"m*s")
        assert pen==0


def test_lexicase_cache_is_precomputed_once_per_program():
    X,y=make_regression(n_samples=70,n_features=3,random_state=4)
    m=small_reg(selection_method="epsilon_lexicase")
    m._prepare_X(X,fit=True);m._fit_n_samples_=len(X);from fastsymbolicgp.primitives import validate_function_set
    m.function_set_=validate_function_set(m.function_set);m._resolve_preset();rng=np.random.default_rng(4)
    pop=[m._make_program(rng) for _ in range(10)];m.evaluation_count_=0;m._evaluate_population(pop,X,y,None,None)
    count={"n":0};orig=m._case_losses
    def wrapped(p,X_,y_): count["n"]+=1; return orig(p,X_,y_)
    m._case_losses=wrapped
    m._prepare_lexicase_cache(pop,X,y)
    assert count["n"]==len(pop)
    for _ in range(20):m._select(pop,rng,X,y)
    assert count["n"]==len(pop)


def test_checkpoint_is_predictable_and_resumable(tmp_path):
    X,y=make_regression(n_samples=100,n_features=3,random_state=5);path=tmp_path/"resume.chk"
    first=small_reg(generations=2,checkpoint_path=str(path),checkpoint_interval=1).fit(X,y)
    loaded=FastSymbolicRegressor.load_checkpoint(path)
    assert np.isfinite(loaded.predict(X[:8])).all()
    assert loaded.evolution_state_["generation"]==2
    resumed=small_reg(generations=4,checkpoint_path=str(path),checkpoint_interval=1,resume_from_checkpoint=True).fit(X,y)
    assert resumed.history_[-1]["generation"]==4
    assert resumed.evolution_state_["generation"]==4


def test_island_evolution_and_migration_are_real():
    X,y=make_regression(n_samples=90,n_features=3,random_state=6)
    m=small_reg(population_size=18,generations=3,n_islands=3,evolution_model="islands",migration_interval=1,migration_size=1).fit(X,y)
    assert m.profile_["n_islands"]==3
    assert all(row["islands"]==3 for row in m.history_)


def test_pareto_search_assigns_rank_and_uses_full_archive():
    X,y=make_regression(n_samples=90,n_features=3,random_state=7)
    m=small_reg(population_size=18,generations=3,optimization="pareto").fit(X,y)
    assert len(m.pareto_archive_)>=1
    assert all("pareto_rank" in p.metadata for p in m.hall_of_fame_)


def test_v100_program_compatibility_helpers():
    p=Program(Node("func","add",(Node("feature",0),Node("const",1.0))))
    p.scale_=2.;p.intercept_=3.;p.raw_fitness_=4.;p.validation_fitness_=5.
    g=p.flat_genome()
    assert g["opcodes"].shape==g["arguments"].shape
    assert g["size"]==p.size==len(p.paths())
    assert p.dag_size>=1 and isinstance(p.structural_hash(),str)


def test_classifier_combines_class_and_sample_weights_and_validation_metric():
    X,y=make_classification(n_samples=120,n_features=4,n_informative=3,n_redundant=0,weights=[.8,.2],random_state=8)
    sw=np.linspace(.5,1.5,len(y))
    m=FastSymbolicClassifier(population_size=12,generations=2,max_depth=3,class_weight="balanced",
                             validation_metric="balanced_accuracy",random_state=2,verbose=0,evaluation_backend="numpy")
    # sample_weight is supplied to fit; constructor line above deliberately keeps sklearn-style API.
    m.fit(X,y,sample_weight=sw)
    assert np.isfinite(m.predict_proba(X[:5])).all()
    assert m.threshold_source_ in {"validation","training_fallback"}


def test_fused_binary_classification_matches_materialized_approx_logistic():
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from fastsymbolicgp import FastSymbolicClassifier
    X,y=load_breast_cancer(return_X_y=True);X=X[:180];y=y[:180]
    common=dict(population_size=14,generations=1,fitness_scaling='approx_logistic',fitness_metric='log_loss',validation_fraction=.2,random_state=812,verbose=0,selection_method='tournament',optimization='scalar')
    fused=FastSymbolicClassifier(**common,fitness_engine='fused').fit(X,y)
    materialized=FastSymbolicClassifier(**common,fitness_engine='materialized').fit(X,y)
    assert fused.resolved_config_['fitness_engine']=='fused'
    assert materialized.resolved_config_['fitness_engine']=='materialized'
    assert fused.get_expression()==materialized.get_expression()
    np.testing.assert_allclose(fused.predict_proba(X),materialized.predict_proba(X),rtol=1e-8,atol=1e-10)
