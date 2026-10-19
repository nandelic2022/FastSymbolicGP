import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from fastsymbolicgp import FastSymbolicClassifier, FastSymbolicRegressor, warmup


def test_named_preset_preserves_explicit_user_parameters():
    X,y=load_breast_cancer(return_X_y=True)
    m=FastSymbolicClassifier(preset='balanced',population_size=37,generations=3,n_islands=3,display='off',verbose=0,random_state=1,evaluation_backend='numpy')
    m.fit(X[:180],y[:180])
    assert m.population_size==37 and m.generations==3 and m.n_islands==3
    assert m.resolved_config_['population_size']==37
    assert 'population_size' in m.preset_skipped_parameters_


def test_auto_planner_small_dataset_keeps_public_constructor_state():
    X,y=load_breast_cancer(return_X_y=True)
    m=FastSymbolicClassifier(preset='auto',generations=2,population_size=20,display='off',verbose=0,random_state=2,evaluation_backend='numpy').fit(X[:220],y[:220])
    assert m.execution_plan_['workload']=='small'
    assert m.n_islands==1 and m.evolution_model=='panmictic'


def test_exact_validation_champion_surface_is_available():
    X,y=load_breast_cancer(return_X_y=True)
    m=FastSymbolicClassifier(population_size=20,generations=4,tournament_size=4,fitness_scaling='elite_logistic',fitness_scaling_interval=99,
        exact_champion=True,exact_champion_candidates=3,validation_fraction=.25,n_islands=1,display='off',verbose=0,random_state=3,evaluation_backend='numpy').fit(X[:260],y[:260])
    assert m.exact_validation_champion_ is not None and m.best_validation_program_ is not None
    vals=[r['best_exact_validation_fitness'] for r in m.history_]
    assert all(v is not None and np.isfinite(v) for v in vals)
    assert all(vals[i]<=vals[i-1]+1e-12 for i in range(1,len(vals)))


def test_profile_summary_manifest_v112():
    X,y=load_diabetes(return_X_y=True)
    m=FastSymbolicRegressor(population_size=16,generations=2,tournament_size=3,display='off',verbose=0,random_state=4,uncertainty='off',evaluation_backend='numpy').fit(X[:180],y[:180])
    assert m.profile_['actual_program_evaluations']>0 and m.profile_['programs_per_second']>0
    assert m.summary(print_output=False)['version']=='1.2.8'
    manifest=m.model_manifest(); assert manifest['format']=='FastSymbolicGP-model-manifest-v3' and manifest['version']=='1.2.8'


def test_flat_genome_and_paths_v1_compatibility():
    X,y=load_diabetes(return_X_y=True)
    m=FastSymbolicRegressor(population_size=12,generations=1,tournament_size=2,display='off',verbose=0,random_state=5,uncertainty='off',evaluation_backend='numpy').fit(X[:100],y[:100])
    g=m.best_program_.flat_genome(); assert g['opcodes'].shape==g['arguments'].shape; assert g['size']==m.best_program_.size; assert len(m.best_program_.paths())==m.best_program_.size


def test_legacy_constant_probability_alias():
    X,y=load_diabetes(return_X_y=True)
    m=FastSymbolicRegressor(population_size=18,generations=2,tournament_size=3,constant_probability=.8,constant_optimization='champion',constant_optimization_rounds=1,
        display='off',verbose=0,random_state=7,uncertainty='off',evaluation_backend='numpy').fit(X[:180],y[:180])
    assert m.constant_optimization_report_['enabled'] is True


def test_warmup_surface():
    assert warmup() is not None
