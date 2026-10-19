import numpy as np
from sklearn.base import clone

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.program import Node, Program
from fastsymbolicgp.canonical import canonicalize_program
from fastsymbolicgp.memetic import optimize_constants
from fastsymbolicgp.model_selection import select_final_model
from fastsymbolicgp.pareto import objective_value, assign_nsga2


def _data(seed=0,n=180):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,3)); y=1.8*X[:,0]-0.4*X[:,1]**2+0.15
    return X,y


def test_semantic_gp_archive_and_diagnostics_are_real():
    X,y=_data()
    m=FastSymbolicRegressor(population_size=22,generations=3,random_state=4,verbose=0,
        selection_method="semantic_tournament",semantic_diversity=True,semantic_archive_size=10,
        semantic_sample_size=48,semantic_neighbors=3,semantic_mutation_rate=0.35,semantic_mutation_trials=3,
        constant_optimization="off").fit(X,y)
    r=m.get_semantic_report()
    assert r["enabled"] is True
    assert 1 <= r["size"] <= 10
    assert r["unique_semantics"] == r["size"]
    assert "semantic_novelty" in m.history_[-1]


def test_age_layered_evolution_tracks_lineage_age():
    X,y=_data(1)
    m=FastSymbolicRegressor(population_size=20,generations=4,random_state=3,verbose=0,
        age_management="layered",age_gap=1,age_layers=3,constant_optimization="off").fit(X,y)
    d=m.get_evolution_diagnostics()
    assert d["age_management"] == "layered"
    assert d["max_age"] >= 1
    assert any(row["max_age"] >= 1 for row in m.history_)


def test_canonicalization_unifies_commutative_associative_forms():
    x0=Node("feature",0); x1=Node("feature",1)
    a=Program(Node("func","add",(x1,Node("func","add",(x0,Node("const",0.0))))))
    b=Program(Node("func","add",(x0,x1)))
    ca=canonicalize_program(a,"canonical")
    cb=canonicalize_program(b,"canonical")
    assert ca.to_string() == cb.to_string()
    assert ca.structural_hash() == cb.structural_hash()


def test_memetic_optimizer_improves_constant_objective():
    p=Program(Node("const",-4.0))
    def objective(q): return (float(q.root.value)-2.5)**2
    q,report=optimize_constants(p,objective,[(-5.0,5.0)],method="hybrid",rounds=4,budget=60)
    assert report["evaluations"] > 0
    assert report["final_objective"] < report["initial_objective"]
    assert abs(float(q.root.value)-2.5) < abs(float(p.root.value)-2.5)


def test_integrated_memetic_search_runs_inside_evolution():
    X,y=_data(2)
    m=FastSymbolicRegressor(population_size=22,generations=3,random_state=7,verbose=0,
        const_probability=0.65,memetic_search="elites",memetic_elites=1,memetic_interval=1,
        memetic_budget=18,constant_optimization="off").fit(X,y)
    assert m.profile_["memetic_objective_evaluations"] > 0
    assert "memetic" in m.profile_["stage_seconds"]


def test_smallest_within_tolerance_model_selection_prefers_simplicity():
    p1=Program(Node("func","add",(Node("feature",0),Node("const",0.1))),validation_loss=1.00)
    p2=Program(Node("feature",0),validation_loss=1.01)
    p3=Program(Node("const",0.0),validation_loss=2.0)
    chosen,report=select_final_model([p1,p2,p3],"smallest_within_tolerance",tolerance=0.02)
    assert chosen.size == 1
    assert report["near_optimal_count"] == 2


def test_pareto_supports_semantic_and_age_objectives():
    a=Program(Node("feature",0),validation_loss=1.0); b=Program(Node("feature",1),validation_loss=1.0)
    a.metadata.update(semantic_novelty=0.2,age=5); b.metadata.update(semantic_novelty=0.8,age=1)
    assert objective_value(b,"semantic_redundancy") < objective_value(a,"semantic_redundancy")
    assert objective_value(b,"age") < objective_value(a,"age")
    assign_nsga2([a,b],("validation_loss","semantic_redundancy","age"))
    assert "pareto_rank" in a.metadata and "pareto_rank" in b.metadata


def test_pareto_knee_and_consensus_model_selection_fit_paths():
    X,y=_data(3)
    for mode in ("pareto_knee","semantic_consensus"):
        m=FastSymbolicRegressor(population_size=18,generations=2,random_state=5,verbose=0,
            model_selection=mode,constant_optimization="off").fit(X,y)
        assert m.model_selection_report_["mode"] == mode
        assert np.isfinite(m.predict(X[:8])).all()


def test_new_v120_parameters_remain_sklearn_cloneable():
    m=FastSymbolicRegressor(semantic_diversity=True,semantic_archive_size=17,age_management="layered",
        memetic_search="elites",memetic_budget=33,model_selection="pareto_knee",simplification="canonical")
    c=clone(m)
    assert c.semantic_archive_size == 17
    assert c.memetic_budget == 33
    assert c.model_selection == "pareto_knee"


def test_binary_classifier_supports_semantic_age_v120_features():
    from fastsymbolicgp import FastSymbolicClassifier
    X,yreg=_data(8,160); y=(yreg>np.median(yreg)).astype(int)
    m=FastSymbolicClassifier(population_size=18,generations=2,random_state=2,verbose=0,
        semantic_diversity=True,selection_method="semantic_tournament",age_management="layered",age_gap=1).fit(X,y)
    assert m.get_semantic_report()["enabled"] is True
    assert m.get_evolution_diagnostics()["max_age"] >= 1
    assert np.isfinite(m.predict_proba(X[:10])).all()


def test_semantic_archive_and_age_survive_checkpoint_resume(tmp_path):
    X,y=_data(9,150); ck=tmp_path/"v120_resume.joblib"
    a=FastSymbolicRegressor(population_size=18,generations=2,random_state=11,verbose=0,
        semantic_diversity=True,age_management="layered",age_gap=1,checkpoint_path=str(ck),checkpoint_interval=1,
        constant_optimization="off").fit(X,y)
    assert ck.exists() and len(a.semantic_archive_)>0
    b=FastSymbolicRegressor(population_size=18,generations=4,random_state=11,verbose=0,
        semantic_diversity=True,age_management="layered",age_gap=1,checkpoint_path=str(ck),checkpoint_interval=1,
        resume_from_checkpoint=True,constant_optimization="off").fit(X,y)
    assert b.n_generations_ >= 3
    assert b.evolution_state_["schema"] == 4
    assert len(b.semantic_archive_)>0
    assert b.get_evolution_diagnostics()["max_age"] >= 1


def test_v120_safe_json_roundtrip_preserves_predictions(tmp_path):
    X,y=_data(10,150)
    m=FastSymbolicRegressor(population_size=18,generations=2,random_state=3,verbose=0,
        simplification="canonical",semantic_diversity=True,model_selection="pareto_knee",constant_optimization="off").fit(X,y)
    path=tmp_path/"model.fsgp.json"; m.save_model_json(path)
    loaded=FastSymbolicRegressor.load_model_json(path)
    assert np.allclose(m.predict(X[:20]),loaded.predict(X[:20]),rtol=1e-10,atol=1e-10)
