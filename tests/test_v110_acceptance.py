import json
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import RandomizedSearchCV

from fastsymbolicgp import (
    FastSymbolicRegressor, FastSymbolicClassifier, FastSymbolicTransformer,
    make_primitive, make_metric, doctor,
)
from fastsymbolicgp.backend import NUMBA_AVAILABLE
from fastsymbolicgp.program import Node, Program, infer_unit
from fastsymbolicgp.units import parse_unit


def small_reg(**kwargs):
    p=dict(population_size=16,generations=3,max_depth=4,patience=20,random_state=7,verbose=0,uncertainty="off")
    p.update(kwargs); return FastSymbolicRegressor(**p)


def small_clf(**kwargs):
    p=dict(population_size=16,generations=3,max_depth=4,patience=20,random_state=7,verbose=0)
    p.update(kwargs); return FastSymbolicClassifier(**p)


def test_version_and_clone():
    import fastsymbolicgp
    assert fastsymbolicgp.__version__ == "1.2.8"
    assert isinstance(clone(small_reg()), FastSymbolicRegressor)
    assert isinstance(clone(small_clf()), FastSymbolicClassifier)


def test_regression_and_lexicase():
    X,y=make_regression(n_samples=100,n_features=4,noise=.1,random_state=1)
    m=small_reg(selection_method="epsilon_lexicase").fit(X,y)
    assert np.isfinite(m.predict(X[:10])).all()
    assert m.n_generations_ >= 1
    assert m.get_expression_stats()["nodes"] >= 1


def test_binary_classifier():
    X,y=make_classification(n_samples=120,n_features=5,n_informative=4,n_redundant=0,random_state=2)
    m=small_clf(selection_method="lexicase").fit(X,y)
    P=m.predict_proba(X[:10])
    assert P.shape==(10,2)
    assert np.allclose(P.sum(axis=1),1.0)


def test_multiclass_ovr():
    X,y=load_iris(return_X_y=True)
    m=small_clf(population_size=12,generations=2).fit(X,y)
    P=m.predict_proba(X[:8])
    assert P.shape==(8,3)
    assert np.allclose(P.sum(axis=1),1.0)


def test_conformal_interval():
    X,y=make_regression(n_samples=140,n_features=4,noise=2,random_state=3)
    m=small_reg(uncertainty="split_conformal",calibration_fraction=.2).fit(X,y)
    p,lo,hi=m.predict_interval(X[:15],.90)
    assert p.shape==lo.shape==hi.shape==(15,)
    assert np.all(lo<=p) and np.all(p<=hi)
    assert m.calibration_size_ > 0


def test_unit_system_and_enforcement():
    assert parse_unit("W") == parse_unit("kg*m^2/s^3")
    root=Node("func","add",(Node("feature",0),Node("feature",1)))
    assert infer_unit(root,[parse_unit("W"),parse_unit("W")]) == parse_unit("W")
    X,y=make_regression(n_samples=80,n_features=3,random_state=4)
    m=small_reg(function_set=("add","sub"),feature_units=["W","W","W"],target_unit="W",unit_policy="enforce").fit(X,y)
    assert np.isfinite(m.predict(X[:5])).all()


def test_constraints_are_recorded():
    X=np.linspace(0,1,100)[:,None]
    y=2*X[:,0]+1
    m=small_reg(constraints={"output_min":0,"monotonic":{"X0":"increasing"},"max_variables":1}).fit(X,y)
    info=m.explain()
    assert "constraint_issues" in info


def test_safe_json_roundtrip(tmp_path):
    X,y=make_regression(n_samples=100,n_features=3,random_state=5)
    m=small_reg(uncertainty="split_conformal").fit(X,y)
    path=tmp_path/"model.json"; m.save_model_json(path)
    m2=FastSymbolicRegressor.load_model_json(path)
    assert np.allclose(m.predict(X[:20]),m2.predict(X[:20]))


def test_transformer():
    X,y=make_regression(n_samples=80,n_features=4,random_state=6)
    t=FastSymbolicTransformer(n_components=2,population_size=12,generations=2,max_depth=3,random_state=1,verbose=0).fit(X,y)
    Z=t.transform(X[:7])
    assert Z.shape==(7,2)
    assert len(t.get_feature_names_out())==2


def test_pareto_and_history():
    X,y=make_regression(n_samples=90,n_features=3,random_state=7)
    m=small_reg().fit(X,y)
    assert len(m.get_pareto_front())>=1
    assert len(m.history_)>=1


def test_numba_backend_equivalence_when_available():
    if not NUMBA_AVAILABLE: pytest.skip("Numba unavailable")
    X,y=make_regression(n_samples=70,n_features=3,random_state=8)
    m=small_reg(evaluation_backend="numba").fit(X,y)
    p_numba=m.predict(X[:20])
    m.best_program_.metadata["evaluation_backend"]="numpy"
    p_numpy=m.predict(X[:20])
    assert np.allclose(p_numba,p_numpy,rtol=1e-10,atol=1e-10)


def test_custom_primitive_and_metric():
    name="pow4_acceptance"
    try:
        make_primitive(lambda x: np.asarray(x,float)**4,name,1)
    except ValueError:
        pass
    metric_name="mean_abs_acceptance"
    try:
        make_metric(lambda y,p,w=None: float(np.mean(np.abs(np.asarray(y)-np.asarray(p)))),metric_name)
    except ValueError:
        pass
    X=np.linspace(-1,1,80)[:,None]; y=X[:,0]**4
    m=small_reg(function_set=("add",name),fitness_metric=metric_name,evaluation_backend="auto").fit(X,y)
    assert np.isfinite(m.predict(X[:10])).all()


def test_search_param_beats_preset():
    X,y=make_regression(n_samples=70,n_features=3,random_state=9)
    base=FastSymbolicRegressor(preset="balanced",population_size=500,generations=50,uncertainty="off",verbose=0,random_state=1)
    base.set_params(population_size=12,generations=2)
    base.fit(X,y)
    assert base.population_size==12
    assert base.generations==2
    assert base.preset_skipped_parameters_["population_size"]==12
    assert base.preset_skipped_parameters_["generations"]==2


def test_fused_request_is_real_for_supported_regression():
    X,y=make_regression(n_samples=70,n_features=3,random_state=10)
    m=small_reg(fitness_engine="fused",fitness_metric="mae").fit(X,y)
    assert m.execution_plan_["requested_fitness_engine"]=="fused"
    assert m.execution_plan_["resolved_fitness_engine"]=="fused"
    assert m.execution_plan_["fitness_engine_fallback_reason"] is None


def test_doctor_and_repro_bundle(tmp_path):
    X,y=make_regression(n_samples=80,n_features=3,random_state=11)
    m=small_reg().fit(X,y)
    d=doctor(X,y,m)
    assert d["passed"]
    out=tmp_path/"bundle"; m.export_reproducibility_bundle(out)
    for f in ["config.json","resolved_config.json","manifest.json","expression.txt","environment.json"]:
        assert (out/f).exists()
