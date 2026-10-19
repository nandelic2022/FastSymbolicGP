import numpy as np
import pytest
from sklearn.datasets import make_regression

from fastsymbolicgp import FastSymbolicRegressor, make_metric, make_primitive
from fastsymbolicgp.backend import (
    NUMBA_AVAILABLE,
    score_population_regression_postfix,
    choose_population_batch_size,
)
from fastsymbolicgp.metrics import regression_loss, fit_linear_scaler
from fastsymbolicgp.program import Node, Program

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="V1.1.1 fused tests require Numba")


def _programs():
    return [
        Program(Node("feature", 0)),
        Program(Node("func", "add", (Node("feature", 0), Node("feature", 1)))),
        Program(Node("func", "mul", (Node("feature", 0), Node("feature", 1)))),
        Program(Node("func", "sin", (Node("feature", 2),))),
    ]


@pytest.mark.parametrize("metric", ["mse", "rmse", "mae", "r2", "huber"])
@pytest.mark.parametrize("scaling", ["linear", "none"])
def test_fused_matches_materialized(metric, scaling):
    rng=np.random.default_rng(91)
    X=rng.normal(size=(137,3)); y=2.3*X[:,0]-0.7*X[:,1]+0.4*np.sin(X[:,2])+rng.normal(0,.15,137)
    w=rng.uniform(.2,2.0,size=len(y))
    tr=np.arange(0,97); va=np.arange(97,137)
    programs=_programs()
    a,b,lf,lv=score_population_regression_postfix(
        X[tr],y[tr],X[va],y[va],programs,metric=metric,validation_metric=metric,
        scaling=scaling,sample_weight_train=w[tr],sample_weight_val=w[va],huber_delta=1.25,n_jobs=2,
    )
    for i,p in enumerate(programs):
        raw=p.root.evaluate(X[tr])
        if scaling=="linear": aa,bb=fit_linear_scaler(raw,y[tr],w[tr])
        else: aa,bb=1.0,0.0
        train_pred=aa*raw+bb
        val_pred=aa*p.root.evaluate(X[va])+bb
        lm=regression_loss(y[tr],train_pred,metric,w[tr],huber_delta=1.25)
        vm=regression_loss(y[va],val_pred,metric,w[va],huber_delta=1.25)
        assert a[i] == pytest.approx(aa, rel=1e-10, abs=1e-10)
        assert b[i] == pytest.approx(bb, rel=1e-10, abs=1e-10)
        assert lf[i] == pytest.approx(lm, rel=1e-9, abs=1e-9)
        assert lv[i] == pytest.approx(vm, rel=1e-9, abs=1e-9)


def test_auto_uses_fused_for_supported_regression():
    X,y=make_regression(n_samples=100,n_features=4,random_state=11)
    m=FastSymbolicRegressor(population_size=14,generations=2,max_depth=3,random_state=4,verbose=0,
                            uncertainty="off",fitness_metric="mae",fitness_engine="auto").fit(X,y)
    assert m.execution_plan_["resolved_fitness_engine"] == "fused"
    assert m.best_program_.metadata["fitness_engine"] == "fused"


def test_auto_falls_back_for_custom_metric():
    name="v111_custom_loss"
    try:
        make_metric(lambda y,p,w=None: float(np.average(np.abs(np.asarray(p)-np.asarray(y)),weights=w)), name)
    except ValueError:
        pass
    X,y=make_regression(n_samples=90,n_features=3,random_state=12)
    m=FastSymbolicRegressor(population_size=12,generations=2,max_depth=3,random_state=1,verbose=0,
                            uncertainty="off",fitness_metric=name,fitness_engine="auto").fit(X,y)
    assert m.execution_plan_["resolved_fitness_engine"] == "materialized"
    assert m.execution_plan_["fitness_engine_fallback_reason"]


def test_forced_fused_rejects_custom_metric_and_primitive():
    metric="v111_custom_forced"
    try: make_metric(lambda y,p,w=None: float(np.mean(np.abs(np.asarray(p)-np.asarray(y)))),metric)
    except ValueError: pass
    X,y=make_regression(n_samples=60,n_features=2,random_state=13)
    with pytest.raises(ValueError,match="Forced fused regression is incompatible"):
        FastSymbolicRegressor(population_size=8,generations=1,verbose=0,uncertainty="off",fitness_metric=metric,fitness_engine="fused").fit(X,y)
    prim="v111_pow4"
    try: make_primitive(lambda x: np.asarray(x)**4,prim,1)
    except ValueError: pass
    with pytest.raises(ValueError,match="Forced fused regression is incompatible"):
        FastSymbolicRegressor(population_size=8,generations=1,verbose=0,uncertainty="off",function_set=("add",prim),fitness_engine="fused").fit(X,y)


def test_memory_budget_chunks_population():
    programs=_programs()*20
    chunk=choose_population_batch_size(programs,memory_budget_mb=0.001,n_jobs=2)
    assert 1 <= chunk < len(programs)


def test_checkpoint_rotates_and_recovers(tmp_path):
    X,y=make_regression(n_samples=80,n_features=3,random_state=14)
    m=FastSymbolicRegressor(population_size=10,generations=1,max_depth=3,random_state=2,verbose=0,uncertainty="off").fit(X,y)
    path=tmp_path/"model.chk"
    m.save_checkpoint(path)
    first=m.predict(X[:10]).copy()
    m.best_program_.intercept += 1.0
    m.save_checkpoint(path)
    assert path.with_suffix(path.suffix+".prev").exists()
    path.write_bytes(b"corrupt")
    recovered=FastSymbolicRegressor.load_checkpoint(path)
    assert str(recovered.checkpoint_recovered_from_).endswith(".prev")
    assert np.allclose(recovered.predict(X[:10]),first)
