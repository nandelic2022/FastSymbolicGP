from __future__ import annotations
import os, platform, sys, json
import numpy as np
from ._version import __version__
from .primitives import available_primitives

def doctor(X=None,y=None,estimator=None):
    checks=[]
    def add(name,status,detail=""): checks.append({"check":name,"status":status,"detail":detail})
    add("FastSymbolicGP version","PASS",__version__); add("Python","PASS",platform.python_version()); add("CPU threads","PASS",str(os.cpu_count() or 1))
    try:
        import sklearn; add("scikit-learn","PASS",sklearn.__version__)
    except Exception as e: add("scikit-learn","FAIL",repr(e))
    try:
        import numba; add("Numba","PASS",numba.__version__)
    except Exception: add("Numba","WARN","optional acceleration dependency not installed")
    if X is not None:
        A=np.asarray(X,float); add("X finite","PASS" if np.isfinite(A).all() else "FAIL",f"shape={A.shape}")
        const=int(np.sum(np.nanstd(A,axis=0)<1e-14)); add("constant features","WARN" if const else "PASS",str(const));
        if A.shape[0]<200: add("sample size","WARN",f"only {A.shape[0]} samples")
        else: add("sample size","PASS",str(A.shape[0]))
    if y is not None:
        yy=np.asarray(y); add("target finite","PASS" if np.isfinite(yy.astype(float)).all() else "FAIL",f"n={len(yy)}")
    if estimator is not None:
        try: estimator.get_params(deep=False); add("sklearn get_params","PASS")
        except Exception as e: add("sklearn get_params","FAIL",repr(e))
        try:
            from sklearn.base import clone; clone(estimator); add("sklearn clone","PASS")
        except Exception as e: add("sklearn clone","FAIL",repr(e))
        if hasattr(estimator,"validate_configuration"):
            try: estimator.validate_configuration(); add("configuration","PASS")
            except Exception as e: add("configuration","FAIL",str(e))
    return {"version":__version__,"checks":checks,"passed":all(c["status"]!="FAIL" for c in checks),"available_primitives":available_primitives()}

def self_test(verbose=True):
    from sklearn.datasets import load_diabetes, load_breast_cancer
    from sklearn.base import clone
    from .regressor import FastSymbolicRegressor
    from .classifier import FastSymbolicClassifier
    results=[]
    try:
        X,y=load_diabetes(return_X_y=True); m=FastSymbolicRegressor(population_size=12,generations=2,max_depth=3,random_state=1,verbose=0,uncertainty="off").fit(X[:120],y[:120]); results.append(("regression",np.isfinite(m.predict(X[:10])).all()))
    except Exception as e: results.append(("regression",False,repr(e)))
    try:
        X,y=load_breast_cancer(return_X_y=True); m=FastSymbolicClassifier(population_size=12,generations=2,max_depth=3,random_state=1,verbose=0).fit(X[:160],y[:160]); results.append(("classification",np.isfinite(m.predict_proba(X[:10])).all()))
    except Exception as e: results.append(("classification",False,repr(e)))
    try: clone(FastSymbolicRegressor()); results.append(("sklearn_clone",True))
    except Exception as e: results.append(("sklearn_clone",False,repr(e)))
    try:
        from .backend import NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            X,y=load_diabetes(return_X_y=True)
            m=FastSymbolicRegressor(population_size=10,generations=1,max_depth=3,random_state=2,verbose=0,uncertainty="off",fitness_metric="mae",fitness_engine="fused").fit(X[:100],y[:100])
            results.append(("fused_regression",m.execution_plan_.get("resolved_fitness_engine")=="fused"))
        else:
            results.append(("fused_regression",True,"SKIP: Numba unavailable"))
    except Exception as e: results.append(("fused_regression",False,repr(e)))
    out={"version":__version__,"passed":all(r[1] for r in results),"results":results}
    if verbose: print(json.dumps(out,indent=2,default=str))
    return out
