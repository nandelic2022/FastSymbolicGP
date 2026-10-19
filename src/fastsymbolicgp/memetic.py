from __future__ import annotations
import numpy as np


def constant_paths(program):
    return [path for path in program.root.paths() if program.root.get(path).kind=="const"]


def _vector(program,paths):
    return np.asarray([float(program.root.get(p).value) for p in paths],float)


def _assign(program,paths,values):
    for path,value in zip(paths,values): program.root.get(path).value=float(value)
    program.invalidate_caches()
    return program


def optimize_constants(program, objective, bounds, method="hybrid", rounds=3, budget=200, random_state=0):
    """Derivative-free memetic optimizer.

    ``objective`` receives a cloned program and returns a scalar to minimise.
    Hybrid mode performs deterministic coordinate refinement and, when SciPy is
    available, a bounded Powell pass with the remaining evaluation budget.
    """
    paths=constant_paths(program); budget=max(1,int(budget)); rounds=max(1,int(rounds))
    if not paths:
        return program.clone(),{"enabled":True,"method":str(method),"constants":0,"evaluations":0,"improved":False}
    q=program.clone(); x=_vector(q,paths); bounds=list(bounds)
    evals=0
    def evaluate(v):
        nonlocal evals
        if evals>=budget: return float("inf")
        cand=_assign(q.clone(),paths,v); val=float(objective(cand)); evals+=1; return val
    best=evaluate(x); initial=best; bestx=x.copy()
    spans=np.asarray([max(float(hi)-float(lo),1e-12) for lo,hi in bounds])
    for r in range(rounds):
        if evals>=budget: break
        frac=0.20/(2**r)
        for j,(lo,hi) in enumerate(bounds):
            if evals>=budget: break
            step=max(spans[j]*frac,1e-10)
            candidates=[bestx[j]-step,bestx[j]+step]
            if lo<=0.0<=hi: candidates.append(0.0)
            for value in candidates:
                trial=bestx.copy(); trial[j]=float(np.clip(value,lo,hi)); score=evaluate(trial)
                if score<best-1e-15: best=score; bestx=trial
    used_scipy=False
    mode=str(method).lower()
    if mode in {"hybrid","powell","scipy"} and evals<budget:
        try:
            from scipy.optimize import minimize
            def fun(v): return evaluate(np.asarray(v,float))
            res=minimize(fun,bestx,method="Powell",bounds=bounds,options={"maxfev":max(1,budget-evals),"xtol":1e-5,"ftol":1e-8,"disp":False})
            used_scipy=True
            if np.isfinite(res.fun) and float(res.fun)<best:
                best=float(res.fun); bestx=np.asarray(res.x,float)
        except Exception:
            pass
    out=_assign(q.clone(),paths,bestx)
    return out,{"enabled":True,"method":mode,"scipy_refinement":bool(used_scipy),"constants":len(paths),"evaluations":int(evals),"initial_objective":float(initial),"final_objective":float(best),"improved":bool(best<initial-1e-15)}
