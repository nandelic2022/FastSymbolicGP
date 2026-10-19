from __future__ import annotations
import numpy as np
from .pareto import objective_value


def _unique(programs):
    out={}
    for p in programs:
        if p is None: continue
        key=p.structural_hash()
        old=out.get(key)
        if old is None or float(p.validation_loss)<float(old.validation_loss): out[key]=p
    return list(out.values())


def select_final_model(candidates, mode="best_validation", tolerance=0.02, X_validation=None):
    pool=[p for p in _unique(candidates) if bool(p.metadata.get("feasible",True)) and np.isfinite(p.validation_loss)]
    if not pool: pool=_unique(candidates)
    if not pool: raise ValueError("No model candidates are available for final selection")
    best=min(pool,key=lambda p:(p.validation_loss,p.size))
    best_val=float(best.validation_loss); tol=max(0.0,float(tolerance)); threshold=best_val+tol*max(1.0,abs(best_val))
    near=[p for p in pool if float(p.validation_loss)<=threshold]
    mode=str(mode or "best_validation").lower()
    if mode in {"best","best_validation","validation"}: chosen=best
    elif mode in {"smallest","smallest_within_tolerance","interpretable"}:
        chosen=min(near,key=lambda p:(p.size,p.depth,p.validation_loss))
    elif mode in {"pareto_knee","knee"}:
        vals=np.asarray([[float(p.validation_loss),float(objective_value(p,"weighted_complexity"))] for p in pool],float)
        lo=np.nanmin(vals,axis=0); hi=np.nanmax(vals,axis=0); den=np.where(hi>lo,hi-lo,1.0); norm=(vals-lo)/den
        idx=int(np.argmin(np.sqrt(np.sum(norm**2,axis=1)))); chosen=pool[idx]
    elif mode in {"semantic_consensus","consensus"}:
        if X_validation is None or len(near)<2: chosen=min(near,key=lambda p:(p.size,p.validation_loss))
        else:
            pred=np.column_stack([p.predict(X_validation) for p in near]); median=np.median(pred,axis=1)
            d=np.mean((pred-median[:,None])**2,axis=0); chosen=near[int(np.argmin(d))]
    else:
        raise ValueError("model_selection must be best_validation, smallest_within_tolerance, pareto_knee, or semantic_consensus")
    report={
        "mode":mode,"candidate_count":len(pool),"near_optimal_count":len(near),"best_validation_loss":best_val,
        "tolerance":tol,"threshold":threshold,"selected_validation_loss":float(chosen.validation_loss),
        "selected_nodes":int(chosen.size),"selected_depth":int(chosen.depth),"selected_expression":chosen.to_string(),
    }
    return chosen.clone(),report
