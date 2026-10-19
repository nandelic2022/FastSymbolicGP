from __future__ import annotations
import numpy as np
from .primitives import get_primitive
from .program import infer_unit
from .units import parse_unit, unit_to_string


def structural_constraint_penalty(program, constraints, feature_names=None):
    if not constraints: return 0.0, []
    root=program.root; funcs=root.function_names(); feats=root.feature_indices(); issues=[]; penalty=0.0
    forbidden=set(constraints.get("forbidden_operators",()))
    bad=forbidden.intersection(funcs)
    if bad: issues.append(f"forbidden operators: {sorted(bad)}"); penalty += 1e6*len(bad)
    mandatory=constraints.get("mandatory_variables",())
    names = [] if feature_names is None else list(feature_names)
    name_to_idx={str(n):i for i,n in enumerate(names)}
    required=[]
    for v in mandatory:
        required.append(name_to_idx[v] if isinstance(v,str) and v in name_to_idx else int(v) if not isinstance(v,str) or str(v).isdigit() else -1)
    missing=[i for i in required if i not in feats]
    if missing: issues.append(f"missing variables: {missing}"); penalty += 1e6*len(missing)
    max_vars=constraints.get("max_variables")
    if max_vars is not None and len(set(feats))>int(max_vars):
        issues.append("too many variables"); penalty += 1e5*(len(set(feats))-int(max_vars))
    max_nl=constraints.get("max_nonlinear_operators")
    if max_nl is not None:
        nl=sum(get_primitive(f).nonlinear for f in funcs)
        if nl>int(max_nl): issues.append("too many nonlinear operators"); penalty += 1e4*(nl-int(max_nl))
    max_depth=constraints.get("max_depth")
    if max_depth is not None and program.depth>int(max_depth):
        issues.append("constraint max_depth violated"); penalty += 1e6*(program.depth-int(max_depth))
    return float(penalty), issues


def unit_constraint_penalty(program, feature_units, target_unit):
    if not feature_units or target_unit is None: return 0.0, None
    try:
        fu=[parse_unit(u) for u in feature_units]; expected=parse_unit(target_unit); got=infer_unit(program.root,fu)
        if got!=expected:
            return 1e8, f"unit mismatch: got {unit_to_string(got)}, expected {unit_to_string(expected)}"
        return 0.0, None
    except Exception as exc:
        return 1e8, str(exc)


def _feature_index(key, name_to_idx, n_features):
    j=name_to_idx.get(str(key),int(key) if str(key).isdigit() else None)
    return j if j is not None and 0<=j<n_features else None


def _latin_hypercube(rng, n, low, high):
    n=max(1,int(n)); d=len(low); out=np.empty((n,d),float)
    for j in range(d):
        edges=(np.arange(n)+rng.random(n))/n
        rng.shuffle(edges)
        out[:,j]=low[j]+edges*(high[j]-low[j])
    return out


def build_constraint_points(X, feature_names, constraints, *, max_points=128):
    """Build deterministic real + virtual points used for scientific constraints.

    Supported configuration keys:
      - virtual_points: explicit full rows or partial {feature: value} mappings.
      - virtual_ranges: {feature: [low, high]} ranges; unspecified features use
        observed finite min/max.
      - virtual_n_points: number of LHS/uniform virtual samples (default 0).
      - virtual_strategy: "latin_hypercube"/"lhs" or "uniform".
      - virtual_seed: deterministic seed (default 0).
    """
    constraints=constraints or {}; X=np.asarray(X,float)
    names=[f"X{i}" for i in range(X.shape[1])] if feature_names is None else list(feature_names)
    name_to_idx={str(n):i for i,n in enumerate(names)}
    if X.shape[0]>int(max_points):
        idx=np.linspace(0,X.shape[0]-1,int(max_points),dtype=int); base=X[idx].copy()
    else:
        base=X.copy()
    med=np.nanmedian(X,axis=0); med=np.where(np.isfinite(med),med,0.0)
    explicit=[]
    for row in constraints.get("virtual_points",()) or ():
        if isinstance(row,dict):
            z=med.copy()
            for k,v in row.items():
                j=_feature_index(k,name_to_idx,X.shape[1])
                if j is not None: z[j]=float(v)
            explicit.append(z)
        else:
            z=np.asarray(row,float).reshape(-1)
            if len(z)!=X.shape[1]: raise ValueError("Each virtual point must have n_features values")
            explicit.append(z)
    nvirt=int(constraints.get("virtual_n_points",0) or 0)
    generated=[]
    if nvirt>0:
        lo=np.nanmin(X,axis=0); hi=np.nanmax(X,axis=0)
        lo=np.where(np.isfinite(lo),lo,med); hi=np.where(np.isfinite(hi),hi,med)
        for k,bounds in (constraints.get("virtual_ranges",{}) or {}).items():
            j=_feature_index(k,name_to_idx,X.shape[1])
            if j is not None:
                lo[j],hi[j]=map(float,bounds)
        # Avoid zero-width dimensions without inventing a large extrapolation range.
        hi=np.where(hi>lo,hi,lo+1e-12)
        rng=np.random.default_rng(int(constraints.get("virtual_seed",0) or 0))
        strategy=str(constraints.get("virtual_strategy","latin_hypercube")).lower()
        if strategy in {"latin_hypercube","lhs","latin"}: generated=_latin_hypercube(rng,nvirt,lo,hi)
        elif strategy=="uniform": generated=rng.uniform(lo,hi,size=(nvirt,X.shape[1]))
        else: raise ValueError("virtual_strategy must be latin_hypercube/lhs or uniform")
    pieces=[base]
    if explicit: pieces.append(np.vstack(explicit))
    if len(generated): pieces.append(np.asarray(generated,float))
    return np.ascontiguousarray(np.vstack(pieces),dtype=float)


def behavioral_constraint_penalty(program, X, feature_names, constraints, *, max_points=128, return_report=False):
    if not constraints:
        return (0.0, [], {"points":0,"violations":0}) if return_report else (0.0, [])
    X=np.asarray(X,float); Z=build_constraint_points(X,feature_names,constraints,max_points=max_points)
    issues=[]; penalty=0.0; details={"points":int(len(Z)),"virtual_points":int(max(0,len(Z)-min(len(X),max_points))),"violations":0,"checks":{}}
    pred=program.predict(Z)
    omin=constraints.get("output_min"); omax=constraints.get("output_max")
    tol=float(constraints.get("output_tolerance",1e-9))
    if omin is not None:
        v=np.maximum(float(omin)-pred,0); amount=float(np.mean(v)); bad=int(np.sum(v>tol)); penalty += 1e3*amount
        details["checks"]["output_min"]={"violations":bad,"max_violation":float(np.max(v,initial=0.0))}
        if bad: issues.append("output_min violated"); details["violations"]+=bad
    if omax is not None:
        v=np.maximum(pred-float(omax),0); amount=float(np.mean(v)); bad=int(np.sum(v>tol)); penalty += 1e3*amount
        details["checks"]["output_max"]={"violations":bad,"max_violation":float(np.max(v,initial=0.0))}
        if bad: issues.append("output_max violated"); details["violations"]+=bad
    monotonic=constraints.get("monotonic",{}) or {}
    names = [f"X{i}" for i in range(X.shape[1])] if feature_names is None else list(feature_names)
    name_to_idx={str(n):i for i,n in enumerate(names)}
    mtol=float(constraints.get("monotonic_tolerance",1e-9))
    for key,direction in monotonic.items():
        j=_feature_index(key,name_to_idx,X.shape[1])
        if j is None: continue
        scale=max(float(np.nanstd(Z[:,j])),1.0); eps=float(constraints.get("derivative_step",1e-4))*scale
        Zp=Z.copy(); Zm=Z.copy(); Zp[:,j]+=eps; Zm[:,j]-=eps
        d=(program.predict(Zp)-program.predict(Zm))/(2*eps)
        if str(direction).lower() in {"increasing","positive","+","nondecreasing"}: v=np.maximum(-d,0)
        elif str(direction).lower() in {"decreasing","negative","-","nonincreasing"}: v=np.maximum(d,0)
        else: raise ValueError(f"Unknown monotonic direction {direction!r}")
        bad=int(np.sum(v>mtol)); penalty += 1e3*float(np.mean(v))
        details["checks"][f"monotonic:{key}"]={"violations":bad,"min_derivative":float(np.min(d)),"max_derivative":float(np.max(d))}
        if bad: issues.append(f"monotonicity violated for {key}"); details["violations"]+=bad
    # Optional physics-informed residual constraints. A residual callable may
    # accept (X, prediction) or (X, prediction, program) and should return a
    # scalar or one residual per constraint point.
    rtol_default=float(constraints.get("residual_tolerance",1e-6))
    for index,rc in enumerate(constraints.get("residuals",()) or ()):
        func=rc.get("function") if isinstance(rc,dict) else rc
        if not callable(func): raise ValueError("Each residual constraint needs a callable 'function'")
        try: residual=func(Z,pred,program)
        except TypeError: residual=func(Z,pred)
        residual=np.asarray(residual,float)
        if residual.ndim==0: residual=np.full(len(Z),float(residual))
        residual=np.abs(residual.reshape(-1))
        if residual.size not in {1,len(Z)}: raise ValueError("Residual function must return a scalar or n_constraint_points values")
        tol=float(rc.get("tolerance",rtol_default)) if isinstance(rc,dict) else rtol_default
        weight=float(rc.get("weight",1.0)) if isinstance(rc,dict) else 1.0
        bad=int(np.sum(residual>tol)); amount=float(np.mean(residual)); penalty += 1e4*weight*amount
        details["checks"][f"residual:{index}"]={"violations":bad,"mean_abs_residual":amount,"max_abs_residual":float(np.max(residual,initial=0.0)),"tolerance":tol}
        if bad: issues.append(f"physics residual {index} violated"); details["violations"]+=bad

    btol=float(constraints.get("boundary_tolerance",1e-6))
    for index,bc in enumerate(constraints.get("boundary",()) or ()):
        row=np.nanmedian(X,axis=0).astype(float); when=bc.get("when",{})
        for k,v in when.items():
            j=_feature_index(k,name_to_idx,X.shape[1])
            if j is not None: row[j]=float(v)
        target=float(bc["output"]); actual=float(program.predict(row[None,:])[0]); err=abs(actual-target); penalty += 1e4*err
        bad=int(err>float(bc.get("tolerance",btol)))
        details["checks"][f"boundary:{index}"]={"violations":bad,"error":float(err),"actual":actual,"target":target}
        if bad: issues.append(f"boundary condition {index} violated"); details["violations"]+=bad
    if return_report: return float(penalty), issues, details
    return float(penalty), issues
