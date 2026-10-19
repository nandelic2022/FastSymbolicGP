from __future__ import annotations
import json, math
from pathlib import Path
from ._version import __version__
from .program import Program, Node


def _latex_node(n,names):
    if n.kind=="feature": return str(names[int(n.value)]).replace("_","\\_")
    if n.kind=="const": return f"{float(n.value):.6g}"
    a=[_latex_node(c,names) for c in n.children]; op=n.value
    if op=="add": return f"\\left({a[0]}+{a[1]}\\right)"
    if op=="sub": return f"\\left({a[0]}-{a[1]}\\right)"
    if op=="mul": return f"\\left({a[0]}\\,{a[1]}\\right)"
    if op=="div": return f"\\frac{{{a[0]}}}{{{a[1]}}}"
    if op=="square": return f"\\left({a[0]}\\right)^2"
    if op=="cube": return f"\\left({a[0]}\\right)^3"
    if op=="sqrt": return f"\\sqrt{{{a[0]}}}"
    if op=="abs": return f"\\left|{a[0]}\\right|"
    if op=="neg": return f"-{a[0]}"
    nm={"sin":"\\sin","cos":"\\cos","tanh":"\\tanh","log":"\\log","exp":"\\exp","max":"\\max","min":"\\min"}.get(op,op)
    return f"{nm}\\left({','.join(a)}\\right)"

def program_to_latex(program,names):
    core=_latex_node(program.root,names); a=float(program.scale); b=float(program.intercept)
    return f"{a:.6g}\\left[{core}\\right] + {b:.6g}"

def _payload(model):
    p={"format":"FastSymbolicGP-safe-json-v2","version":__version__,"estimator":model.__class__.__name__,"feature_names":[str(x) for x in model._symbolic_feature_names_],"n_features_in":int(model.n_features_in_),"params":{k:v for k,v in model.get_params(deep=False).items() if isinstance(v,(str,int,float,bool,type(None),list,tuple,dict))},"manifest":model.model_manifest()}
    if getattr(model,"_is_multiclass_",False):
        p["classes"]=model.classes_.tolist(); p["models"]=[e.best_program_.to_dict() for e in model.estimators_]; p["thresholds"]=[getattr(e,"decision_threshold_",.5) for e in model.estimators_]
    else:
        p["program"]=model.best_program_.to_dict(); p["classes"]=getattr(model,"classes_",None).tolist() if hasattr(getattr(model,"classes_",None),"tolist") else None; p["decision_threshold"]=getattr(model,"decision_threshold_",None); p["conformal_scores"]=getattr(model,"conformal_scores_",None).tolist() if hasattr(getattr(model,"conformal_scores_",None),"tolist") else None
    return p

def save_model_json(model,path):
    path=Path(path); path.write_text(json.dumps(_payload(model),indent=2,default=str),encoding="utf-8"); return str(path)

def load_model_json(path,expected_class=None):
    d=json.loads(Path(path).read_text(encoding="utf-8"));
    if d.get("format")!="FastSymbolicGP-safe-json-v2": raise ValueError("Unsupported FastSymbolicGP JSON format")
    from .regressor import FastSymbolicRegressor
    from .classifier import FastSymbolicClassifier
    cls=expected_class or ({"FastSymbolicRegressor":FastSymbolicRegressor,"FastSymbolicClassifier":FastSymbolicClassifier,"FastSymbolicMultiClassifier":FastSymbolicClassifier}.get(d["estimator"]))
    if cls is None: raise ValueError(f"Unsupported estimator {d['estimator']}")
    allowed=cls().get_params(deep=False); params={k:v for k,v in d.get("params",{}).items() if k in allowed}; m=cls(**params); import numpy as np
    m._symbolic_feature_names_=np.asarray(d["feature_names"],dtype=object); m.n_features_in_=int(d["n_features_in"]); m.resolved_config_=d.get("manifest",{}).get("resolved_config",{}); m.profile_={"loaded_from_safe_json":True,"version":d.get("version")}; m.history_=[]; m.pareto_archive_=[]
    if "models" in d:
        m._is_multiclass_=True; m.classes_=np.asarray(d["classes"]); m.estimators_=[]
        for c,pd,t in zip(m.classes_,d["models"],d.get("thresholds",[])):
            e=FastSymbolicClassifier(**params); e._symbolic_feature_names_=m._symbolic_feature_names_; e.n_features_in_=m.n_features_in_; e.classes_=np.asarray([0,1]); e.best_program_=Program.from_dict(pd); e.decision_threshold_=float(t); e._is_multiclass_=False; e.resolved_config_={}; e.profile_={}; e.history_=[]; e.pareto_archive_=[]; m.estimators_.append(e)
        m.best_program_=m.estimators_[0].best_program_
    else:
        m._is_multiclass_=False; m.best_program_=Program.from_dict(d["program"]); m.best_expression_=m.best_program_.to_string(m._symbolic_feature_names_)
        if d.get("classes") is not None: m.classes_=np.asarray(d["classes"])
        if d.get("decision_threshold") is not None: m.decision_threshold_=float(d["decision_threshold"])
        cs=d.get("conformal_scores"); m.conformal_scores_=None if cs is None else np.asarray(cs,float)
    return m
