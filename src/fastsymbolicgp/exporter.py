"""Dependency-light deployment exporters for fitted symbolic models."""
from __future__ import annotations
from pathlib import Path


def _expr(n,lang):
    if n.kind=="feature":return f"x[{int(n.value)}]"
    if n.kind=="const":return format(float(n.value),".17g")
    a=[_expr(c,lang) for c in n.children];op=str(n.value)
    if op=="add":return f"({a[0]} + {a[1]})"
    if op=="sub":return f"({a[0]} - {a[1]})"
    if op=="mul":return f"({a[0]} * {a[1]})"
    if op=="div":return f"pdiv({a[0]}, {a[1]})"
    if op=="neg":return f"(-{a[0]})"
    if op=="square":return f"({a[0]}*{a[0]})"
    if op=="cube":return f"({a[0]}*{a[0]}*{a[0]})"
    if op=="inv":return f"pinv({a[0]})"
    if op=="log":return f"plog({a[0]})"
    if op=="sqrt":return f"psqrt({a[0]})"
    if op=="exp":return f"pexp({a[0]})"
    if op in {"sin","cos","tanh","abs","max","min"}:
        if lang=="python":mp={"sin":"math.sin","cos":"math.cos","tanh":"math.tanh","abs":"abs","max":"max","min":"min"}
        elif lang in {"c","cpp"}:mp={"sin":"sin","cos":"cos","tanh":"tanh","abs":"fabs","max":"fmax","min":"fmin"}
        else:mp={"sin":"Math.sin","cos":"Math.cos","tanh":"Math.tanh","abs":"Math.abs","max":"Math.max","min":"Math.min"}
        return f"{mp[op]}({', '.join(a)})"
    raise ValueError(f"Portable export does not support primitive {op!r}")

def _python(model):
    if getattr(model,"_is_multiclass_",False):raise ValueError("Portable Python exporter currently supports regression and binary classification")
    e=_expr(model.best_program_.root,"python");scale=float(model.best_program_.scale);inter=float(model.best_program_.intercept)
    h='''import math\nimport numpy as np\ndef finite(v):\n    if math.isnan(v): return 0.0\n    if math.isinf(v): return 1e6 if v>0 else -1e6\n    return max(-1e6,min(1e6,v))\ndef pdiv(a,b): return 1.0 if abs(b)<=1e-12 else finite(a/b)\ndef plog(a): return finite(math.log(abs(a)+1e-12))\ndef psqrt(a): return finite(math.sqrt(abs(a)))\ndef pexp(a): return finite(math.exp(max(-30.0,min(30.0,a))))\ndef pinv(a): return 0.0 if abs(a)<=1e-12 else finite(1.0/a)\ndef sigmoid(a):\n    a=max(-35.0,min(35.0,a)); return 1.0/(1.0+math.exp(-a))\n'''
    if hasattr(model,"classes_"):
        if getattr(model,"calibration_model_",None) is not None:raise ValueError("Portable export of calibrated classifiers is not supported; export safe JSON instead")
        th=float(getattr(model,"decision_threshold_",.5));classes=model.classes_.tolist()
        body=f'''\nSCALE={scale!r}\nINTERCEPT={inter!r}\nTHRESHOLD={th!r}\nCLASSES={classes!r}\ndef predict_probability_one(x): return sigmoid(SCALE*finite({e})+INTERCEPT)\ndef predict_one(x): return CLASSES[1] if predict_probability_one(x)>=THRESHOLD else CLASSES[0]\ndef predict(X): return np.asarray([predict_one(row) for row in X])\n'''
    else:
        body=f'''\nSCALE={scale!r}\nINTERCEPT={inter!r}\ndef predict_one(x): return SCALE*finite({e})+INTERCEPT\ndef predict(X): return np.asarray([predict_one(row) for row in X],dtype=float)\n'''
    return h+body

def _c_like(model,lang):
    if getattr(model,"_is_multiclass_",False):raise ValueError("Portable exporter currently supports regression and binary classification")
    if hasattr(model,"classes_") and getattr(model,"calibration_model_",None) is not None:raise ValueError("Portable export of calibrated classifiers is not supported")
    e=_expr(model.best_program_.root,lang);s=float(model.best_program_.scale);b=float(model.best_program_.intercept)
    if lang in {"c","cpp"}:
        head='#include <math.h>\nstatic double pdiv(double a,double b){return fabs(b)<=1e-12?1.0:a/b;}\nstatic double plog(double a){return log(fabs(a)+1e-12);}\nstatic double psqrt(double a){return sqrt(fabs(a));}\nstatic double pexp(double a){return exp(fmax(-30.0,fmin(30.0,a)));}\nstatic double pinv(double a){return fabs(a)<=1e-12?0.0:1.0/a;}\nstatic double sigmoid(double a){a=fmax(-35.0,fmin(35.0,a));return 1.0/(1.0+exp(-a));}\n'
        if hasattr(model,"classes_"):return head+f'double predict_probability(const double *x){{return sigmoid(({s:.17g})*({e})+({b:.17g}));}}\nint predict(const double *x){{return predict_probability(x)>={float(model.decision_threshold_):.17g}?1:0;}}\n'
        return head+f'double predict(const double *x){{return ({s:.17g})*({e})+({b:.17g});}}\n'
    # JavaScript is useful for Blogger/browser deployment; Java/Kotlin use same Math expression but wrappers differ.
    if lang=="javascript":
        head='const pdiv=(a,b)=>Math.abs(b)<=1e-12?1:a/b,plog=a=>Math.log(Math.abs(a)+1e-12),psqrt=a=>Math.sqrt(Math.abs(a)),pexp=a=>Math.exp(Math.max(-30,Math.min(30,a))),pinv=a=>Math.abs(a)<=1e-12?0:1/a,sigmoid=a=>{a=Math.max(-35,Math.min(35,a));return 1/(1+Math.exp(-a));};\n'
        if hasattr(model,"classes_"):return head+f'function predictProbability(x){{return sigmoid(({s:.17g})*({e})+({b:.17g}));}}\nfunction predict(x){{return predictProbability(x)>={float(model.decision_threshold_):.17g}?1:0;}}\n'
        return head+f'function predict(x){{return ({s:.17g})*({e})+({b:.17g});}}\n'
    raise ValueError(f"Exporter language {lang!r} is not implemented in V1.1.4")

def export_model(model,path,language="python"):
    language=str(language).lower();path=Path(path)
    if language in {"json","safe_json"}:
        return model.save_model_json(path)
    if language in {"latex","tex"}:
        path.write_text(str(model.to_latex())+"\n",encoding="utf-8");return str(path)
    if language=="python":text=_python(model)
    elif language in {"c","cpp","javascript"}:text=_c_like(model,language)
    elif language in {"java","kotlin"}:raise ValueError(f"{language} exporter is temporarily unavailable in V1.1.4; use Python/C/C++/JavaScript or safe JSON")
    else:raise ValueError(f"Unsupported export language {language!r}")
    path.write_text(text,encoding="utf-8");return str(path)
