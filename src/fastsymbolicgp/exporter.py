"""Dependency-light deployment exporters for fitted symbolic models."""
from __future__ import annotations

from pathlib import Path


def _call(name, args, lang):
    maps = {
        "python": {"max": "max", "min": "min", "abs": "abs", "sin": "math.sin", "cos": "math.cos", "tanh": "math.tanh"},
        "c": {"max": "fmax", "min": "fmin", "abs": "fabs", "sin": "sin", "cos": "cos", "tanh": "tanh"},
        "cpp": {"max": "std::max", "min": "std::min", "abs": "std::abs", "sin": "std::sin", "cos": "std::cos", "tanh": "std::tanh"},
        "java": {"max": "Math.max", "min": "Math.min", "abs": "Math.abs", "sin": "Math.sin", "cos": "Math.cos", "tanh": "Math.tanh"},
        "kotlin": {"max": "maxOf", "min": "minOf", "abs": "kotlin.math.abs", "sin": "kotlin.math.sin", "cos": "kotlin.math.cos", "tanh": "kotlin.math.tanh"},
        "javascript": {"max": "Math.max", "min": "Math.min", "abs": "Math.abs", "sin": "Math.sin", "cos": "Math.cos", "tanh": "Math.tanh"},
    }
    return f"{maps[lang][name]}({', '.join(args)})"


def _expr(node, language="python"):
    lang = language.lower()
    if node.kind == "feature":
        return f"x[{int(node.feature)}]"
    if node.kind == "constant":
        return format(float(node.value), ".17g")
    args = [_expr(child, lang) for child in node.children]
    name = str(node.name)
    if name == "add": return f"({args[0]} + {args[1]})"
    if name == "sub": return f"({args[0]} - {args[1]})"
    if name == "mul": return f"({args[0]} * {args[1]})"
    if name == "div": return f"pdiv({args[0]}, {args[1]})"
    if name in {"max", "min"}: return _call(name, args, lang)
    if name == "abs": return _call("abs", args, lang)
    if name == "neg": return f"(-{args[0]})"
    if name == "log": return f"plog({args[0]})"
    if name == "sqrt": return f"psqrt({args[0]})"
    if name == "exp": return f"pexp({args[0]})"
    if name == "inv": return f"pinv({args[0]})"
    if name == "is_missing": return f"is_missing({args[0]})"
    if name == "coalesce": return f"coalesce({args[0]}, {args[1]})"
    if name in {"sin", "cos", "tanh"}: return _call(name, args, lang)
    raise ValueError(f"Unsupported primitive for export: {name}")


def _python_helpers():
    return '''import math
import numpy as np

def finite(v):
    if math.isnan(v): return 0.0
    if math.isinf(v): return 1e6 if v > 0 else -1e6
    return max(-1e6, min(1e6, v))
def pdiv(a,b): return 1.0 if abs(b) <= 1e-12 else finite(a/b)
def plog(a): return finite(math.log(abs(a)+1e-12))
def psqrt(a): return finite(math.sqrt(abs(a)))
def pexp(a): return finite(math.exp(max(-30.0,min(30.0,a))))
def pinv(a): return 0.0 if abs(a) <= 1e-12 else finite(1.0/a)
def is_missing(a): return 1.0 if math.isnan(a) else 0.0
def coalesce(a,b): return b if math.isnan(a) else a
def sigmoid(a):
    a=max(-35.0,min(35.0,a)); return 1.0/(1.0+math.exp(-a))
def softmax(values):
    m=max(values); z=[math.exp(max(-50.0,min(50.0,v-m))) for v in values]; s=sum(z); return [v/s for v in z]
'''


def _python_calibration_lines(model, variable="p", indent="    "):
    calibration = getattr(model, "calibration_model_", None)
    if calibration is None:
        return []
    name = calibration.__class__.__name__
    if name == "PlattCalibrator":
        coef = float(calibration.model_.coef_[0, 0]); intercept = float(calibration.model_.intercept_[0])
        return [f"{indent}{variable} = sigmoid(({coef!r})*{variable} + ({intercept!r}))"]
    if name == "BetaCalibrator":
        coef = calibration.model_.coef_[0]; intercept = float(calibration.model_.intercept_[0])
        return [
            f"{indent}lp=math.log(max({variable},1e-12)); lq=-math.log(max(1.0-{variable},1e-12))",
            f"{indent}{variable} = sigmoid(({float(coef[0])!r})*lp + ({float(coef[1])!r})*lq + ({intercept!r}))",
        ]
    if name == "IsotonicRegression":
        xs = calibration.X_thresholds_.tolist(); ys = calibration.y_thresholds_.tolist()
        return [f"{indent}{variable} = float(np.interp({variable}, {xs!r}, {ys!r}))"]
    raise ValueError(f"Unsupported calibration model for Python export: {name}")


def _python_binary_function(model, name="binary_probability"):
    lines = []
    for i, program in enumerate(model.ensemble_programs_):
        lines.append(f"    raw_{i} = finite({_expr(program.root, 'python')})")
        lines.append(f"    member_{i} = sigmoid(({program.scale_!r}) * raw_{i} + ({program.intercept_!r}))")
    weighted = " + ".join(f"({float(w)!r})*member_{i}" for i, w in enumerate(model.ensemble_weights_))
    lines.append(f"    p = max(1e-12, min(1.0-1e-12, {weighted}))")
    lines.extend(_python_calibration_lines(model))
    lines.append("    return max(1e-12, min(1.0-1e-12, p))")
    return f"def {name}(x):\n" + "\n".join(lines) + "\n"


def _export_python(model):
    helpers = _python_helpers()
    # Regressor
    if not hasattr(model, "classes_"):
        expression = _expr(model.best_program_.root, "python")
        return helpers + f'''\nSCALE={float(model.scale_)!r}\nINTERCEPT={float(model.intercept_)!r}\ndef predict_one(x): return SCALE*finite({expression})+INTERCEPT\ndef predict(X): return np.asarray([predict_one(row) for row in X],dtype=float)\n'''
    # Multiclass classifier
    if getattr(model, "_is_multiclass_", False):
        if getattr(model, "multiclass_strategy_", "ovr") == "shared_softmax":
            lines = [helpers]
            for i, program in enumerate(model.shared_programs_):
                lines.append(f"def shared_{i}(x): return finite({_expr(program.root, 'python')})\n")
            mean = model.shared_feature_mean_.tolist(); scale = model.shared_feature_scale_.tolist()
            coef = model.softmax_model_.coef_.tolist(); intercept = model.softmax_model_.intercept_.tolist()
            temperature = 1.0 if getattr(model, "multiclass_calibrator_", None) is None else float(model.multiclass_calibrator_.temperature_)
            lines.append(f"MEAN={mean!r}\nSCALE={scale!r}\nCOEF={coef!r}\nINTERCEPT={intercept!r}\nTEMPERATURE={temperature!r}\nCLASSES={model.classes_.tolist()!r}\n")
            lines.append("def predict_proba_one(x):\n")
            lines.append("    z=[(globals()[f'shared_{i}'](x)-MEAN[i])/SCALE[i] for i in range(len(MEAN))]\n")
            lines.append("    logits=[INTERCEPT[k]+sum(COEF[k][j]*z[j] for j in range(len(z))) for k in range(len(COEF))]\n")
            lines.append("    return softmax([v/TEMPERATURE for v in logits])\n")
        else:
            lines = [helpers]
            for i, estimator in enumerate(model.estimators_):
                lines.append(_python_binary_function(estimator, f"class_probability_{i}"))
            temperature = 1.0 if getattr(model, "multiclass_calibrator_", None) is None else float(model.multiclass_calibrator_.temperature_)
            lines.append(f"CLASSES={model.classes_.tolist()!r}\nTEMPERATURE={temperature!r}\n")
            lines.append("def predict_proba_one(x):\n")
            lines.append(f"    scores=[class_probability_{i}(x) for i in range({len(model.estimators_)})]\n")
            lines.append("    total=sum(scores)\n    base=[1.0/len(scores) for _ in scores] if total<=1e-15 else [v/total for v in scores]\n")
            lines.append("    return softmax([math.log(max(v,1e-12))/TEMPERATURE for v in base])\n")
        lines.append("def predict_one(x):\n    p=predict_proba_one(x); return CLASSES[max(range(len(p)), key=lambda i:p[i])]\n")
        lines.append("def predict_proba(X): return np.asarray([predict_proba_one(row) for row in X], dtype=float)\n")
        lines.append("def predict(X): return np.asarray([predict_one(row) for row in X])\n")
        return "\n".join(lines)
    # Binary classifier
    lines = [helpers, _python_binary_function(model)]
    lines.append(f"CLASSES={model.classes_.tolist()!r}\nTHRESHOLD={float(getattr(model, 'decision_threshold_', 0.5))!r}\n")
    lines.append("def predict_proba_one(x):\n    p=binary_probability(x); return [1.0-p,p]\n")
    lines.append("def predict_one(x):\n    return CLASSES[1] if binary_probability(x)>=THRESHOLD else CLASSES[0]\n")
    lines.append("def predict_proba(X): return np.asarray([predict_proba_one(row) for row in X], dtype=float)\n")
    lines.append("def predict(X): return np.asarray([predict_one(row) for row in X])\n")
    return "\n".join(lines)


def _calibration_info(model):
    calibration = getattr(model, "calibration_model_", None)
    if calibration is None:
        return None
    name = calibration.__class__.__name__
    if name == "PlattCalibrator":
        return ("platt", float(calibration.model_.coef_[0, 0]), float(calibration.model_.intercept_[0]))
    if name == "BetaCalibrator":
        coef = calibration.model_.coef_[0]
        return ("beta", float(coef[0]), float(coef[1]), float(calibration.model_.intercept_[0]))
    raise NotImplementedError(f"Portable export does not support {name}; use export_python")


def _portable_p_update(model, lang, indent="  "):
    info = _calibration_info(model)
    if info is None:
        return []
    if info[0] == "platt":
        return [f"{indent}p=sigmoid(({info[1]:.17g})*p+({info[2]:.17g}));"]
    if lang in {"c", "cpp"}:
        return [f"{indent}p=sigmoid(({info[1]:.17g})*log(fmax(p,1e-12))+({info[2]:.17g})*(-log(fmax(1.0-p,1e-12)))+({info[3]:.17g}));"]
    if lang == "java":
        return [f"{indent}p=sigmoid(({info[1]:.17g})*Math.log(Math.max(p,1e-12))+({info[2]:.17g})*(-Math.log(Math.max(1.0-p,1e-12)))+({info[3]:.17g}));"]
    if lang == "kotlin":
        return [f"{indent}p=sigmoid(({info[1]:.17g})*kotlin.math.ln(maxOf(p,1e-12))+({info[2]:.17g})*(-kotlin.math.ln(maxOf(1.0-p,1e-12)))+({info[3]:.17g}))"]
    return [f"{indent}p=sigmoid(({info[1]:.17g})*Math.log(Math.max(p,1e-12))+({info[2]:.17g})*(-Math.log(Math.max(1-p,1e-12)))+({info[3]:.17g}));"]


def _portable_source(model, language):
    lang = language.lower()
    if getattr(model, "_is_multiclass_", False):
        raise NotImplementedError(f"{lang} export supports binary classifiers and regressors; use export_python for multiclass")
    is_regressor = not hasattr(model, "classes_")
    programs = [model.best_program_] if is_regressor else model.ensemble_programs_
    expressions = [_expr(program.root, lang) for program in programs]
    if lang in {"c", "cpp"}:
        header = "#include <cmath>\n#include <algorithm>\nusing std::isnan; using std::isinf; using std::fmax; using std::fmin; using std::fabs; using std::log; using std::sqrt; using std::exp;\n" if lang == "cpp" else "#include <math.h>\n"
        h = header + '''static double finite_v(double v){if(isnan(v))return 0.0;if(isinf(v))return v>0?1e6:-1e6;return fmax(-1e6,fmin(1e6,v));}
static double pdiv(double a,double b){return fabs(b)<=1e-12?1.0:finite_v(a/b);} static double plog(double a){return finite_v(log(fabs(a)+1e-12));}
static double psqrt(double a){return finite_v(sqrt(fabs(a)));} static double pexp(double a){return finite_v(exp(fmax(-30.0,fmin(30.0,a))));}
static double pinv(double a){return fabs(a)<=1e-12?0.0:finite_v(1.0/a);} static double is_missing(double a){return isnan(a)?1.0:0.0;}
static double coalesce(double a,double b){return isnan(a)?b:a;} static double sigmoid(double a){a=fmax(-35.0,fmin(35.0,a));return 1.0/(1.0+exp(-a));}
'''
        if is_regressor:
            return h + f"double predict(const double* x){{return ({float(model.scale_):.17g})*finite_v({expressions[0]})+({float(model.intercept_):.17g});}}\n"
        body=[h,"double predict_probability(const double* x){"]
        for i,(p,e) in enumerate(zip(programs,expressions)): body.append(f"  double m{i}=sigmoid(({p.scale_:.17g})*finite_v({e})+({p.intercept_:.17g}));")
        body.append("  double p="+"+".join(f"({float(w):.17g})*m{i}" for i,w in enumerate(model.ensemble_weights_))+";")
        body.extend(_portable_p_update(model,lang))
        body.append("  return fmax(1e-12,fmin(1.0-1e-12,p));\n}")
        body.append(f"int predict(const double* x){{return predict_probability(x)>={float(model.decision_threshold_):.17g}?1:0;}}")
        return "\n".join(body)
    if lang == "java":
        h='''public final class FastSymbolicModel {
  private static double finite(double v){if(Double.isNaN(v))return 0.0;if(Double.isInfinite(v))return v>0?1e6:-1e6;return Math.max(-1e6,Math.min(1e6,v));}
  private static double pdiv(double a,double b){return Math.abs(b)<=1e-12?1.0:finite(a/b);} private static double plog(double a){return finite(Math.log(Math.abs(a)+1e-12));}
  private static double psqrt(double a){return finite(Math.sqrt(Math.abs(a)));} private static double pexp(double a){return finite(Math.exp(Math.max(-30,Math.min(30,a))));}
  private static double pinv(double a){return Math.abs(a)<=1e-12?0.0:finite(1.0/a);} private static double is_missing(double a){return Double.isNaN(a)?1.0:0.0;}
  private static double coalesce(double a,double b){return Double.isNaN(a)?b:a;} private static double sigmoid(double a){a=Math.max(-35,Math.min(35,a));return 1.0/(1.0+Math.exp(-a));}
'''
        if is_regressor:
            return h+f"  public static double predict(double[] x){{return ({float(model.scale_):.17g})*finite({expressions[0]})+({float(model.intercept_):.17g});}}\n}}\n"
        body=[h,"  public static double predictProbability(double[] x){"]
        for i,(p,e) in enumerate(zip(programs,expressions)): body.append(f"    double m{i}=sigmoid(({p.scale_:.17g})*finite({e})+({p.intercept_:.17g}));")
        body.append("    double p="+"+".join(f"({float(w):.17g})*m{i}" for i,w in enumerate(model.ensemble_weights_))+";")
        body.extend(_portable_p_update(model,lang,"    "))
        body.append("    return Math.max(1e-12,Math.min(1.0-1e-12,p));\n  }")
        body.append(f"  public static int predict(double[] x){{return predictProbability(x)>={float(model.decision_threshold_):.17g}?1:0;}}\n}}")
        return "\n".join(body)
    if lang == "kotlin":
        h='''object FastSymbolicModel {
  private fun finite(v:Double)=when{v.isNaN()->0.0;v==Double.POSITIVE_INFINITY->1e6;v==Double.NEGATIVE_INFINITY->-1e6;else->v.coerceIn(-1e6,1e6)}
  private fun pdiv(a:Double,b:Double)=if(kotlin.math.abs(b)<=1e-12)1.0 else finite(a/b)
  private fun plog(a:Double)=finite(kotlin.math.ln(kotlin.math.abs(a)+1e-12)); private fun psqrt(a:Double)=finite(kotlin.math.sqrt(kotlin.math.abs(a)))
  private fun pexp(a:Double)=finite(kotlin.math.exp(a.coerceIn(-30.0,30.0))); private fun pinv(a:Double)=if(kotlin.math.abs(a)<=1e-12)0.0 else finite(1.0/a)
  private fun is_missing(a:Double)=if(a.isNaN())1.0 else 0.0; private fun coalesce(a:Double,b:Double)=if(a.isNaN())b else a
  private fun sigmoid(a0:Double):Double{val a=a0.coerceIn(-35.0,35.0);return 1.0/(1.0+kotlin.math.exp(-a))}
'''
        if is_regressor:
            return h+f"  fun predict(x:DoubleArray)=({float(model.scale_):.17g})*finite({expressions[0]})+({float(model.intercept_):.17g})\n}}\n"
        body=[h,"  fun predictProbability(x:DoubleArray):Double{"]
        for i,(p,e) in enumerate(zip(programs,expressions)): body.append(f"    val m{i}=sigmoid(({p.scale_:.17g})*finite({e})+({p.intercept_:.17g}))")
        body.append("    var p="+"+".join(f"({float(w):.17g})*m{i}" for i,w in enumerate(model.ensemble_weights_)))
        body.extend(_portable_p_update(model,lang,"    "))
        body.append("    return p.coerceIn(1e-12,1.0-1e-12)\n  }")
        body.append(f"  fun predict(x:DoubleArray)=if(predictProbability(x)>={float(model.decision_threshold_):.17g})1 else 0\n}}")
        return "\n".join(body)
    if lang == "javascript":
        h='''const finite=v=>Number.isNaN(v)?0:(v===Infinity?1e6:(v===-Infinity?-1e6:Math.max(-1e6,Math.min(1e6,v))));
const pdiv=(a,b)=>Math.abs(b)<=1e-12?1:finite(a/b),plog=a=>finite(Math.log(Math.abs(a)+1e-12)),psqrt=a=>finite(Math.sqrt(Math.abs(a))),pexp=a=>finite(Math.exp(Math.max(-30,Math.min(30,a)))),pinv=a=>Math.abs(a)<=1e-12?0:finite(1/a),is_missing=a=>Number.isNaN(a)?1:0,coalesce=(a,b)=>Number.isNaN(a)?b:a,sigmoid=a0=>{const a=Math.max(-35,Math.min(35,a0));return 1/(1+Math.exp(-a));};
'''
        if is_regressor:
            return h+f"function predict(x){{return ({float(model.scale_):.17g})*finite({expressions[0]})+({float(model.intercept_):.17g});}}\nmodule.exports={{predict}};\n"
        body=[h,"function predictProbability(x){"]
        for i,(p,e) in enumerate(zip(programs,expressions)): body.append(f"  const m{i}=sigmoid(({p.scale_:.17g})*finite({e})+({p.intercept_:.17g}));")
        body.append("  let p="+"+".join(f"({float(w):.17g})*m{i}" for i,w in enumerate(model.ensemble_weights_))+";")
        body.extend(_portable_p_update(model,lang))
        body.append("  return Math.max(1e-12,Math.min(1-1e-12,p));\n}")
        body.append(f"function predict(x){{return predictProbability(x)>={float(model.decision_threshold_):.17g}?1:0;}}\nmodule.exports={{predictProbability,predict}};")
        return "\n".join(body)
    raise ValueError(f"Unsupported export language: {lang}")


def export_model(model, path, language="python"):
    language = str(language).lower()
    source = _export_python(model) if language == "python" else _portable_source(model, language)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source + "\n", encoding="utf-8")
    return str(path)
