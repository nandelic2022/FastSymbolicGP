from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray


def _finite(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)


def protected_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.ones(np.broadcast_shapes(a.shape, b.shape), dtype=float)
    aa, bb = np.broadcast_arrays(a, b)
    np.divide(aa, bb, out=out, where=np.abs(bb) > 1e-12)
    return _finite(out)


def protected_log(a): return _finite(np.log(np.abs(a) + 1e-12))
def protected_sqrt(a): return _finite(np.sqrt(np.abs(a)))
def protected_exp(a): return _finite(np.exp(np.clip(a, -30.0, 30.0)))
def protected_inv(a):
    a = np.asarray(a, float); out = np.zeros_like(a)
    np.divide(1.0, a, out=out, where=np.abs(a) > 1e-12)
    return _finite(out)
def sigmoid(a):
    a=np.clip(np.asarray(a,float),-35,35); return 1/(1+np.exp(-a))
def softplus(a):
    a=np.asarray(a,float); return _finite(np.log1p(np.exp(-np.abs(a))) + np.maximum(a,0))
def is_missing(a): return np.isnan(np.asarray(a,float)).astype(float)
def coalesce(a,b):
    aa,bb=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float)); return _finite(np.where(np.isnan(aa),bb,aa))

@dataclass(frozen=True)
class Primitive:
    name: str
    arity: int
    function: Callable
    complexity: float = 1.0
    nonlinear: bool = False
    numba_compatible: bool = True

_REGISTRY: dict[str, Primitive] = {}

def register_primitive(name: str, arity: int, function: Callable, *, complexity=1.0,
                       nonlinear=False, numba_compatible=False, overwrite=False) -> Primitive:
    name = str(name)
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Primitive {name!r} already exists")
    p = Primitive(name, int(arity), function, float(complexity), bool(nonlinear), bool(numba_compatible))
    _REGISTRY[name] = p
    return p

def make_primitive(function: Callable, name: str, arity: int, **kwargs) -> Primitive:
    return register_primitive(name, arity, function, **kwargs)

def get_primitive(name: str) -> Primitive:
    try: return _REGISTRY[name]
    except KeyError: raise ValueError(f"Unknown primitive {name!r}. Available: {sorted(_REGISTRY)}")

def available_primitives(): return tuple(sorted(_REGISTRY))

def validate_function_set(function_set):
    if isinstance(function_set, str):
        presets = {
            "fast": ("add","sub","mul","div","abs"),
            "basic": ("add","sub","mul","div"),
            "scientific": ("add","sub","mul","div","sin","cos","tanh","log","sqrt","exp","abs","inv"),
        }
        if function_set not in presets: raise ValueError(f"Unknown function_set preset {function_set!r}")
        function_set = presets[function_set]
    names=tuple(function_set)
    if not names: raise ValueError("function_set must not be empty")
    for n in names: get_primitive(n)
    return names

# Built-ins
_builtin = {
 "add": (2, lambda a,b:_finite(a+b),1,False), "sub": (2,lambda a,b:_finite(a-b),1,False),
 "mul": (2,lambda a,b:_finite(a*b),1,False), "div": (2,protected_div,2,False),
 "max": (2,np.maximum,1.5,True), "min": (2,np.minimum,1.5,True),
 "sin": (1,lambda a:_finite(np.sin(a)),3,True), "cos": (1,lambda a:_finite(np.cos(a)),3,True),
 "tanh": (1,lambda a:_finite(np.tanh(a)),3,True), "abs": (1,lambda a:_finite(np.abs(a)),1.2,True),
 "neg": (1,lambda a:_finite(-a),1,False), "log": (1,protected_log,3,True),
 "sqrt": (1,protected_sqrt,2,True), "exp": (1,protected_exp,4,True), "inv": (1,protected_inv,2,False),
 "square": (1,lambda a:_finite(a*a),1.2,False), "cube": (1,lambda a:_finite(a*a*a),1.5,False),
 "sign": (1,lambda a:_finite(np.sign(a)),1.5,True), "relu": (1,lambda a:_finite(np.maximum(a,0)),1.5,True),
 "softplus": (1,softplus,3,True), "log1p": (1,lambda a:_finite(np.log1p(np.abs(a))),2.5,True),
 "clip": (1,lambda a:_finite(np.clip(a,-1e3,1e3)),1.2,True),
 "is_missing": (1,is_missing,1.0,False), "coalesce": (2,coalesce,1.0,False),
}
for _n,(_a,_f,_c,_nl) in _builtin.items():
    register_primitive(_n,_a,_f,complexity=_c,nonlinear=_nl,numba_compatible=(_n not in {"is_missing","coalesce"}))

DEFAULT_FUNCTION_SET=("add","sub","mul","div")
