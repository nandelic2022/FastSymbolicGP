from __future__ import annotations
from collections import Counter
import re

# FastSymbolicGP tracks *dimensions*, not conversion factors. Therefore kW, MW and
# W are dimensionally equivalent, as are Wh/kWh/J. This is deliberate: feature
# values remain in the units supplied by the dataset; the unit engine only rejects
# dimensionally impossible expressions.
_BASE_UNITS = {
    "1": {}, "": {}, "-": {}, "dimensionless": {}, "%": {},
    "m": {"m":1}, "s":{"s":1}, "kg":{"kg":1}, "K":{"K":1}, "A":{"A":1},
    "mol":{"mol":1}, "cd":{"cd":1},
    "Hz":{"s":-1}, "N":{"kg":1,"m":1,"s":-2}, "Pa":{"kg":1,"m":-1,"s":-2},
    "J":{"kg":1,"m":2,"s":-2}, "W":{"kg":1,"m":2,"s":-3},
    "V":{"kg":1,"m":2,"s":-3,"A":-1}, "C":{"A":1,"s":1},
    "Ohm":{"kg":1,"m":2,"s":-3,"A":-2}, "S":{"kg":-1,"m":-2,"s":3,"A":2},
    "T":{"kg":1,"s":-2,"A":-1}, "Wb":{"kg":1,"m":2,"s":-2,"A":-1},
    "L":{"m":3}, "l":{"m":3}, "min":{"s":1}, "h":{"s":1}, "rpm":{"s":-1},
    "bar":{"kg":1,"m":-1,"s":-2}, "psi":{"kg":1,"m":-1,"s":-2},
    "Wh":{"kg":1,"m":2,"s":-2},
    "degC":{"K":1}, "Celsius":{"K":1}, "°C":{"K":1},
}

# Prefixes are accepted only when the remaining symbol is known. Their scale is
# intentionally ignored because dimensional GP validates dimensions, not magnitudes.
_PREFIXES=("da","Y","Z","E","P","T","G","M","k","h","d","c","m","u","µ","n","p","f")


def _clean(d):
    return tuple(sorted((k,int(v)) for k,v in d.items() if v))


def _lookup_symbol(symbol: str):
    if symbol in _BASE_UNITS:
        return _BASE_UNITS[symbol]
    # Common spelling aliases.
    aliases={"sec":"s","second":"s","seconds":"s","meter":"m","metre":"m","kelvin":"K","amp":"A"}
    if symbol in aliases:
        return _BASE_UNITS[aliases[symbol]]
    for prefix in _PREFIXES:
        if symbol.startswith(prefix) and len(symbol)>len(prefix):
            base=symbol[len(prefix):]
            if base in _BASE_UNITS and base not in {"kg"}:  # kg is already an SI base unit.
                return _BASE_UNITS[base]
    return None


def parse_unit(unit):
    if unit is None:
        return ()
    if isinstance(unit, tuple):
        return _clean(dict(unit))
    if isinstance(unit, list):
        return _clean(dict(unit))
    u=str(unit).strip().replace("·","*").replace(" ","")
    if u in _BASE_UNITS:
        return _clean(_BASE_UNITS[u])
    d=Counter(); sign=1
    for tok in re.split(r'([*/])',u):
        tok=tok.strip()
        if not tok:
            continue
        if tok=='*':
            sign=1; continue
        if tok=='/':
            sign=-1; continue
        m=re.fullmatch(r'([A-Za-zµ°%]+)(?:\^(-?\d+))?',tok)
        if not m:
            raise ValueError(f"Unsupported unit token {tok!r}")
        base=m.group(1); power=int(m.group(2) or 1)*sign
        bd=_lookup_symbol(base)
        if bd is None:
            # Unknown symbols are allowed as independent dimensions, preserving
            # backwards compatibility while still enforcing dimensional consistency.
            bd={base:1}
        for k,v in bd.items():
            d[k]+=v*power
    return _clean(d)


def mul_units(a,b):
    d=Counter(dict(a)); d.update(dict(b)); return _clean(d)


def div_units(a,b):
    d=Counter(dict(a))
    for k,v in dict(b).items(): d[k]-=v
    return _clean(d)


def pow_unit(a,p):
    return _clean({k:v*int(p) for k,v in dict(a).items()})


def equivalent_units(a,b):
    return parse_unit(a)==parse_unit(b)


def unit_to_string(u):
    u=parse_unit(u) if not isinstance(u,tuple) else u
    if not u: return "1"
    positive=[]; negative=[]
    for k,v in u:
        if v>0: positive.append(k if v==1 else f"{k}^{v}")
        elif v<0: negative.append(k if v==-1 else f"{k}^{-v}")
    num="*".join(positive) or "1"
    return num if not negative else num+"/"+"*".join(negative)
