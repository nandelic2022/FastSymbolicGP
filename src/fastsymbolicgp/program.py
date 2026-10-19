from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import copy, math, re, hashlib
from functools import lru_cache
import numpy as np
from .primitives import get_primitive
from .units import mul_units, div_units, pow_unit, parse_unit

@dataclass
class Node:
    kind: str
    value: Any
    children: tuple["Node", ...] = ()
    unit: Any = None
    _size_cache: int | None = field(default=None, init=False, repr=False, compare=False)
    _depth_cache: int | None = field(default=None, init=False, repr=False, compare=False)
    def clone(self): return Node(self.kind,self.value,tuple(c.clone() for c in self.children),self.unit)
    @property
    def size(self):
        if self._size_cache is None: self._size_cache=1+sum(c.size for c in self.children)
        return int(self._size_cache)
    @property
    def depth(self):
        if self._depth_cache is None: self._depth_cache=1+max((c.depth for c in self.children),default=0)
        return int(self._depth_cache)
    def feature_indices(self):
        out=[]
        if self.kind=="feature": out.append(int(self.value))
        for c in self.children: out.extend(c.feature_indices())
        return out
    def function_names(self):
        out=[]
        if self.kind=="func": out.append(str(self.value))
        for c in self.children: out.extend(c.function_names())
        return out
    def constants(self):
        out=[]
        if self.kind=="const": out.append(float(self.value))
        for c in self.children: out.extend(c.constants())
        return out
    def paths(self,p=()):
        out=[p]
        for i,c in enumerate(self.children): out.extend(c.paths(p+(i,)))
        return out
    def get(self,path):
        n=self
        for i in path: n=n.children[i]
        return n
    def replace(self,path,new):
        if not path: return new.clone()
        cs=list(self.children); cs[path[0]]=cs[path[0]].replace(path[1:],new); return Node(self.kind,self.value,tuple(cs),self.unit)
    def replace_shared(self,path,new):
        """Persistent-tree replacement used by the evolutionary hot path.

        Population programs are treated as immutable during variation.  Reusing
        untouched subtrees avoids deep-copying entire parents for every child.
        Routines that optimise constants still call ``Program.clone()`` first,
        so in-place constant updates never leak across population members.
        """
        if not path:
            return new
        i=path[0]
        cs=list(self.children)
        cs[i]=cs[i].replace_shared(path[1:],new)
        return Node(self.kind,self.value,tuple(cs),self.unit)
    @property
    def name(self):
        return str(self.value) if self.kind == "func" else None
    @property
    def feature(self):
        return int(self.value) if self.kind == "feature" else None
    def iter_paths(self,p=()):
        yield p,self
        for i,c in enumerate(self.children):
            yield from c.iter_paths(p+(i,))
    def evaluate(self,X):
        if self.kind=="feature": return np.asarray(X[:,int(self.value)],float)
        if self.kind=="const": return np.full(X.shape[0],float(self.value))
        p=get_primitive(str(self.value)); args=[c.evaluate(X) for c in self.children]
        try: out=p.function(*args)
        except (FloatingPointError, OverflowError): out=np.zeros(X.shape[0],float)
        except Exception as exc: raise RuntimeError(f"Primitive {p.name!r} failed during evaluation: {exc}") from exc
        return np.nan_to_num(np.asarray(out,float),nan=0,posinf=1e6,neginf=-1e6)
    def to_string(self,names=None):
        if self.kind=="feature": return str(names[int(self.value)]) if names is not None else f"X{int(self.value)}"
        if self.kind=="const":
            text=f"{float(self.value):.10g}"
            return text if self.unit in {None,"","1",()} else f"{text}[{self.unit}]"
        name=str(self.value); a=[c.to_string(names) for c in self.children]
        if name in {"add","sub","mul","div"}:
            sym={"add":"+","sub":"-","mul":"*","div":"/"}[name]; return f"({a[0]} {sym} {a[1]})"
        if name=="neg": return f"(-{a[0]})"
        if name=="square": return f"({a[0]}**2)"
        if name=="cube": return f"({a[0]}**3)"
        return f"{name}({', '.join(a)})"
    def to_dict(self): return {"kind":self.kind,"value":self.value,"children":[c.to_dict() for c in self.children],"unit":self.unit}
    @classmethod
    def from_dict(cls,d): return cls(d["kind"],d["value"],tuple(cls.from_dict(c) for c in d.get("children",[])),d.get("unit"))

@dataclass
class Program:
    root: Node | None
    loss: float = math.inf
    validation_loss: float = math.inf
    scale: float = 1.0
    intercept: float = 0.0
    constraint_penalty: float = 0.0
    metadata: dict | None = None
    def __post_init__(self):
        if self.metadata is None: self.metadata={}
    def clone(self):
        # Public/deep clone. Flat genomes are immutable arrays during evolution;
        # copy metadata deeply so external mutation cannot alias estimator state.
        meta=copy.deepcopy(self.metadata or {})
        root=None if self.root is None else self.root.clone()
        q=Program(root,loss=float(self.loss),validation_loss=float(self.validation_loss),
                  scale=float(self.scale),intercept=float(self.intercept),constraint_penalty=float(self.constraint_penalty),metadata=meta)
        if root is not None and "_compiled_postfix" in q.metadata:
            ops,args=q.metadata["_compiled_postfix"][1:]; q.metadata["_compiled_postfix"]=(id(root),ops,args)
        return q
    def evolution_clone(self):
        """Cheap immutable clone for elites/HOF during evolution."""
        meta=dict(self.metadata or {})
        return Program(self.root,loss=float(self.loss),validation_loss=float(self.validation_loss),
                       scale=float(self.scale),intercept=float(self.intercept),constraint_penalty=float(self.constraint_penalty),metadata=meta)
    def ensure_root(self):
        """Materialize a Python tree only when a cold-path feature needs it."""
        if self.root is None:
            from .backend import compile_program, node_from_postfix
            ops,args=compile_program(self); self.root=node_from_postfix(ops,args)
            self.metadata["_compiled_postfix"]=(id(self.root),ops,args)
            self.metadata["tree_materialized"]=True
        return self.root
    def as_flat_only(self):
        """Return an equivalent program whose authoritative representation is postfix."""
        from .backend import compile_program, analyze_postfix
        ops,args=compile_program(self); starts,depths,preorder=analyze_postfix(ops)
        meta={k:v for k,v in (self.metadata or {}).items() if k not in {"_paths_cache","_dag_size_cache"}}
        q=Program(None,loss=float(self.loss),validation_loss=float(self.validation_loss),scale=float(self.scale),intercept=float(self.intercept),constraint_penalty=float(self.constraint_penalty),metadata=dict(meta))
        q.metadata["_compiled_postfix"]=(id(None),ops,args); q.metadata["_flat_topology"]=(starts,depths,preorder)
        q.metadata["_size_cache"]=int(len(ops)); q.metadata["_depth_cache"]=int(depths[-1]); q.metadata["flat_only"]=True
        return q
    @classmethod
    def from_postfix(cls,opcodes,arguments,*,loss=math.inf,validation_loss=math.inf,scale=1.0,intercept=0.0,constraint_penalty=0.0,metadata=None,topology=None):
        from .backend import analyze_postfix
        ops=np.ascontiguousarray(np.asarray(opcodes,dtype=np.int16)); args=np.ascontiguousarray(np.asarray(arguments,dtype=np.float64))
        starts,depths,preorder=analyze_postfix(ops) if topology is None else topology
        meta=dict(metadata or {}); meta["_compiled_postfix"]=(id(None),ops,args); meta["_flat_topology"]=(starts,depths,preorder); meta["_size_cache"]=int(len(ops)); meta["_depth_cache"]=int(depths[-1]); meta["flat_only"]=True
        return cls(None,loss=float(loss),validation_loss=float(validation_loss),scale=float(scale),intercept=float(intercept),constraint_penalty=float(constraint_penalty),metadata=meta)
    @property
    def scale_(self): return float(self.scale)
    @scale_.setter
    def scale_(self,v): self.scale=float(v)
    @property
    def intercept_(self): return float(self.intercept)
    @intercept_.setter
    def intercept_(self,v): self.intercept=float(v)
    @property
    def raw_fitness_(self): return float(self.loss)
    @raw_fitness_.setter
    def raw_fitness_(self,v): self.loss=float(v)
    @property
    def validation_fitness_(self): return float(self.validation_loss)
    @validation_fitness_.setter
    def validation_fitness_(self,v): self.validation_loss=float(v)
    @property
    def selection_fitness_(self): return float(self.validation_loss + self.constraint_penalty)
    @selection_fitness_.setter
    def selection_fitness_(self,v): self.metadata["selection_fitness_override"]=float(v)
    @property
    def metadata_(self): return self.metadata
    @metadata_.setter
    def metadata_(self,v): self.metadata=dict(v or {})
    def _ensure_size_depth(self):
        size=self.metadata.get("_size_cache"); depth=self.metadata.get("_depth_cache")
        if size is not None and depth is not None:
            return int(size),int(depth)
        if self.root is None:
            from .backend import compile_program, analyze_postfix
            ops,_=compile_program(self); topo=self.metadata.get("_flat_topology")
            if topo is None: topo=analyze_postfix(ops); self.metadata["_flat_topology"]=topo
            size,depth=len(ops),int(topo[1][-1])
        else:
            size,depth=self.root.size,self.root.depth
        self.metadata["_size_cache"]=int(size); self.metadata["_depth_cache"]=int(depth)
        return int(size),int(depth)
    @property
    def size(self):
        return self._ensure_size_depth()[0]
    @property
    def depth(self):
        return self._ensure_size_depth()[1]
    @property
    def dag_size(self):
        value=self.metadata.get("_dag_size_cache")
        if value is None:
            root=self.ensure_root(); value=len({node.to_string() for _,node in root.iter_paths()}); self.metadata["_dag_size_cache"]=int(value)
        return int(value)
    def structural_hash(self):
        value=self.metadata.get("_structural_hash_cache")
        if value is None:
            try:
                cached=(self.metadata or {}).get("_compiled_postfix")
                if self.root is None and cached is not None: ops,args=cached[1],cached[2]
                else:
                    from .backend import compile_program
                    ops,args=compile_program(self)
                h=hashlib.blake2b(digest_size=16); h.update(memoryview(ops)); h.update(memoryview(args)); value=h.hexdigest()
            except Exception:
                value=hashlib.blake2b(self.ensure_root().to_string().encode("utf-8"),digest_size=16).hexdigest()
            self.metadata["_structural_hash_cache"]=value
        return str(value)
    def invalidate_caches(self):
        if self.metadata is not None:
            for key in ("_compiled_postfix","_flat_topology","_paths_cache","_size_cache","_depth_cache","_dag_size_cache","_structural_hash_cache"):
                self.metadata.pop(key,None)
        return self
    def flat_genome(self):
        try:
            from .backend import compile_program
            opcodes,arguments=compile_program(self)
        except Exception:
            rows=list(self.root.iter_paths()); opcodes=np.arange(len(rows),dtype=np.int16); arguments=np.zeros(len(rows),dtype=float)
        return {"opcodes":opcodes,"arguments":arguments,"size":self.size,"depth":self.depth}
    def paths(self):
        cached=self.metadata.get("_paths_cache")
        if cached is None:
            cached=tuple(self.ensure_root().paths()); self.metadata["_paths_cache"]=cached
        return cached
    def raw(self,X):
        X=np.asarray(X,float); backend=str((self.metadata or {}).get("evaluation_backend","numpy")).lower()
        if backend in {"numba","auto","population_numba","postfix_numba"}:
            try:
                from .backend import can_numba, execute_numba
                if can_numba(self): return execute_numba(self,X,(self.metadata or {}).get("n_jobs",-1))
            except Exception:
                pass
        return np.nan_to_num(self.ensure_root().evaluate(X),nan=0.0,posinf=1e6,neginf=-1e6)
    def predict(self,X): return self.scale*self.raw(X)+self.intercept
    def to_string(self,names=None): return self.ensure_root().to_string(names)
    def to_dict(self): return {"root":self.ensure_root().to_dict(),"scale":self.scale,"intercept":self.intercept,"metadata":self.metadata or {}}
    @classmethod
    def from_dict(cls,d): return cls(Node.from_dict(d["root"]),scale=float(d.get("scale",1)),intercept=float(d.get("intercept",0)),metadata=d.get("metadata",{}))


@lru_cache(maxsize=128)
def _choices_for_arity(function_set_tuple,arity):
    return tuple(f for f in function_set_tuple if get_primitive(f).arity==int(arity))

def random_tree(rng,n_features,function_set,max_depth,const_range=(-1,1),const_probability=.15,depth=0):
    if depth>=max_depth or (depth>0 and rng.random()<0.25):
        if rng.random()<const_probability: return Node("const",float(rng.uniform(*const_range)))
        return Node("feature",int(rng.integers(0,n_features)))
    name=str(function_set[int(rng.integers(0,len(function_set)))]); p=get_primitive(name)
    return Node("func",name,tuple(random_tree(rng,n_features,function_set,max_depth,const_range,const_probability,depth+1) for _ in range(p.arity)))

def subtree_crossover(a,b,rng):
    paths_a=a.paths(); paths_b=b.paths(); pa=paths_a[int(rng.integers(0,len(paths_a)))]; pb=paths_b[int(rng.integers(0,len(paths_b)))]
    return Program(a.root.replace_shared(pa,b.root.get(pb)))
def subtree_mutation(a,rng,n_features,function_set,max_depth,const_range,const_probability):
    paths=a.paths(); p=paths[int(rng.integers(0,len(paths)))]; new=random_tree(rng,n_features,function_set,max(1,min(3,max_depth)),const_range,const_probability)
    return Program(a.root.replace_shared(p,new))
def point_mutation(a,rng,function_set,n_features,const_range):
    paths=a.paths(); p=paths[int(rng.integers(0,len(paths)))]; n=a.root.get(p)
    if n.kind=="feature": new=Node("feature",int(rng.integers(0,n_features)))
    elif n.kind=="const": new=Node("const",float(np.clip(float(n.value)+rng.normal(0,.25),const_range[0],const_range[1])),(),n.unit)
    else:
        arity=len(n.children); choices=_choices_for_arity(tuple(map(str,function_set)),arity); new=Node("func",str(choices[int(rng.integers(0,len(choices)))]),n.children,n.unit) if choices else n
    return Program(a.root.replace_shared(p,new))

def _postfix_span(root,path):
    """Return [start,end) of a tree path inside its postfix genome."""
    start=0; node=root
    if not path: return 0,int(root.size)
    for index in path:
        # A node's postfix segment is its children's segments followed by its root op.
        for child in node.children[:index]: start += int(child.size)
        node=node.children[index]
    return int(start),int(start+node.size)

def _attach_compiled_genome(program,opcodes,arguments):
    ops=np.ascontiguousarray(opcodes,dtype=np.int16); args=np.ascontiguousarray(arguments,dtype=np.float64)
    program.metadata["_compiled_postfix"]=(id(program.root),ops,args)
    program.metadata["flat_genome_inherited"]=True
    return program

def subtree_crossover_flat(a,b,rng):
    """Exact tree crossover plus zero-recompile postfix genome inheritance."""
    paths_a=a.paths(); paths_b=b.paths()
    pa=paths_a[int(rng.integers(0,len(paths_a)))]; pb=paths_b[int(rng.integers(0,len(paths_b)))]
    child=Program(a.root.replace_shared(pa,b.root.get(pb)))
    try:
        from .backend import compile_program
        ao,aa=compile_program(a); bo,ba=compile_program(b)
        sa,ea=_postfix_span(a.root,pa); sb,eb=_postfix_span(b.root,pb)
        co=np.concatenate((ao[:sa],bo[sb:eb],ao[ea:])); ca=np.concatenate((aa[:sa],ba[sb:eb],aa[ea:]))
        _attach_compiled_genome(child,co,ca)
    except Exception:
        pass
    return child

def subtree_mutation_flat(a,rng,n_features,function_set,max_depth,const_range,const_probability):
    """Exact subtree mutation while splicing the new subtree genome directly."""
    paths=a.paths(); p=paths[int(rng.integers(0,len(paths)))]
    new=random_tree(rng,n_features,function_set,max(1,min(3,max_depth)),const_range,const_probability)
    child=Program(a.root.replace_shared(p,new))
    try:
        from .backend import compile_program
        ao,aa=compile_program(a); no,na=compile_program(Program(new))
        s,e=_postfix_span(a.root,p)
        co=np.concatenate((ao[:s],no,ao[e:])); ca=np.concatenate((aa[:s],na,aa[e:]))
        _attach_compiled_genome(child,co,ca)
    except Exception:
        pass
    return child

def point_mutation_flat(a,rng,function_set,n_features,const_range):
    """Exact point mutation with O(genome) array copy and no tree recompilation."""
    paths=a.paths(); p=paths[int(rng.integers(0,len(paths)))]; n=a.root.get(p)
    if n.kind=="feature": new=Node("feature",int(rng.integers(0,n_features)))
    elif n.kind=="const": new=Node("const",float(np.clip(float(n.value)+rng.normal(0,.25),const_range[0],const_range[1])),(),n.unit)
    else:
        arity=len(n.children); choices=_choices_for_arity(tuple(map(str,function_set)),arity); new=Node("func",str(choices[int(rng.integers(0,len(choices)))]),n.children,n.unit) if choices else n
    child=Program(a.root.replace_shared(p,new))
    try:
        from .backend import compile_program, OPS
        ao,aa=compile_program(a); co=ao.copy(); ca=aa.copy(); _,e=_postfix_span(a.root,p); k=e-1
        if new.kind=="feature": co[k]=OPS["feature"]; ca[k]=float(new.value)
        elif new.kind=="const": co[k]=OPS["const"]; ca[k]=float(new.value)
        else: co[k]=OPS[str(new.value)]; ca[k]=0.0
        _attach_compiled_genome(child,co,ca)
    except Exception:
        pass
    return child

def simplify_node(n:Node):
    if n.kind!="func": return n.clone()
    cs=tuple(simplify_node(c) for c in n.children); name=n.value
    if all(c.kind=="const" for c in cs):
        try:
            u=infer_unit(Node("func",name,cs),[])
        except Exception:
            u=None
        if u==():
            X=np.zeros((1,1)); val=Node("func",name,cs).evaluate(X)[0]; return Node("const",float(val))
    if name=="add":
        if cs[0].kind=="const" and abs(float(cs[0].value))<1e-15: return cs[1]
        if cs[1].kind=="const" and abs(float(cs[1].value))<1e-15: return cs[0]
    if name=="sub" and cs[1].kind=="const" and abs(float(cs[1].value))<1e-15: return cs[0]
    if name=="mul":
        for i in (0,1):
            if cs[i].kind=="const" and abs(float(cs[i].value)-1)<1e-15: return cs[1-i]
            if cs[i].kind=="const" and abs(float(cs[i].value))<1e-15: return Node("const",0.0)
    if name=="div" and cs[1].kind=="const" and abs(float(cs[1].value)-1)<1e-15: return cs[0]
    # Canonical commutative ordering
    if name in {"add","mul","max","min"} and cs[0].to_string()>cs[1].to_string(): cs=(cs[1],cs[0])
    return Node("func",name,cs)

def simplify_program(p):
    q=p.clone(); q.root=simplify_node(q.root); q.invalidate_caches(); return q

def infer_unit(node,feature_units):
    if node.kind=="const": return parse_unit(node.unit) if getattr(node,"unit",None) not in {None,"","1",()} else ()
    if node.kind=="feature": return feature_units[int(node.value)]
    name=node.value; u=[infer_unit(c,feature_units) for c in node.children]
    if name in {"add","sub","max","min"}:
        if u[0]!=u[1]: raise ValueError("unit mismatch")
        return u[0]
    if name=="mul": return mul_units(u[0],u[1])
    if name=="div": return div_units(u[0],u[1])
    if name=="square": return pow_unit(u[0],2)
    if name=="cube": return pow_unit(u[0],3)
    if name in {"neg","abs","clip","relu"}: return u[0]
    if name=="is_missing": return ()
    if name=="coalesce":
        if u[0]!=u[1]: raise ValueError("coalesce requires matching units")
        return u[0]
    if name in {"sin","cos","tanh","log","exp","softplus","log1p","sign"}:
        if u[0]: raise ValueError(f"{name} requires dimensionless input")
        return ()
    if name in {"sqrt"}:
        d=dict(u[0]);
        if any(v%2 for v in d.values()): raise ValueError("sqrt requires even unit powers")
        return tuple(sorted((k,v//2) for k,v in d.items() if v//2))
    if name=="inv": return tuple(sorted((k,-v) for k,v in u[0]))
    return ()


# V1 stable public name
SymbolicProgram = Program


# ---------------------------------------------------------------------------
# V1.1.4 strongly typed / dimension-aware generation and variation
# ---------------------------------------------------------------------------
def _normalize_units(feature_units):
    return [parse_unit(u) for u in feature_units]


def _unit_root(unit, power):
    d=dict(unit)
    if power == 0:
        return ()
    if any(v % power for v in d.values()):
        return None
    return tuple(sorted((k, v//power) for k,v in d.items() if v//power))


def _inv_unit(unit):
    return tuple(sorted((k,-v) for k,v in dict(unit).items() if v))


def _constant_spec_for_unit(constant_units, desired_unit):
    if not constant_units:
        return None
    desired=parse_unit(desired_unit) if not isinstance(desired_unit,tuple) else desired_unit
    items=constant_units.items() if isinstance(constant_units,dict) else ((u,None) for u in constant_units)
    for unit,spec in items:
        if parse_unit(unit)==desired:
            return unit,spec
    return None


def _terminal_for_unit(rng, feature_units, desired_unit, const_range=(-1,1), const_probability=.15, constant_units=None):
    matches=[i for i,u in enumerate(feature_units) if u==desired_unit]
    cspec=_constant_spec_for_unit(constant_units,desired_unit)
    allow_const=(desired_unit==() or cspec is not None)
    if allow_const and (not matches or rng.random()<float(const_probability)):
        if cspec is None:
            lo,hi=const_range; label=None
        else:
            label,spec=cspec
            if spec is None: lo,hi=const_range
            elif isinstance(spec,(tuple,list)) and len(spec)==2: lo,hi=map(float,spec)
            elif isinstance(spec,dict): lo=float(spec.get("min",const_range[0])); hi=float(spec.get("max",const_range[1]))
            else: lo,hi=const_range
        return Node("const",float(rng.uniform(lo,hi)),(),label)
    if matches:
        return Node("feature",int(rng.choice(matches)))
    if allow_const:
        if cspec is None: lo,hi=const_range; label=None
        else:
            label,spec=cspec
            if isinstance(spec,(tuple,list)) and len(spec)==2: lo,hi=map(float,spec)
            elif isinstance(spec,dict): lo=float(spec.get("min",const_range[0])); hi=float(spec.get("max",const_range[1]))
            else: lo,hi=const_range
        return Node("const",float(rng.uniform(lo,hi)),(),label)
    return None

def _unit_closure(feature_units, constant_units=None, rounds=2, max_units=128):
    available=set(feature_units)|{()}
    if constant_units:
        items=constant_units.keys() if isinstance(constant_units,dict) else constant_units
        available.update(parse_unit(u) for u in items)
    for _ in range(max(0,int(rounds))):
        before=len(available); base=list(available)
        for a in base:
            available.add(_inv_unit(a))
            for b in base:
                available.add(mul_units(a,b)); available.add(div_units(a,b))
                if len(available)>=int(max_units): return available
        if len(available)==before: break
    return available


def _candidate_unit_rules(function_set, feature_units, desired_unit, constant_units=None):
    """Yield unit-valid grammar productions for the requested output dimension."""
    available=_unit_closure(feature_units,constant_units)
    rules=[]
    for name in function_set:
        name=str(name)
        arity=get_primitive(name).arity
        if arity==1:
            if name in {"neg","abs","clip","relu"}:
                rules.append((name,(desired_unit,)))
            elif name=="square":
                root=_unit_root(desired_unit,2)
                if root is not None: rules.append((name,(root,)))
            elif name=="cube":
                root=_unit_root(desired_unit,3)
                if root is not None: rules.append((name,(root,)))
            elif name=="sqrt":
                rules.append((name,(pow_unit(desired_unit,2),)))
            elif name=="inv":
                rules.append((name,(_inv_unit(desired_unit),)))
            elif name in {"sin","cos","tanh","log","exp","softplus","log1p","sign","is_missing"}:
                if desired_unit==(): rules.append((name,((),)))
        elif arity==2:
            if name in {"add","sub","max","min","coalesce"}:
                rules.append((name,(desired_unit,desired_unit)))
            elif name=="mul":
                # Always-valid dimensionless split plus decompositions from known feature units.
                rules.append((name,(desired_unit,())))
                rules.append((name,((),desired_unit)))
                for ua in available:
                    for ub in available:
                        if mul_units(ua,ub)==desired_unit:
                            rules.append((name,(ua,ub)))
            elif name=="div":
                rules.append((name,(desired_unit,())))
                rules.append((name,((),_inv_unit(desired_unit))))
                for ua in available:
                    for ub in available:
                        if div_units(ua,ub)==desired_unit:
                            rules.append((name,(ua,ub)))
    # Deduplicate while preserving order.
    out=[]; seen=set()
    for rule in rules:
        key=(rule[0],tuple(rule[1]))
        if key not in seen:
            seen.add(key); out.append(rule)
    return out


def typed_random_tree(rng,n_features,function_set,max_depth,feature_units,target_unit,
                      const_range=(-1,1),const_probability=.15,depth=1,constant_units=None):
    """Generate a dimensionally valid tree by construction.

    ``max_depth`` uses the public Program depth convention: a terminal has depth 1.
    The generator falls back to a unit-compatible terminal when no valid primitive rule
    can be expanded within the remaining depth.
    """
    feature_units=_normalize_units(feature_units)
    desired=parse_unit(target_unit) if not isinstance(target_unit,tuple) else target_unit
    terminal=_terminal_for_unit(rng,feature_units,desired,const_range,const_probability,constant_units)
    if depth>=int(max_depth):
        if terminal is not None: return terminal
        raise ValueError(f"No unit-compatible terminal for {desired!r} at maximum depth")
    if depth>1 and rng.random()<0.22 and terminal is not None:
        return terminal
    rules=_candidate_unit_rules(function_set,feature_units,desired,constant_units)
    if not rules:
        if terminal is not None: return terminal
        raise ValueError(f"No typed production can generate unit {desired!r} from the available features/functions")
    # Randomized retries let child-generation failures choose another valid production.
    order=rng.permutation(len(rules))
    for idx in order:
        name,child_units=rules[int(idx)]
        try:
            children=tuple(typed_random_tree(
                rng,n_features,function_set,max_depth,feature_units,u,
                const_range,const_probability,depth+1,constant_units,
            ) for u in child_units)
            node=Node("func",name,children)
            if infer_unit(node,feature_units)==desired:
                return node
        except ValueError:
            continue
    if terminal is not None: return terminal
    raise ValueError(f"Unable to construct typed subtree for unit {desired!r}")


def typed_subtree_crossover(a,b,rng,feature_units):
    """Crossover only between subtrees with identical inferred units."""
    fu=_normalize_units(feature_units)
    paths_a=list(a.paths()); rng.shuffle(paths_a)
    by_unit={}
    for pb,node in b.root.iter_paths():
        try: u=infer_unit(node,fu)
        except Exception: continue
        by_unit.setdefault(u,[]).append(pb)
    for pa in paths_a:
        try: ua=infer_unit(a.root.get(pa),fu)
        except Exception: continue
        matches=by_unit.get(ua,())
        if matches:
            pb=matches[int(rng.integers(0,len(matches)))]
            return Program(a.root.replace(pa,b.root.get(pb)))
    return a.clone()


def typed_subtree_mutation(a,rng,n_features,function_set,max_depth,feature_units,const_range,const_probability,constant_units=None):
    """Replace a subtree with a newly generated subtree of the same unit."""
    fu=_normalize_units(feature_units); paths=list(a.paths()); rng.shuffle(paths)
    for path in paths:
        node=a.root.get(path)
        try: desired=infer_unit(node,fu)
        except Exception: continue
        remaining=max(1,int(max_depth)-len(path))
        try:
            new=typed_random_tree(rng,n_features,function_set,remaining,fu,desired,const_range,const_probability,1,constant_units)
            return Program(a.root.replace(path,new))
        except ValueError:
            continue
    return a.clone()


def typed_point_mutation(a,rng,function_set,n_features,feature_units,const_range,constant_units=None):
    """Unit-preserving point mutation; invalid candidates are rejected locally."""
    fu=_normalize_units(feature_units); paths=list(a.paths()); rng.shuffle(paths)
    for path in paths:
        old=a.root.get(path)
        try: expected=infer_unit(old,fu)
        except Exception: continue
        candidates=[]
        if old.kind=="feature":
            candidates=[Node("feature",i) for i,u in enumerate(fu) if u==expected]
            spec=_constant_spec_for_unit(constant_units,expected)
            if expected==() or spec is not None:
                if spec is None: lo,hi=const_range; label=None
                else:
                    label,cfg=spec
                    if isinstance(cfg,(tuple,list)) and len(cfg)==2: lo,hi=map(float,cfg)
                    elif isinstance(cfg,dict): lo=float(cfg.get("min",const_range[0])); hi=float(cfg.get("max",const_range[1]))
                    else: lo,hi=const_range
                candidates.append(Node("const",float(rng.uniform(lo,hi)),(),label))
        elif old.kind=="const":
            label=getattr(old,"unit",None); spec=_constant_spec_for_unit(constant_units,expected)
            if spec is not None and isinstance(spec[1],(tuple,list)) and len(spec[1])==2: lo,hi=map(float,spec[1])
            else: lo,hi=const_range
            candidates=[Node("const",float(np.clip(float(old.value)+rng.normal(0,.25),lo,hi)),(),label)]
            candidates += [Node("feature",i) for i,u in enumerate(fu) if u==expected]
        else:
            for name in function_set:
                if get_primitive(name).arity!=len(old.children): continue
                candidate=Node("func",str(name),old.children)
                try:
                    if infer_unit(candidate,fu)==expected: candidates.append(candidate)
                except Exception: pass
        if candidates:
            return Program(a.root.replace(path,candidates[int(rng.integers(0,len(candidates)))]))
    return a.clone()

# ---------------------------------------------------------------------------
# V1.2.7 rootless flat-genome variation (standard fast path)
# ---------------------------------------------------------------------------
def _flat_arrays(program):
    cached=(program.metadata or {}).get("_compiled_postfix")
    if cached is not None:
        return cached[1],cached[2]
    from .backend import compile_program
    return compile_program(program)

def _flat_topology(program):
    topo=(program.metadata or {}).get("_flat_topology")
    if topo is None:
        from .backend import analyze_postfix
        ops,_=_flat_arrays(program); topo=analyze_postfix(ops); program.metadata["_flat_topology"]=topo
    return topo


def random_genome(rng,n_features,function_set,max_depth,const_range=(-1,1),const_probability=.15,depth=0):
    """Generate the exact postfix equivalent of ``random_tree`` without Nodes.

    The order and type of RNG calls intentionally mirror ``random_tree`` so the
    quality-preserving fast path keeps the same stochastic search trajectory.
    """
    from .backend import OPS
    ops=[]; args=[]
    def grow(d):
        if d>=max_depth or (d>0 and rng.random()<0.25):
            if rng.random()<const_probability:
                ops.append(OPS["const"]); args.append(float(rng.uniform(*const_range)))
            else:
                ops.append(OPS["feature"]); args.append(float(rng.integers(0,n_features)))
            return
        name=str(function_set[int(rng.integers(0,len(function_set)))]); op=OPS[name]; arity=2 if op<20 else 1
        for _ in range(arity): grow(d+1)
        ops.append(op); args.append(0.0)
    grow(depth)
    return np.asarray(ops,dtype=np.int16),np.asarray(args,dtype=np.float64)


def subtree_crossover_genome(a,b,rng,max_depth,max_nodes):
    """Tree-equivalent crossover performed only on postfix genomes."""
    from .backend import analyze_postfix
    ao,aa=_flat_arrays(a); bo,ba=_flat_arrays(b)
    astarts,adepths,apre=_flat_topology(a); bstarts,bdepths,bpre=_flat_topology(b)
    ia=int(rng.integers(0,len(apre))); ib=int(rng.integers(0,len(bpre)))
    ta=int(apre[ia]); tb=int(bpre[ib]); sa=int(astarts[ta]); ea=ta+1; sb=int(bstarts[tb]); eb=tb+1
    new_len=len(ao)-(ea-sa)+(eb-sb)
    if new_len>int(max_nodes): return None
    co=np.concatenate((ao[:sa],bo[sb:eb],ao[ea:])); ca=np.concatenate((aa[:sa],ba[sb:eb],aa[ea:]))
    topo=analyze_postfix(co)
    if int(topo[1][-1])>int(max_depth): return None
    return Program.from_postfix(co,ca,metadata={"flat_genome_inherited":True},topology=topo)


def subtree_mutation_genome(a,rng,n_features,function_set,max_depth,max_nodes,const_range,const_probability):
    """Tree-equivalent subtree mutation without Python tree construction."""
    from .backend import analyze_postfix
    ao,aa=_flat_arrays(a); starts,depths,pre=_flat_topology(a)
    idx=int(rng.integers(0,len(pre))); token=int(pre[idx]); ss=int(starts[token]); ee=token+1
    no,na=random_genome(rng,n_features,function_set,max(1,min(3,max_depth)),const_range,const_probability)
    new_len=len(ao)-(ee-ss)+len(no)
    if new_len>int(max_nodes): return None
    co=np.concatenate((ao[:ss],no,ao[ee:])); ca=np.concatenate((aa[:ss],na,aa[ee:]))
    topo=analyze_postfix(co)
    if int(topo[1][-1])>int(max_depth): return None
    return Program.from_postfix(co,ca,metadata={"flat_genome_inherited":True},topology=topo)


def point_mutation_genome(a,rng,function_set,n_features,const_range):
    """Exact point mutation directly on postfix arrays.

    Because point mutation preserves primitive arity, the parent's topology is
    reusable by the child with no analysis pass.
    """
    from .backend import OPS
    ao,aa=_flat_arrays(a); starts,depths,pre=_flat_topology(a)
    idx=int(rng.integers(0,len(pre))); token=int(pre[idx]); op=int(ao[token])
    co=ao.copy(); ca=aa.copy()
    if op==OPS["feature"]:
        co[token]=OPS["feature"]; ca[token]=float(rng.integers(0,n_features))
    elif op==OPS["const"]:
        ca[token]=float(np.clip(float(ca[token])+rng.normal(0,.25),const_range[0],const_range[1]))
    else:
        arity=2 if op<20 else 1
        choices=_choices_for_arity(tuple(map(str,function_set)),arity)
        if choices:
            name=str(choices[int(rng.integers(0,len(choices)))]); co[token]=OPS[name]; ca[token]=0.0
    topo=(starts,depths,pre)
    return Program.from_postfix(co,ca,metadata={"flat_genome_inherited":True},topology=topo)
