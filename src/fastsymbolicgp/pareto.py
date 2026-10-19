from __future__ import annotations
import math
from .primitives import get_primitive

DEFAULT_OBJECTIVES=("validation_loss","nodes","constraint_penalty")


def _weighted_complexity(p):
    total=0.0
    root=p.ensure_root() if getattr(p,"root",None) is None else p.root
    for _,node in root.iter_paths():
        if node.kind=="func": total+=float(get_primitive(str(node.value)).complexity)
        else: total+=0.25
    return float(total)


def objective_value(p,name):
    key=str(name).lower()
    if key in {"validation_loss","val_loss","loss"}: return float(p.validation_loss)
    if key in {"training_loss","train_loss"}: return float(p.loss)
    if key in {"nodes","size"}: return float(p.size)
    if key in {"dag_nodes","dag_size"}: return float(p.dag_size)
    if key=="depth": return float(p.depth)
    if key in {"constraint_penalty","constraints"}: return float(p.constraint_penalty)
    if key in {"variables","unique_variables"}: return float(len(set(p.root.feature_indices())))
    if key in {"nonlinear_operators","nonlinear_ops"}: return float(sum(bool(get_primitive(f).nonlinear) for f in p.root.function_names()))
    if key in {"weighted_complexity","inference_cost","flops"}: return _weighted_complexity(p)
    if key in {"semantic_redundancy","semantic_similarity"}: return -float(p.metadata.get("semantic_novelty",0.0))
    if key in {"age","lineage_age"}: return float(p.metadata.get("age",0))
    if key.startswith("metadata:"):
        return float(p.metadata.get(key.split(":",1)[1],math.inf))
    raise ValueError(f"Unknown Pareto objective {name!r}")


def normalize_objectives(objectives=None):
    values=DEFAULT_OBJECTIVES if objectives is None else tuple(objectives)
    if not values: raise ValueError("pareto_objectives must contain at least one objective")
    allowed={"validation_loss","val_loss","loss","training_loss","train_loss","nodes","size","dag_nodes","dag_size","depth","constraint_penalty","constraints","variables","unique_variables","nonlinear_operators","nonlinear_ops","weighted_complexity","inference_cost","flops","semantic_redundancy","semantic_similarity","age","lineage_age"}
    for name in values:
        if str(name).lower() not in allowed and not str(name).lower().startswith("metadata:"):
            raise ValueError(f"Unknown Pareto objective {name!r}")
    return tuple(map(str,values))


def objective_values(p,objectives=None):
    objs=normalize_objectives(objectives)
    vals=tuple(objective_value(p,name) for name in objs)
    p.metadata["pareto_objectives"]=objs
    p.metadata["pareto_objective_values"]=vals
    return vals


def _dominates_values(va,vb,fa=True,fb=True):
    if fa and not fb: return True
    if fb and not fa: return False
    any_better=False
    for x,y in zip(va,vb):
        if x>y: return False
        if x<y: any_better=True
    return any_better


def dominates(a,b,objectives=None):
    objs=normalize_objectives(objectives)
    return _dominates_values(
        objective_values(a,objs), objective_values(b,objs),
        bool(a.metadata.get("feasible",True)), bool(b.metadata.get("feasible",True)),
    )


def _front_indices(pop,values,feasible):
    n=len(pop)
    dominates_set=[[] for _ in range(n)]
    dominated_count=[0]*n
    first=[]
    for i in range(n):
        vi=values[i]; fi=feasible[i]
        for j in range(i+1,n):
            vj=values[j]; fj=feasible[j]
            if _dominates_values(vi,vj,fi,fj):
                dominates_set[i].append(j); dominated_count[j]+=1
            elif _dominates_values(vj,vi,fj,fi):
                dominates_set[j].append(i); dominated_count[i]+=1
    for i,c in enumerate(dominated_count):
        if c==0: first.append(i)
    fronts=[]; current=first
    while current:
        fronts.append(current)
        nxt=[]
        for i in current:
            for j in dominates_set[i]:
                dominated_count[j]-=1
                if dominated_count[j]==0: nxt.append(j)
        current=nxt
    return fronts


def nondominated_sort(population,objectives=None):
    pop=list(population)
    if not pop: return []
    objs=normalize_objectives(objectives)
    values=[objective_values(p,objs) for p in pop]
    feasible=[bool(p.metadata.get("feasible",True)) for p in pop]
    return [[pop[i] for i in front] for front in _front_indices(pop,values,feasible)]


def _assign_crowding(front,values):
    for p in front: p.metadata["crowding_distance"]=0.0
    if len(front)<=2:
        for p in front: p.metadata["crowding_distance"]=math.inf
        return
    m=len(values[0])
    for obj in range(m):
        order=sorted(range(len(front)),key=lambda i:values[i][obj])
        lo=values[order[0]][obj]; hi=values[order[-1]][obj]
        front[order[0]].metadata["crowding_distance"]=math.inf
        front[order[-1]].metadata["crowding_distance"]=math.inf
        if not math.isfinite(lo) or not math.isfinite(hi) or hi<=lo: continue
        scale=hi-lo
        for k in range(1,len(order)-1):
            p=front[order[k]]
            if math.isinf(p.metadata["crowding_distance"]): continue
            prev=values[order[k-1]][obj]; nxt=values[order[k+1]][obj]
            p.metadata["crowding_distance"] += float((nxt-prev)/scale)


def assign_nsga2(population,objectives=None):
    """Assign Pareto rank/crowding once using precomputed objective vectors."""
    pop=list(population)
    if not pop: return []
    objs=normalize_objectives(objectives)
    values=[objective_values(p,objs) for p in pop]
    feasible=[bool(p.metadata.get("feasible",True)) for p in pop]
    idx_fronts=_front_indices(pop,values,feasible)
    fronts=[]
    for rank,indices in enumerate(idx_fronts):
        front=[pop[i] for i in indices]
        front_values=[values[i] for i in indices]
        for p in front: p.metadata["pareto_rank"]=int(rank)
        _assign_crowding(front,front_values)
        fronts.append(front)
    return fronts


def update_archive(archive,candidates,max_size=256,objectives=None):
    """Incrementally maintain the non-dominated archive.

    Candidate programs are compared by reference and only the final archive is
    cloned. This avoids deep-copying every population member on every update.
    """
    objs=normalize_objectives(objectives)
    max_size=max(1,int(max_size))

    unique={}
    for p in list(archive)+list(candidates):
        key=p.structural_hash() if hasattr(p,"structural_hash") else p.to_string()
        old=unique.get(key)
        if old is None or objective_values(p,objs)<objective_values(old,objs):
            unique[key]=p

    feasible_pool=[p for p in unique.values() if bool(p.metadata.get("feasible",True))]
    pool=feasible_pool if feasible_pool else list(unique.values())

    front=[]; front_vals=[]; front_feasible=[]
    for p in pool:
        vp=objective_values(p,objs); fp=bool(p.metadata.get("feasible",True))
        dominated=False; keep=[]; keep_vals=[]; keep_feasible=[]
        for q,vq,fq in zip(front,front_vals,front_feasible):
            if _dominates_values(vq,vp,fq,fp):
                dominated=True; break
            if not _dominates_values(vp,vq,fp,fq):
                keep.append(q); keep_vals.append(vq); keep_feasible.append(fq)
        if dominated: continue
        front=keep+[p]; front_vals=keep_vals+[vp]; front_feasible=keep_feasible+[fp]

    if len(front)>max_size:
        _assign_crowding(front,front_vals)
        order=sorted(range(len(front)), key=lambda i:(-float(front[i].metadata.get("crowding_distance",0.0)), front_vals[i]))[:max_size]
        front=[front[i] for i in order]
    else:
        front.sort(key=lambda p:objective_values(p,objs))
    return front
