"""Cross-run interpretability and stability helpers (V1-compatible surface)."""
from __future__ import annotations
from collections import Counter


def feature_stability(models) -> dict:
    models=list(models)
    if not models: return {}
    present=Counter(); total=Counter()
    for model in models:
        for feature,stats in getattr(model,"feature_usage_",{}).items():
            present[feature]+=1; total[feature]+=int(stats.get("count",1)) if isinstance(stats,dict) else 1
    n=len(models)
    return {f:{"model_fraction":present[f]/n,"models":present[f],"total_occurrences":total[f]} for f in sorted(present,key=lambda k:(-present[k],k))}


def subexpression_stability(models,top_k=50) -> list[dict]:
    models=list(models); counts=Counter()
    for model in models:
        if getattr(model,"_is_multiclass_",False): programs=[e.best_program_ for e in model.estimators_]
        else: programs=[model.best_program_]
        seen=set()
        for program in programs:
            for _,node in program.root.iter_paths(): seen.add(node.to_string(getattr(model,"_symbolic_feature_names_",None)))
        counts.update(seen)
    n=max(1,len(models))
    return [{"subexpression":s,"models":c,"model_fraction":c/n} for s,c in counts.most_common(int(top_k))]
