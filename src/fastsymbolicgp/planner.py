"""Non-mutating execution planner for V1.1.4."""
from __future__ import annotations
import os

def build_execution_plan(estimator,X):
    n,p=map(int,X.shape);cpus=max(1,int(os.cpu_count() or 1))
    if n<=5000 and p<=128: workload="small"; islands=1
    elif n<=100000 and p<=512: workload="medium"; islands=min(2,cpus)
    else: workload="large"; islands=min(4,cpus)
    return {"workload":workload,"n_samples":n,"n_features":p,"cpu_count":cpus,"recommended_n_islands":islands,"recommended_evolution_model":"islands" if islands>1 else "panmictic","applied":False}
