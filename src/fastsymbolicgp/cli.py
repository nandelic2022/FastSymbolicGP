from __future__ import annotations
import argparse, json

def diagnose():
    from .diagnostics import doctor;print(json.dumps(doctor(),indent=2,default=str))
def self_test():
    from .diagnostics import self_test as run;out=run(True);raise SystemExit(0 if out["passed"] else 1)
def fit_csv():
    p=argparse.ArgumentParser();p.add_argument("csv");p.add_argument("--target",required=True);p.add_argument("--task",choices=["regression","classification"],default="regression");p.add_argument("--preset",default="balanced");p.add_argument("--output",default="fastsymbolicgp_model.json");a=p.parse_args()
    import pandas as pd
    df=pd.read_csv(a.csv);y=df.pop(a.target).to_numpy();X=df
    if a.task=="regression":from .regressor import FastSymbolicRegressor as E
    else:from .classifier import FastSymbolicClassifier as E
    m=E(preset=a.preset).fit(X,y);m.save_model_json(a.output);print(m.get_expression());print(f"Saved {a.output}")
def benchmark():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="fastsymbolicgp_benchmark");p.add_argument("--task",choices=["classification","regression"],default="classification");p.add_argument("--datasets",default="");p.add_argument("--algorithms",default="fastsymbolicgp");p.add_argument("--runs",type=int,default=1);p.add_argument("--folds",type=int,default=5);p.add_argument("--population-size",type=int,default=32);p.add_argument("--generations",type=int,default=60);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
    from .benchmark import run_publication_benchmark
    out=run_publication_benchmark(a.output_dir,task=a.task,datasets=tuple(x for x in a.datasets.split(',') if x) or None,algorithms=tuple(x for x in a.algorithms.split(',') if x),runs=a.runs,folds=a.folds,population_size=a.population_size,generations=a.generations,seed=a.seed);print(json.dumps({"rows":len(out["raw_results"]),"failures":len(out["failures"]),"output_dir":a.output_dir},indent=2))

def certify():
    p=argparse.ArgumentParser(description="Certify a fitted FastSymbolicGP joblib model")
    p.add_argument("model"); p.add_argument("--output",default="fastsymbolicgp_scientific_certificate.json")
    a=p.parse_args()
    import joblib
    m=joblib.load(a.model); cert=m.get_scientific_certificate();
    from pathlib import Path
    Path(a.output).write_text(json.dumps(cert,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"passed":cert["passed"],"output":a.output,"version":cert["version"]},indent=2))
    raise SystemExit(0 if cert["passed"] else 2)
