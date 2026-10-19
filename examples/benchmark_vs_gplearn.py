"""Apples-to-apples FastSymbolicGP vs gplearn speed/quality benchmark.

Install gplearn first if needed:
    python -m pip install gplearn
"""
from time import perf_counter
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from fastsymbolicgp import FastSymbolicRegressor

X,y=load_diabetes(return_X_y=True,as_frame=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42)

# JIT warm-up outside timing.
FastSymbolicRegressor(population_size=20,generations=1,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",random_state=42,verbose=0,display="off",constant_optimization="off",uncertainty="none").fit(Xtr,ytr)

fsgp=FastSymbolicRegressor(
    population_size=500,generations=100,function_set=("add","sub","mul","div"),
    max_depth=8,max_nodes=60,fitness_metric="rmse",fitness_scaling="linear",validation_fraction=.20,
    evaluation_backend="numba",fitness_engine="fused",optimization="scalar",
    semantic_diversity=False,age_management="off",memetic_search="off",n_jobs=-1,
    random_state=42,verbose=0,display="off",patience=1000,
)
t0=perf_counter(); fsgp.fit(Xtr,ytr); fsgp_s=perf_counter()-t0
fp=fsgp.predict(Xte)

print("\nFastSymbolicGP")
print("  seconds :",round(fsgp_s,4))
print("  R2      :",round(r2_score(yte,fp),6))
print("  RMSE    :",round(mean_squared_error(yte,fp)**.5,6))
print("  prog/s  :",round(fsgp.profile_.get("programs_per_second",0),1))
print("  engine  :",fsgp.profile_.get("variation_engine"))
print("  equation:",fsgp.get_expression())

try:
    from gplearn.genetic import SymbolicRegressor
except Exception as exc:
    print("\ngplearn is not installed. Run: python -m pip install gplearn")
    raise SystemExit(0) from exc

gp=SymbolicRegressor(
    population_size=500,generations=100,function_set=("add","sub","mul","div"),
    tournament_size=20,metric="rmse",parsimony_coefficient=0.0,
    random_state=42,n_jobs=-1,verbose=0,
)
t0=perf_counter(); gp.fit(Xtr,ytr); gp_s=perf_counter()-t0
gp_pred=gp.predict(Xte)
print("\ngplearn")
print("  seconds :",round(gp_s,4))
print("  R2      :",round(r2_score(yte,gp_pred),6))
print("  RMSE    :",round(mean_squared_error(yte,gp_pred)**.5,6))
print("  equation:",gp._program)

print("\nComparison")
print("  speed ratio (gplearn/FastSymbolicGP):",round(gp_s/max(fsgp_s,1e-12),3),"x")
print("  R2 delta (FastSymbolicGP-gplearn):",round(r2_score(yte,fp)-r2_score(yte,gp_pred),6))
