from time import perf_counter
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from fastsymbolicgp import FastSymbolicRegressor

X,y=load_diabetes(return_X_y=True,as_frame=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
FastSymbolicRegressor(population_size=20,generations=1,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",random_state=1,verbose=0).fit(X_train,y_train)
model=FastSymbolicRegressor(population_size=500,generations=100,function_set=("add","sub","mul","div"),max_depth=8,max_nodes=60,fitness_metric="rmse",fitness_scaling="linear",validation_fraction=0.20,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",semantic_diversity=False,age_management="off",memetic_search="off",n_jobs=-1,random_state=42,verbose=0,patience=1000)
t0=perf_counter(); model.fit(X_train,y_train); elapsed=perf_counter()-t0
pred=model.predict(X_test)
lin=LinearRegression().fit(X_train,y_train)
print("FastSymbolicGP version:", __import__("fastsymbolicgp").__version__)
print("runtime_seconds:", elapsed)
print("test_r2:", r2_score(y_test,pred))
print("test_rmse:", mean_squared_error(y_test,pred)**0.5)
print("linear_baseline_r2:", r2_score(y_test,lin.predict(X_test)))
print("programs_per_second:", model.profile_["programs_per_second"])
print("evaluation_seconds:", model.profile_["stage_seconds"]["evaluation"])
print("variation_seconds:", model.profile_["stage_seconds"]["variation"])
print("fused_training_passes:", model.execution_plan_.get("fused_training_passes"))
print("tournament_sampling:", model.profile_.get("tournament_sampling"))
print("equation:", model.get_expression())
