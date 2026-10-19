import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.backend import compile_program
from fastsymbolicgp.program import (
    Program, random_tree, subtree_crossover, subtree_crossover_flat,
    subtree_mutation, subtree_mutation_flat, point_mutation, point_mutation_flat,
)


def _parents(seed=7):
    rng=np.random.default_rng(seed)
    a=Program(random_tree(rng,5,("add","sub","mul","div"),4,(-1,1),.2))
    b=Program(random_tree(rng,5,("add","sub","mul","div"),4,(-1,1),.2))
    return a,b


def _assert_genome_matches_tree(child):
    inherited=compile_program(child)
    expected_ops=inherited[0].copy(); expected_args=inherited[1].copy()
    child.metadata.pop("_compiled_postfix",None)
    fresh=compile_program(child)
    assert np.array_equal(expected_ops,fresh[0])
    assert np.array_equal(expected_args,fresh[1])


def test_flat_crossover_preserves_exact_tree_and_rng_path():
    a,b=_parents()
    r1=np.random.default_rng(100); r2=np.random.default_rng(100)
    old=subtree_crossover(a,b,r1); new=subtree_crossover_flat(a,b,r2)
    assert old.to_string()==new.to_string()
    _assert_genome_matches_tree(new)


def test_flat_subtree_mutation_preserves_exact_tree_and_rng_path():
    a,_=_parents()
    args=(5,("add","sub","mul","div"),8,(-1,1),.2)
    r1=np.random.default_rng(101); r2=np.random.default_rng(101)
    old=subtree_mutation(a,r1,*args); new=subtree_mutation_flat(a,r2,*args)
    assert old.to_string()==new.to_string()
    _assert_genome_matches_tree(new)


def test_flat_point_mutation_preserves_exact_tree_and_rng_path():
    a,_=_parents()
    r1=np.random.default_rng(102); r2=np.random.default_rng(102)
    old=point_mutation(a,r1,("add","sub","mul","div"),5,(-1,1))
    new=point_mutation_flat(a,r2,("add","sub","mul","div"),5,(-1,1))
    assert old.to_string()==new.to_string()
    _assert_genome_matches_tree(new)


def test_v126_fast_path_profile_and_quality_smoke():
    X,y=load_diabetes(return_X_y=True,as_frame=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    # warm fused signature
    FastSymbolicRegressor(population_size=20,generations=1,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",random_state=42,verbose=0,display="off",constant_optimization="off",uncertainty="none").fit(Xtr,ytr)
    m=FastSymbolicRegressor(population_size=200,generations=10,function_set=("add","sub","mul","div"),max_depth=8,max_nodes=60,fitness_metric="rmse",fitness_scaling="linear",validation_fraction=.2,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",semantic_diversity=False,age_management="off",memetic_search="off",n_jobs=-1,random_state=42,verbose=0,display="off",constant_optimization="off",uncertainty="none",patience=1000).fit(Xtr,ytr)
    assert m.profile_["variation_engine"]in {"flat_genome_inheritance_v1","numba_rootless_genome_v2"}
    assert m.resolved_config_["evaluation_backend"]=="numba"
    assert m.resolved_config_["fitness_engine"]=="fused"
    assert np.isfinite(m.predict(Xte)).all()
    assert m.profile_["actual_program_evaluations"]>0
