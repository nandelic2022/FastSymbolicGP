import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.program import (
    Program, random_tree,
    subtree_crossover_flat, subtree_crossover_genome,
    subtree_mutation_flat, subtree_mutation_genome,
    point_mutation_flat, point_mutation_genome,
)
from fastsymbolicgp.backend import compile_program


def _parents():
    fs=("add","sub","mul","div")
    rng=np.random.default_rng(123)
    a=Program(random_tree(rng,5,fs,5,(-1,1),.2))
    b=Program(random_tree(rng,5,fs,5,(-1,1),.2))
    return fs,a,b,a.as_flat_only(),b.as_flat_only()


def _same_genome(a,b):
    ao,aa=compile_program(a); bo,ba=compile_program(b)
    return np.array_equal(ao,bo) and np.array_equal(aa,ba)


def test_rootless_crossover_is_tree_equivalent():
    fs,a,b,af,bf=_parents()
    for seed in range(12):
        r1=np.random.default_rng(seed); r2=np.random.default_rng(seed)
        tree=subtree_crossover_flat(a,b,r1)
        flat=subtree_crossover_genome(af,bf,r2,100,1000)
        assert flat is not None and flat.root is None
        assert _same_genome(tree,flat)


def test_rootless_mutations_are_tree_equivalent():
    fs,a,b,af,bf=_parents()
    for seed in range(12):
        r1=np.random.default_rng(seed); r2=np.random.default_rng(seed)
        tree=subtree_mutation_flat(a,r1,5,fs,8,(-1,1),.2)
        flat=subtree_mutation_genome(af,r2,5,fs,8,1000,(-1,1),.2)
        assert flat is not None and _same_genome(tree,flat)
        r1=np.random.default_rng(seed); r2=np.random.default_rng(seed)
        tree=point_mutation_flat(a,r1,fs,5,(-1,1))
        flat=point_mutation_genome(af,r2,fs,5,(-1,1))
        assert _same_genome(tree,flat)


def test_numba_rootless_fast_path_quality_smoke():
    X,y=load_diabetes(return_X_y=True,as_frame=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    FastSymbolicRegressor(population_size=20,generations=1,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",random_state=42,verbose=0,display="off",constant_optimization="off",uncertainty="none").fit(Xtr,ytr)
    m=FastSymbolicRegressor(
        population_size=200,generations=10,function_set=("add","sub","mul","div"),
        max_depth=8,max_nodes=60,evaluation_backend="numba",fitness_engine="fused",optimization="scalar",
        semantic_diversity=False,age_management="off",memetic_search="off",random_state=42,verbose=0,display="off",
        constant_optimization="off",uncertainty="none",patience=1000,
    ).fit(Xtr,ytr)
    assert m.profile_["variation_engine"]=="numba_rootless_genome_v2"
    assert m.best_program_.root is not None  # final champion is materialized for public API
    assert np.isfinite(m.predict(Xte)).all()
    assert r2_score(yte,m.predict(Xte))>-1.0


def test_constraints_force_safe_tree_fallback():
    X=np.linspace(0,1,80)[:,None]; y=1+2*X[:,0]
    m=FastSymbolicRegressor(population_size=20,generations=2,random_state=3,verbose=0,display="off",
        constraints={"output_min":0.0,"monotonic":{"X0":"increasing"}},constant_optimization="off",uncertainty="none").fit(X,y)
    assert m.profile_["variation_engine"]!="numba_rootless_genome_v2"
