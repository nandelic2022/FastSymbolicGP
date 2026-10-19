import numpy as np
from sklearn.datasets import load_diabetes, load_iris

from fastsymbolicgp import FastSymbolicRegressor, FastSymbolicClassifier
from fastsymbolicgp.backend import compile_program
from fastsymbolicgp.program import Node, Program, infer_unit
from fastsymbolicgp.units import parse_unit


def _tiny_reg(**kwargs):
    params=dict(population_size=12,generations=2,tournament_size=3,random_state=123,
                verbose=0,display='off',validation_fraction=0.2,fitness_engine='auto')
    params.update(kwargs)
    return FastSymbolicRegressor(**params)


def test_typed_generation_and_variation_keep_units_valid():
    rng=np.random.default_rng(1)
    X=np.column_stack([rng.normal(100,10,120), rng.normal(size=120)])
    y=2.0*X[:,0]+0.1*X[:,1]
    model=_tiny_reg(
        population_size=16,generations=3,function_set=('add','sub','mul','div','neg','abs'),
        feature_units=['W','1'],target_unit='W',unit_policy='enforce',typed_generation='on',
        max_depth=6,n_islands=2,evolution_model='islands',
    ).fit(X,y)
    fu=[parse_unit('W'),parse_unit('1')]
    target=parse_unit('W')
    programs=[p for island in model.evolution_state_['islands'] for p in island]
    assert programs
    assert all(infer_unit(p.root,fu)==target for p in programs)
    assert any(p.metadata.get('typed_generated') or p.metadata.get('typed_variation') for p in programs)


def test_compiled_postfix_cache_is_reused():
    p=Program(Node('func','add',(Node('feature',0),Node('const',1.0))))
    a1,b1=compile_program(p)
    a2,b2=compile_program(p)
    assert np.array_equal(a1,a2) and np.array_equal(b1,b2)
    assert p.metadata.get('compile_cache_misses',0)==1
    assert p.metadata.get('compile_cache_hits',0)>=1


def test_fused_batch_structural_duplicates_are_scored_once():
    X,y=load_diabetes(return_X_y=True)
    model=_tiny_reg(population_size=8,generations=1,fitness_engine='fused').fit(X[:100],y[:100])
    base=model.best_program_.clone()
    pop=[base.clone() for _ in range(5)]
    model._fitness_cache_token_=None; model._fitness_cache_={}; model._fitness_cache_hits_=0; model._fitness_cache_misses_=0
    before=model.evaluation_count_
    model._evaluate_population(pop,X[:80],y[:80],None,None)
    assert model.evaluation_count_-before==1
    assert model._duplicate_fitness_reuse_>=4
    assert all(np.isclose(p.loss,pop[0].loss) for p in pop)


def test_cross_island_population_batching_and_stage_profile():
    X,y=load_diabetes(return_X_y=True)
    model=_tiny_reg(population_size=18,generations=2,n_islands=3,evolution_model='islands',island_parallel=True).fit(X[:140],y[:140])
    assert model.execution_plan_['island_evaluation_mode']=='cross_island_population_batch'
    assert model.profile_['cross_island_population_batching'] is True
    assert {'evaluation','lexicase_cache_and_sort','variation','migration','checkpoint'} <= set(model.profile_['stage_seconds'])
    assert model.history_[0]['unique_structures'] > 0


def test_parallel_ovr_uses_outer_workers_and_limits_child_threads():
    X,y=load_iris(return_X_y=True)
    model=FastSymbolicClassifier(
        population_size=10,generations=1,tournament_size=3,random_state=7,verbose=0,display='off',
        validation_fraction=0.2,ovr_parallel=True,ovr_n_jobs=2,n_jobs=-1,fitness_engine='auto',
    ).fit(X,y)
    assert model.profile_['ovr_workers']==2
    assert model.profile_['nested_child_n_jobs']==1
    assert len(model.estimators_)==3
    assert all(e.n_jobs==1 for e in model.estimators_)
    assert model.predict(X[:5]).shape==(5,)


def test_constructor_parameters_remain_stable_with_new_runtime_controls():
    m=_tiny_reg(n_islands=3,island_parallel=False,evaluation_cache=False,fitness_cache_max_entries=123,typed_generation='off')
    before=m.get_params(deep=False).copy()
    X,y=load_diabetes(return_X_y=True)
    m.fit(X[:100],y[:100])
    after=m.get_params(deep=False)
    for key in ('n_islands','island_parallel','evaluation_cache','fitness_cache_max_entries','typed_generation'):
        assert after[key]==before[key]
