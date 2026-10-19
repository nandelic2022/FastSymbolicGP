import numpy as np

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.constraints import build_constraint_points, behavioral_constraint_penalty
from fastsymbolicgp.pareto import assign_nsga2, dominates, objective_values
from fastsymbolicgp.program import Node, Program, infer_unit
from fastsymbolicgp.serialization import save_model_json, load_model_json
from fastsymbolicgp.units import parse_unit, equivalent_units


def test_engineering_unit_aliases_and_prefixes_are_dimensionally_equivalent():
    assert equivalent_units('kW','W')
    assert equivalent_units('MW','W')
    assert equivalent_units('kWh','J')
    assert equivalent_units('bar','Pa')
    assert equivalent_units('degC','K')
    assert parse_unit('kg*m/s^3') == parse_unit('W/m')


def test_unitful_constant_round_trip_and_inference(tmp_path):
    node=Node('const',15.0,(), 'kW')
    p=Program(node)
    assert infer_unit(p.root,[]) == parse_unit('W')
    assert '[kW]' in p.to_string()
    clone=Program.from_dict(p.to_dict())
    assert clone.root.unit=='kW'
    assert infer_unit(clone.root,[]) == parse_unit('W')


def test_typed_generation_can_use_engineering_constant_units():
    rng=np.random.default_rng(4)
    X=rng.normal(size=(120,1))
    y=np.full(120,15.0)
    model=FastSymbolicRegressor(
        population_size=12,generations=2,max_depth=4,random_state=2,verbose=0,
        function_set=('add','mul','div'),feature_units=['1'],target_unit='W',unit_policy='enforce',
        typed_generation='on',constant_units={'kW':(10.0,20.0)},const_probability=1.0,
        uncertainty='off',constant_optimization='off',evaluation_backend='numpy',fitness_engine='materialized',
    ).fit(X,y)
    report=model.get_unit_report()
    assert report['passed'] is True
    assert report['expression_unit']==report['target_unit']
    assert any(parse_unit(c['unit'])==parse_unit('W') for c in report['typed_constants'])


def test_virtual_constraint_points_are_deterministic_and_respect_ranges():
    X=np.column_stack([np.linspace(0,1,20),np.linspace(10,20,20)])
    c={'virtual_n_points':16,'virtual_strategy':'lhs','virtual_seed':17,'virtual_ranges':{'X0':[-2,2],'X1':[100,200]}}
    a=build_constraint_points(X,['X0','X1'],c,max_points=8)
    b=build_constraint_points(X,['X0','X1'],c,max_points=8)
    assert np.allclose(a,b)
    virt=a[8:]
    assert len(virt)==16
    assert np.all((-2<=virt[:,0]) & (virt[:,0]<=2))
    assert np.all((100<=virt[:,1]) & (virt[:,1]<=200))


def test_physics_residual_constraint_reports_violation_and_success():
    X=np.linspace(-2,2,50)[:,None]
    good=Program(Node('feature',0),scale=2.0,intercept=1.0)
    bad=Program(Node('feature',0),scale=1.0,intercept=0.0)
    constraints={
        'virtual_n_points':12,'virtual_ranges':{'X0':[-3,3]},
        'residuals':[{'function':lambda Z,pred: pred-(2*Z[:,0]+1),'tolerance':1e-9}]
    }
    pg,ig,rg=behavioral_constraint_penalty(good,X,['X0'],constraints,return_report=True)
    pb,ib,rb=behavioral_constraint_penalty(bad,X,['X0'],constraints,return_report=True)
    assert not ig and rg['violations']==0 and pg < 1e-8
    assert ib and rb['violations']>0 and pb>0


def test_hard_monotonic_constraints_produce_feasible_champion_with_virtual_points():
    x=np.linspace(0,5,160)
    X=x[:,None]; y=3*x+2
    model=FastSymbolicRegressor(
        population_size=18,generations=3,max_depth=4,random_state=8,verbose=0,
        function_set=('add','sub'),constraints={
            'mode':'hard','mandatory_variables':['X0'],'monotonic':{'X0':'increasing'},
            'virtual_n_points':20,'virtual_ranges':{'X0':[-1,6]},'virtual_seed':3,
        }, uncertainty='off',constant_optimization='off',evaluation_backend='numpy',fitness_engine='materialized',
    ).fit(X,y)
    report=model.get_constraint_report()
    assert report['passed'] is True
    assert model.best_program_.metadata.get('feasible',True) is True
    assert report['behavioral']['points'] >= 20


def test_configurable_pareto_objectives_and_feasibility_domination():
    a=Program(Node('feature',0),loss=1.0,validation_loss=1.0,constraint_penalty=0.0)
    b=Program(Node('func','square',(Node('feature',0),)),loss=.8,validation_loss=.8,constraint_penalty=0.0)
    c=Program(Node('const',1.0),loss=.1,validation_loss=.1,constraint_penalty=0.0)
    a.metadata['feasible']=True; b.metadata['feasible']=True; c.metadata['feasible']=False
    objs=('validation_loss','weighted_complexity','variables')
    fronts=assign_nsga2([a,b,c],objs)
    assert tuple(a.metadata['pareto_objectives'])==objs
    assert len(objective_values(a,objs))==3
    assert dominates(a,c,objs) is True
    assert c.metadata['pareto_rank']>0
    assert fronts


def test_model_pareto_dataframe_exposes_selected_objectives():
    rng=np.random.default_rng(9); X=rng.normal(size=(100,2)); y=2*X[:,0]-X[:,1]
    objectives=('validation_loss','dag_nodes','weighted_complexity','variables')
    model=FastSymbolicRegressor(
        population_size=12,generations=2,max_depth=3,random_state=4,verbose=0,
        pareto_objectives=objectives,uncertainty='off',constant_optimization='off',evaluation_backend='numpy',fitness_engine='materialized'
    ).fit(X,y)
    df=model.get_pareto_front_dataframe()
    cols=set(df.columns) if hasattr(df,'columns') else set(df[0])
    for name in objectives:
        assert f'objective:{name}' in cols


def test_unitful_model_safe_json_round_trip(tmp_path):
    rng=np.random.default_rng(12); X=rng.normal(size=(80,1)); y=np.full(80,12.0)
    model=FastSymbolicRegressor(
        population_size=8,generations=1,max_depth=3,random_state=5,verbose=0,
        function_set=('add','mul'),feature_units=['1'],target_unit='W',unit_policy='enforce',
        typed_generation='on',constant_units={'kW':(10.0,20.0)},const_probability=1.0,
        uncertainty='off',constant_optimization='off',evaluation_backend='numpy',fitness_engine='materialized',
    ).fit(X,y)
    path=tmp_path/'unit_model.json'; model.save_model_json(path)
    loaded=FastSymbolicRegressor.load_model_json(path)
    assert np.allclose(model.predict(X[:12]),loaded.predict(X[:12]))
    assert any(getattr(n,'unit',None) for _,n in loaded.best_program_.root.iter_paths() if n.kind=='const')


def test_hard_scientific_constraints_also_work_on_fused_path():
    x=np.linspace(0,4,120); X=x[:,None]; y=2*x+1
    model=FastSymbolicRegressor(
        population_size=14,generations=2,max_depth=4,random_state=3,verbose=0,
        function_set=('add','sub'),constraints={'mode':'hard','mandatory_variables':['X0'],
            'monotonic':{'X0':'increasing'},'virtual_n_points':10,'virtual_ranges':{'X0':[-1,5]}},
        uncertainty='off',constant_optimization='off',fitness_engine='auto'
    ).fit(X,y)
    assert model.get_constraint_report()['passed'] is True
    assert model.best_program_.metadata.get('feasible',True) is True
