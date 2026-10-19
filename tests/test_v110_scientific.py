import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_diabetes, load_iris

from fastsymbolicgp import (
    FastSymbolicRegressor, FastSymbolicClassifier, FastSymbolicTransformer,
    make_primitive, register_metric, doctor, self_test,
)
from fastsymbolicgp.program import Program, Node
from fastsymbolicgp.backend import NUMBA_AVAILABLE


def small_reg(**kwargs):
    p=dict(population_size=12,generations=2,max_depth=3,random_state=7,verbose=0,
           uncertainty='off',constant_optimization='off',evaluation_backend='numpy')
    p.update(kwargs); return FastSymbolicRegressor(**p)


def test_self_test_and_clone():
    assert self_test(verbose=False)['passed']
    assert isinstance(clone(small_reg()), FastSymbolicRegressor)


def test_ccpp_v100_style_parameters_are_accepted():
    X,y=load_diabetes(return_X_y=True)
    m=FastSymbolicRegressor(population_size=10,generations=1,tournament_size=3,max_depth=3,max_nodes=31,
        const_probability=.2,parsimony_coefficient=.001,patience=3,selection_tolerance=.02,
        exact_champion_candidates=4,validation_fraction=.2,preset=None,fitness_engine='auto',
        evaluation_backend='numpy',random_state=1,verbose=0,uncertainty='off',constant_optimization='off').fit(X[:100],y[:100])
    assert np.isfinite(m.predict(X[:5])).all()
    assert m.resolved_config_['exact_champion_candidates']==4


def test_epsilon_lexicase_constraints_and_pareto():
    X,y=load_diabetes(return_X_y=True)
    m=small_reg(selection_method='epsilon_lexicase',constraints={'output_min':0,'max_variables':5,'monotonic':{'X0':'increasing'}}).fit(X[:180],y[:180])
    assert len(m.get_pareto_front())>0
    assert m.get_expression_stats()['unique_variables']<=5
    assert np.isfinite(m.predict(X[:10])).all()


def test_split_conformal_reserves_calibration_before_search():
    X,y=load_diabetes(return_X_y=True)
    m=FastSymbolicRegressor(population_size=12,generations=2,max_depth=3,random_state=4,verbose=0,
        uncertainty='split_conformal',calibration_fraction=.2,evaluation_backend='numpy',constant_optimization='off').fit(X[:200],y[:200])
    p,lo,hi=m.predict_interval(X[:12],confidence=.9)
    assert m.calibration_size_==len(m.calibration_indices_)>0
    assert np.all(lo<=p) and np.all(p<=hi)


def test_units_enforce_and_penalty():
    rng=np.random.default_rng(1); X=rng.uniform(.2,2.0,(100,1)); y=X[:,0]**2
    m=small_reg(population_size=24,generations=2,function_set=('mul','square'),feature_units={'X0':'m'},target_unit='m^2',unit_policy='enforce').fit(X,y)
    assert m.best_program_.metadata.get('constraint_issues',[])==[]


def test_safe_json_round_trip(tmp_path):
    X,y=load_diabetes(return_X_y=True); m=small_reg().fit(X[:130],y[:130]); path=tmp_path/'m.json'; m.save_model_json(path)
    loaded=FastSymbolicRegressor.load_model_json(path)
    assert np.allclose(m.predict(X[:20]),loaded.predict(X[:20]))


def test_multiclass_and_transformer():
    X,y=load_iris(return_X_y=True)
    c=FastSymbolicClassifier(population_size=8,generations=1,max_depth=2,random_state=2,verbose=0,evaluation_backend='numpy').fit(X,y)
    assert c.predict_proba(X[:4]).shape==(4,3)
    assert len(c.predict_set(X[:4],.9))==4
    Xt,yt=load_diabetes(return_X_y=True)
    t=FastSymbolicTransformer(n_components=2,population_size=8,generations=1,max_depth=2,random_state=3,verbose=0)
    Z=t.fit_transform(Xt[:80],yt[:80]); assert Z.shape==(80,2)


def test_missing_value_strategies():
    X,y=load_diabetes(return_X_y=True); X=X[:100].copy(); y=y[:100]; X[0,0]=np.nan; X[4,3]=np.nan
    for strategy in ('median','constant','native'):
        m=small_reg(missing_value_strategy=strategy,missing_value_constant=-1).fit(X,y)
        assert np.isfinite(m.predict(X[:10])).all()


def test_custom_primitive_and_metric():
    name='quartic_v110_test'
    try: make_primitive(lambda x: np.asarray(x,float)**4,name,1,nonlinear=True)
    except ValueError: pass
    metric='negcorr_v110_test'
    def corr(y,p,w=None):
        y=np.asarray(y,float); p=np.asarray(p,float)
        return float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 else 0.0
    try: register_metric(metric,corr,direction='maximize')
    except ValueError: pass
    X,y=load_diabetes(return_X_y=True)
    m=small_reg(function_set=('add','mul',name),fitness_metric=metric).fit(X[:100],y[:100])
    assert np.isfinite(m.predict(X[:5])).all()


def test_numba_numpy_equivalence():
    if not NUMBA_AVAILABLE: pytest.skip('Numba optional dependency not installed')
    p=Program(Node('func','add',(Node('func','square',(Node('feature',0),)),Node('const',2.0))))
    X=np.linspace(-2,2,101)[:,None]
    p.metadata['evaluation_backend']='numpy'; a=p.raw(X)
    p.metadata['evaluation_backend']='numba'; p.metadata['n_jobs']=1; b=p.raw(X)
    assert np.allclose(a,b,rtol=1e-12,atol=1e-12)


def test_doctor_and_reproducibility_bundle(tmp_path):
    X,y=load_diabetes(return_X_y=True); m=small_reg().fit(X[:100],y[:100])
    assert doctor(X[:50],y[:50],estimator=m)['passed']
    out=tmp_path/'bundle'; m.export_reproducibility_bundle(out)
    for n in ('config.json','resolved_config.json','manifest.json','expression.txt','environment.json'):
        assert (out/n).exists()
