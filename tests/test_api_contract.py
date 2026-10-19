import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fastsymbolicgp import FastSymbolicRegressor


def test_pipeline_and_grid_search():
    X,y=load_diabetes(return_X_y=True); X=X[:90]; y=y[:90]
    model=FastSymbolicRegressor(population_size=8,generations=1,max_depth=2,validation_fraction=0,
        uncertainty='off',constant_optimization='off',evaluation_backend='numpy',random_state=1,verbose=0)
    pipe=Pipeline([('scale',StandardScaler()),('gp',model)])
    gs=GridSearchCV(pipe,{'gp__parsimony_coefficient':[0.0,0.01]},cv=2,error_score='raise')
    gs.fit(X,y)
    assert np.isfinite(gs.predict(X[:5])).all()


def test_feature_count_validation():
    X,y=load_diabetes(return_X_y=True)
    m=FastSymbolicRegressor(population_size=8,generations=1,max_depth=2,validation_fraction=0,
        uncertainty='off',constant_optimization='off',evaluation_backend='numpy',random_state=1,verbose=0).fit(X[:80],y[:80])
    with pytest.raises(ValueError): m.predict(X[:3,:-1])
