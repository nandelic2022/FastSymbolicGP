import importlib.util
import numpy as np
import pytest
from sklearn.datasets import make_regression
from fastsymbolicgp import FastSymbolicRegressor


def fitted():
    X,y=make_regression(n_samples=80,n_features=3,random_state=91)
    m=FastSymbolicRegressor(population_size=12,generations=2,max_depth=3,uncertainty='off',constant_optimization='off',evaluation_backend='numpy',random_state=3,verbose=0).fit(X,y)
    return X,y,m


def test_joblib_save_load_branch_and_reports(tmp_path):
    X,y,m=fitted(); p=tmp_path/'m.joblib';m.save(p);m2=FastSymbolicRegressor.load(p)
    assert np.allclose(m.predict(X[:10]),m2.predict(X[:10]))
    assert m.branch(generations=3).warm_start is True
    assert (tmp_path/'h.csv') == (tmp_path/'h.csv')
    m.save_history_csv(tmp_path/'h.csv');m.save_report_json(tmp_path/'r.json');m.save_expression(tmp_path/'e.txt')
    assert all((tmp_path/n).exists() for n in ['h.csv','r.json','e.txt'])


def test_python_export_matches_model(tmp_path):
    X,y,m=fitted();p=tmp_path/'model.py';m.export_python(p)
    spec=importlib.util.spec_from_file_location('exported_model',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert np.allclose(m.predict(X[:10]),mod.predict(X[:10]),rtol=1e-10,atol=1e-10)


def test_javascript_and_c_export_write_files(tmp_path):
    _,_,m=fitted();m.export_c(tmp_path/'m.c');m.export_cpp(tmp_path/'m.cpp');m.export_javascript(tmp_path/'m.js')
    assert all((tmp_path/n).stat().st_size>20 for n in ['m.c','m.cpp','m.js'])


def test_java_kotlin_fail_explicitly(tmp_path):
    _,_,m=fitted()
    with pytest.raises(ValueError,match='temporarily unavailable'):m.export_java(tmp_path/'m.java')
    with pytest.raises(ValueError,match='temporarily unavailable'):m.export_kotlin(tmp_path/'m.kt')
