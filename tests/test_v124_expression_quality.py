import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.program import Node, Program
from fastsymbolicgp.simplification import safe_simplify_program, certified_simplify_program


def test_safe_rules_respect_protected_division_semantics():
    x=Node("feature",0)
    same=Program(Node("func","div",(x,x)))
    q=safe_simplify_program(same)
    assert q.size==1 and q.root.kind=="const" and float(q.root.value)==1.0

    x_over_zero=Program(Node("func","div",(x,Node("const",0.0))))
    q0=safe_simplify_program(x_over_zero)
    assert q0.size==1 and float(q0.root.value)==1.0

    zero_over_x=Program(Node("func","div",(Node("const",0.0),x)))
    q2=safe_simplify_program(zero_over_x)
    # 0/x is NOT globally zero because protected division returns 1 at x=0.
    assert q2.root.kind=="func" and q2.root.value=="div"


def test_certified_pruning_can_remove_empirically_safe_protected_artifact():
    X=np.linspace(1.0,2.0,80).reshape(-1,1)
    p=Program(Node("func","add",(
        Node("feature",0),
        Node("func","div",(Node("const",0.0),Node("feature",0)))
    )),scale=2.0,intercept=3.0)
    q,report=certified_simplify_program(p,X,atol=1e-12,rtol=1e-12)
    assert report["passed"] is True
    assert q.size < p.size
    assert np.allclose(p.predict(X),q.predict(X),rtol=1e-12,atol=1e-12)


def test_fitted_equation_contains_affine_scaling_and_raw_equation_is_available():
    X,y=load_diabetes(return_X_y=True,as_frame=True)
    Xtr,_,ytr,_=train_test_split(X,y,test_size=.2,random_state=42)
    m=FastSymbolicRegressor(population_size=24,generations=2,random_state=2,verbose=0,
        constant_optimization="off",uncertainty="none",final_simplification="safe").fit(Xtr,ytr)
    raw=m.get_raw_expression(); eq=m.get_expression()
    assert raw in eq or eq==raw
    if abs(float(m.best_program_.scale)-1.0)>1e-15 or abs(float(m.best_program_.intercept))>1e-15:
        assert eq != raw


def test_end_to_end_simplification_report_is_exposed():
    X,y=load_diabetes(return_X_y=True,as_frame=True)
    m=FastSymbolicRegressor(population_size=30,generations=3,random_state=3,verbose=0,
        constant_optimization="champion",uncertainty="none",final_simplification="certified").fit(X.iloc[:180],y.iloc[:180])
    r=m.get_simplification_report()
    assert r["passed"] is True
    assert r["final_nodes"] <= r["original_nodes"]
    assert r["nodes_removed"] >= 0
    assert "max_abs_prediction_delta" in r


def test_v124_parameters_are_cloneable():
    m=FastSymbolicRegressor(final_simplification="certified",simplification_tolerance=1e-9,
        simplification_reference_points=777)
    c=clone(m)
    assert c.final_simplification=="certified"
    assert c.simplification_tolerance==1e-9
    assert c.simplification_reference_points==777
