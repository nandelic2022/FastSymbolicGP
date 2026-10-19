import re
from sklearn.datasets import load_diabetes
from fastsymbolicgp import FastSymbolicRegressor


def _model(verbose=2, display="dashboard"):
    X,y=load_diabetes(return_X_y=True,as_frame=True)
    m=FastSymbolicRegressor(
        population_size=12,generations=3,tournament_size=3,
        max_depth=4,max_nodes=20,evaluation_backend="numpy",
        fitness_engine="materialized",optimization="scalar",
        uncertainty="off",constant_optimization="off",
        verbose=verbose,display=display,random_state=7,
    )
    return m,X.iloc[:100],y.iloc[:100]


def test_awesome_console_banner_and_finish(capsys):
    m,X,y=_model(verbose=1)
    m.fit(X,y)
    out=capsys.readouterr().out
    assert "FastSymbolicGP 1.2.8" in out
    assert "SYMBOLIC REGRESSION" in out
    assert "TRAINING COMPLETE" in out
    assert "Final equation" in out


def test_verbose_2_prints_every_generation(capsys):
    m,X,y=_model(verbose=2)
    m.fit(X,y)
    out=capsys.readouterr().out
    assert re.search(r"G\s+1/3",out)
    assert re.search(r"G\s+2/3",out)
    assert re.search(r"G\s+3/3",out)


def test_display_off_is_silent(capsys):
    m,X,y=_model(verbose=2,display="off")
    m.fit(X,y)
    out=capsys.readouterr().out
    assert "FastSymbolicGP" not in out
    assert "TRAINING COMPLETE" not in out
