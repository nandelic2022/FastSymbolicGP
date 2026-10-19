import numpy as np
from sklearn.datasets import make_regression

from fastsymbolicgp import FastSymbolicRegressor


def _data():
    return make_regression(n_samples=140, n_features=4, noise=0.1, random_state=12)


def _model(**kwargs):
    params=dict(
        population_size=24, generations=4, max_depth=4, max_nodes=40,
        random_state=12, verbose=0, uncertainty='off', constant_optimization='off',
        evaluation_backend='numba', fitness_engine='auto', selection_method='tournament',
    )
    params.update(kwargs)
    return FastSymbolicRegressor(**params)


def test_v121_default_is_fast_scalar_but_pareto_reporting_survives():
    X,y=_data(); m=_model().fit(X,y)
    assert m.optimization == 'scalar'
    assert m.profile_['nsga2_calls'] == 0
    # No per-generation Pareto archive maintenance: one terminal reporting update only.
    assert m.profile_['pareto_archive_updates'] == 1
    assert len(m.get_pareto_front()) >= 1


def test_v121_pareto_ranks_once_per_generation():
    X,y=_data(); m=_model(optimization='pareto').fit(X,y)
    assert m.profile_['nsga2_calls'] == m.n_generations_
    assert m.profile_['pareto_archive_updates'] == m.n_generations_
    assert all('pareto_rank' in p.metadata for p in m.hall_of_fame_)


def test_v121_state_snapshot_is_not_cloned_every_generation():
    X,y=_data(); m=_model(generations=5).fit(X,y)
    assert m.profile_['state_snapshot_count'] == 1
    assert m.evolution_state_['generation'] == 5


def test_v121_checkpoint_only_snapshots_checkpoint_generations_plus_terminal(tmp_path):
    X,y=_data(); path=tmp_path/'fast.chk'
    m=_model(generations=5, checkpoint_path=str(path), checkpoint_interval=2).fit(X,y)
    assert path.exists()
    assert m.profile_['state_snapshot_count'] == 3  # generations 2, 4, and terminal 5
    assert m.evolution_state_['generation'] == 5


def test_v121_profile_exposes_hot_path_timings():
    X,y=_data(); m=_model(optimization='pareto').fit(X,y)
    stages=m.profile_['stage_seconds']
    for key in ('evaluation','nsga2','pareto_archive','state_snapshot','variation'):
        assert key in stages and np.isfinite(stages[key]) and stages[key] >= 0
