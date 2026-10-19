from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.benchmark import run_publication_benchmark
from fastsymbolicgp.publication import compare_algorithms, paired_rank_biserial, stability_report, robustness_report


def test_paired_rank_biserial_direction():
    x=np.array([5,6,7,8],float); y=np.array([1,2,3,4],float)
    assert paired_rank_biserial(x,y,higher_is_better=True) == 1.0
    assert paired_rank_biserial(x,y,higher_is_better=False) == -1.0


def test_compare_algorithms_uses_only_common_blocks():
    rows=[]
    for split in range(1,5):
        rows.append({'dataset':'d','split':split,'algorithm':'a','score':10+split})
        rows.append({'dataset':'d','split':split,'algorithm':'b','score':8+split})
        if split < 4:
            rows.append({'dataset':'d','split':split,'algorithm':'c','score':6+split})
    rep=compare_algorithms(rows,'score',higher_is_better=True)
    assert rep['n_common_blocks']==3
    assert rep['average_ranks']['a'] < rep['average_ranks']['b'] < rep['average_ranks']['c']
    assert len(rep['pairwise_wilcoxon'])==3


def test_publication_benchmark_symbolic_and_resume(tmp_path):
    kwargs=dict(output_dir=tmp_path,task='regression',datasets=('nguyen1',),algorithms=('fastsymbolicgp','linear'),runs=1,folds=2,population_size=10,generations=1,seed=7,fastsymbolic_params={'max_depth':3,'validation_fraction':0.2})
    first=run_publication_benchmark(**kwargs)
    assert len(first['raw_results'])==4
    assert not first['failures']
    second=run_publication_benchmark(**kwargs)
    assert len(second['raw_results'])==4
    assert (tmp_path/'statistical_comparison.json').exists()
    assert (tmp_path/'average_ranks.csv').exists()
    assert (tmp_path/'pairwise_wilcoxon_holm.csv').exists()


def _tiny_model(seed=1):
    rng=np.random.default_rng(1); X=rng.normal(size=(100,2)); y=2*X[:,0]-0.5*X[:,1]
    m=FastSymbolicRegressor(population_size=12,generations=2,max_depth=3,random_state=seed,verbose=0,uncertainty='off').fit(X,y)
    return m,X,y


def test_scientific_certificate_and_bundle_v2(tmp_path):
    m,X,y=_tiny_model()
    cert=m.get_scientific_certificate(X)
    assert cert['format']=='FastSymbolicGP-scientific-certificate-v1'
    assert cert['passed'] is True
    out=Path(m.export_reproducibility_bundle(tmp_path/'bundle',X=X,y=y))
    expected={'config.json','resolved_config.json','manifest.json','scientific_certificate.json','expression.txt','environment.json','SHA256SUMS.json','X_reference.npy','y_reference.npy','robustness.json'}
    assert expected.issubset({p.name for p in out.iterdir()})
    hashes=json.loads((out/'SHA256SUMS.json').read_text())
    assert 'manifest.json' in hashes and len(hashes['manifest.json'])==64


def test_stability_and_robustness_reports():
    m1,X,_=_tiny_model(3); m2,_,_=_tiny_model(4)
    s=stability_report([m1,m2],X[:20])
    assert s['n_models']==2
    assert s['unique_expressions']>=1
    assert s['prediction'] is not None
    r=robustness_report(m1,X[:20],noise_levels=(0.01,),repeats=2,random_state=5)
    assert len(r['rows'])==1
    assert r['rows'][0]['rmse_prediction_change']>=0


def test_save_scientific_certificate(tmp_path):
    m,X,_=_tiny_model()
    path=tmp_path/'certificate.json'
    assert m.save_scientific_certificate(path,X)==str(path)
    d=json.loads(path.read_text())
    assert d['version']=='1.2.8'
    assert d['passed'] is True
