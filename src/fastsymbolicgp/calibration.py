"""Probability calibration helpers retained from the V1 stable surface."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold

class PlattCalibrator(BaseEstimator):
    def __init__(self,C=1.0,random_state=None): self.C=C;self.random_state=random_state
    def fit(self,p,y,sample_weight=None): self.model_=LogisticRegression(C=float(self.C),solver="lbfgs",max_iter=500,random_state=self.random_state).fit(np.asarray(p).reshape(-1,1),y,sample_weight=sample_weight);return self
    def predict(self,p): return self.model_.predict_proba(np.asarray(p).reshape(-1,1))[:,1]
class BetaCalibrator(BaseEstimator):
    def __init__(self,C=1.0,random_state=None): self.C=C;self.random_state=random_state
    @staticmethod
    def _features(p): p=np.clip(np.asarray(p,float),1e-9,1-1e-9);return np.c_[np.log(p),-np.log1p(-p)]
    def fit(self,p,y,sample_weight=None): self.model_=LogisticRegression(C=float(self.C),solver="lbfgs",max_iter=500,random_state=self.random_state).fit(self._features(p),y,sample_weight=sample_weight);return self
    def predict(self,p): return self.model_.predict_proba(self._features(p))[:,1]
class TemperatureScaler(BaseEstimator):
    def __init__(self,min_temperature=.05,max_temperature=20):self.min_temperature=min_temperature;self.max_temperature=max_temperature
    @staticmethod
    def _softmax(z): z=z-z.max(axis=1,keepdims=True);e=np.exp(np.clip(z,-50,50));return e/e.sum(axis=1,keepdims=True)
    def fit(self,probabilities,y,sample_weight=None):
        p=np.clip(np.asarray(probabilities,float),1e-12,1);p/=p.sum(axis=1,keepdims=True);classes=np.unique(y);enc=np.searchsorted(classes,y);w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,float);lp=np.log(p)
        grid=np.exp(np.linspace(np.log(self.min_temperature),np.log(self.max_temperature),121));loss=[]
        for t in grid:
            q=self._softmax(lp/t);loss.append(-np.sum(w*np.log(np.clip(q[np.arange(len(y)),enc],1e-12,1)))/w.sum())
        self.temperature_=float(grid[int(np.argmin(loss))]);self.classes_=classes;return self
    def predict_proba(self,p): p=np.clip(np.asarray(p,float),1e-12,1);p/=p.sum(axis=1,keepdims=True);return self._softmax(np.log(p)/self.temperature_)

def _make(method,random_state=None):
    m=str(method).lower()
    if m in {"platt","sigmoid","cross_fitted_platt"}:return PlattCalibrator(random_state=random_state)
    if m=="beta":return BetaCalibrator(random_state=random_state)
    if m=="isotonic":return IsotonicRegression(out_of_bounds="clip")
    if m in {"none","identity"}:return None
    raise ValueError(f"Unsupported probability calibration: {method}")
def apply_calibrator(model,p): return np.asarray(p,float) if model is None else np.asarray(model.predict(p),float)
def fit_calibrator(p,y,method="none",sample_weight=None,cv=3,random_state=None):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9);y=np.asarray(y,int);cands=["none","platt","isotonic","beta"] if str(method).lower()=="auto" else [str(method).lower()];reports={};best=(np.inf,"none")
    min_class=int(np.bincount(y).min()) if len(y) else 0;n_splits=max(2,min(int(cv),min_class)) if min_class>=2 else 0
    for cand in cands:
        if cand=="none" or n_splits==0:oof=p.copy()
        else:
            oof=np.empty_like(p);sp=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
            for tr,va in sp.split(p,y):
                m=_make(cand,random_state);m.fit(p[tr],y[tr],sample_weight=None if sample_weight is None else np.asarray(sample_weight)[tr]);oof[va]=apply_calibrator(m,p[va])
        ll=float(log_loss(y,np.clip(oof,1e-6,1-1e-6),sample_weight=sample_weight,labels=[0,1]));reports[cand]={"oof_log_loss":ll,"oof_brier_score":float(brier_score_loss(y,np.clip(oof,1e-6,1-1e-6),sample_weight=sample_weight))}
        if ll<best[0]:best=(ll,cand)
    selected=best[1];final=_make(selected,random_state)
    if final is not None:final.fit(p,y,sample_weight=sample_weight)
    return final,{"requested_method":method,"selected_method":selected,"cv_splits":n_splits,"candidates":reports},p
