from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from .regressor import FastSymbolicRegressor
from .classifier import FastSymbolicClassifier

class FastSymbolicEnsembleRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,n_estimators=5,aggregation="median",population_size=100,generations=60,preset=None,random_state=None,n_jobs=-1,verbose=0,**estimator_params):
        self.n_estimators=n_estimators; self.aggregation=aggregation; self.population_size=population_size; self.generations=generations; self.preset=preset; self.random_state=random_state; self.n_jobs=n_jobs; self.verbose=verbose; self.estimator_params=estimator_params
    def fit(self,X,y):
        self.estimators_=[]
        for i in range(int(self.n_estimators)):
            p=dict(self.estimator_params); p.update(population_size=self.population_size,generations=self.generations,preset=self.preset,random_state=(self.random_state or 0)+i*1009,n_jobs=self.n_jobs,verbose=self.verbose)
            self.estimators_.append(FastSymbolicRegressor(**p).fit(X,y))
        return self
    def predict(self,X):
        check_is_fitted(self,"estimators_"); P=np.column_stack([e.predict(X) for e in self.estimators_]); return np.median(P,axis=1) if self.aggregation=="median" else np.mean(P,axis=1)
    def score(self,X,y):
        from sklearn.metrics import r2_score; return r2_score(y,self.predict(X))

class FastSymbolicEnsembleClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,n_estimators=5,population_size=100,generations=60,preset=None,random_state=None,n_jobs=-1,verbose=0,**estimator_params):
        self.n_estimators=n_estimators; self.population_size=population_size; self.generations=generations; self.preset=preset; self.random_state=random_state; self.n_jobs=n_jobs; self.verbose=verbose; self.estimator_params=estimator_params
    def fit(self,X,y):
        self.estimators_=[]
        for i in range(int(self.n_estimators)):
            p=dict(self.estimator_params); p.update(population_size=self.population_size,generations=self.generations,preset=self.preset,random_state=(self.random_state or 0)+i*1009,n_jobs=self.n_jobs,verbose=self.verbose)
            self.estimators_.append(FastSymbolicClassifier(**p).fit(X,y))
        self.classes_=self.estimators_[0].classes_; return self
    def predict_proba(self,X): check_is_fitted(self,"estimators_"); return np.mean(np.stack([e.predict_proba(X) for e in self.estimators_]),axis=0)
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
