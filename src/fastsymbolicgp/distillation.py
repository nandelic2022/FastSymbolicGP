"""Symbolic model distillation utilities retained from V1."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from .regressor import FastSymbolicRegressor

class DistilledSymbolicClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,population_size=64,generations=120,max_nodes=40,temperature=2.0,random_state=None,model_params=None):
        self.population_size=population_size;self.generations=generations;self.max_nodes=max_nodes;self.temperature=temperature;self.random_state=random_state;self.model_params=model_params
    def fit(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y);classes=np.unique(y);enc=np.searchsorted(classes,y);P=np.full((len(y),len(classes)),.02/max(1,len(classes)-1));P[np.arange(len(y)),enc]=.98
        class Teacher:
            def __init__(self,c,p):self.classes_=c;self.p=p
            def predict_proba(self,X):return self.p
            def get_expression_stats(self):return {"source":"smoothed_hard_labels"}
        return self.fit_from_teacher(Teacher(classes,P),X)
    def fit_from_teacher(self,teacher,X):
        X=np.asarray(X,float);P=np.clip(teacher.predict_proba(X),1e-7,1);P/=P.sum(axis=1,keepdims=True);self.classes_=np.asarray(teacher.classes_);self.n_features_in_=X.shape[1];self.teacher_expression_stats_=teacher.get_expression_stats() if hasattr(teacher,"get_expression_stats") else {"source":teacher.__class__.__name__}
        params=dict(self.model_params or {});params.setdefault("population_size",int(self.population_size));params.setdefault("generations",int(self.generations));params.setdefault("max_nodes",int(self.max_nodes));params.setdefault("max_depth",max(4,int(np.ceil(np.log2(max(2,int(self.max_nodes)))))+2));params.setdefault("optimization","pareto");params.setdefault("uncertainty","off");params.setdefault("random_state",self.random_state);params.setdefault("verbose",0)
        self.regressors_=[]
        if len(self.classes_)==2:
            p=P[:,1];target=np.log(p/(1-p))/max(float(self.temperature),1e-9);self.regressors_.append(FastSymbolicRegressor(**params).fit(X,target))
        else:
            logits=np.log(P);logits-=logits.mean(axis=1,keepdims=True);logits/=max(float(self.temperature),1e-9)
            for i in range(len(self.classes_)):
                q=dict(params);q["random_state"]=None if self.random_state is None else int(self.random_state)+i*4099;self.regressors_.append(FastSymbolicRegressor(**q).fit(X,logits[:,i]))
        self.expression_stats_={"models":len(self.regressors_),"nodes_total":sum(r.best_program_.size for r in self.regressors_),"dag_nodes_total":sum(r.best_program_.dag_size for r in self.regressors_),"depth_max":max(r.best_program_.depth for r in self.regressors_)};return self
    @staticmethod
    def _softmax(z):z=z-z.max(axis=1,keepdims=True);e=np.exp(np.clip(z,-50,50));return e/e.sum(axis=1,keepdims=True)
    def predict_proba(self,X):
        check_is_fitted(self,"regressors_")
        if len(self.classes_)==2:
            logit=self.regressors_[0].predict(X)*float(self.temperature);p=1/(1+np.exp(-np.clip(logit,-35,35)));return np.c_[1-p,p]
        return self._softmax(np.column_stack([r.predict(X) for r in self.regressors_])*float(self.temperature))
    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
    def score(self,X,y,sample_weight=None):return accuracy_score(y,self.predict(X),sample_weight=sample_weight)
    def get_expression(self):check_is_fitted(self,"regressors_");return self.regressors_[0].get_expression() if len(self.classes_)==2 else {str(c):r.get_expression() for c,r in zip(self.classes_,self.regressors_)}
    def get_expression_stats(self):check_is_fitted(self,"expression_stats_");return dict(self.expression_stats_)
