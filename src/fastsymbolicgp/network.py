"""Experimental multilayer symbolic network retained from the V1 stable surface."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from .transformer import FastSymbolicTransformer

class FastSymbolicNetworkClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,symbolic_layers=(16,8),transformer_params=None,bridge_regularization=1.0,output_regularization=1.0,inherit_original_features=False,random_state=None,verbose=1):
        self.symbolic_layers=symbolic_layers;self.transformer_params=transformer_params;self.bridge_regularization=bridge_regularization;self.output_regularization=output_regularization;self.inherit_original_features=inherit_original_features;self.random_state=random_state;self.verbose=verbose
    def fit(self,X,y,sample_weight=None):
        X=np.asarray(X,float);y=np.asarray(y);self.classes_=np.unique(y);self.n_features_in_=X.shape[1];self.transformers_=[];self.scalers_=[];self.bridges_=[];current=X
        for i,width in enumerate(tuple(self.symbolic_layers)):
            params=dict(self.transformer_params or {});params.setdefault("random_state",None if self.random_state is None else int(self.random_state)+i*10007);params.setdefault("verbose",max(0,int(self.verbose)-1))
            t=FastSymbolicTransformer(n_components=int(width),task="classification",component_selection="mrmr",standardize_outputs=False,model_params=params,random_state=params["random_state"],verbose=params["verbose"])
            sym=t.fit_transform(current,y,sample_weight=sample_weight);sc=StandardScaler().fit(sym);sym=sc.transform(sym);self.transformers_.append(t);self.scalers_.append(sc)
            if i<len(tuple(self.symbolic_layers))-1:
                b=LogisticRegression(C=float(self.bridge_regularization),solver="lbfgs",max_iter=1000,random_state=self.random_state).fit(sym,y,sample_weight=sample_weight);self.bridges_.append(b);pieces=[sym,b.predict_proba(sym)];
                if self.inherit_original_features:pieces.insert(0,current)
                current=np.column_stack(pieces)
            else:current=sym
        self.output_model_=LogisticRegression(C=float(self.output_regularization),solver="lbfgs",max_iter=1000,random_state=self.random_state).fit(current,y,sample_weight=sample_weight)
        self.layer_expressions_=[t.get_expressions() for t in self.transformers_];self.network_stats_={"layers":len(self.transformers_),"symbolic_widths":[len(x) for x in self.layer_expressions_],"equations_total":sum(map(len,self.layer_expressions_)),"dag_nodes_total":sum(p.dag_size for t in self.transformers_ for p in t.programs_)};return self
    def _forward(self,X):
        current=np.asarray(X,float)
        for i,(t,sc) in enumerate(zip(self.transformers_,self.scalers_)):
            sym=sc.transform(t.transform(current))
            if i<len(self.bridges_):
                pieces=[sym,self.bridges_[i].predict_proba(sym)]
                if self.inherit_original_features:pieces.insert(0,current)
                current=np.column_stack(pieces)
            else:current=sym
        return current
    def predict_proba(self,X):check_is_fitted(self,"output_model_");return self.output_model_.predict_proba(self._forward(X))
    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
    def score(self,X,y,sample_weight=None):return accuracy_score(y,self.predict(X),sample_weight=sample_weight)
    def get_layer_expressions(self):check_is_fitted(self,"layer_expressions_");return [list(x) for x in self.layer_expressions_]
    def get_network_stats(self):check_is_fitted(self,"network_stats_");return dict(self.network_stats_)
