"""Supervised symbolic feature construction with V1-compatible API."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from .classifier import FastSymbolicClassifier
from .regressor import FastSymbolicRegressor

class FastSymbolicTransformer(TransformerMixin, BaseEstimator):
    def __init__(self,n_components=10,task="auto",component_selection="mrmr",max_correlation=.90,redundancy_penalty=.10,
                 include_original_features=False,standardize_outputs=True,model_params=None,
                 population_size=80,generations=40,max_depth=7,function_set=("add","sub","mul","div"),
                 selection_method="epsilon_lexicase",random_state=None,n_jobs=-1,verbose=0):
        self.n_components=n_components;self.task=task;self.component_selection=component_selection;self.max_correlation=max_correlation;self.redundancy_penalty=redundancy_penalty
        self.include_original_features=include_original_features;self.standardize_outputs=standardize_outputs;self.model_params=model_params
        self.population_size=population_size;self.generations=generations;self.max_depth=max_depth;self.function_set=function_set;self.selection_method=selection_method;self.random_state=random_state;self.n_jobs=n_jobs;self.verbose=verbose
    def fit(self,X,y,sample_weight=None):
        names=getattr(X,"columns",None);A=np.asarray(X,float);y=np.asarray(y);self.n_features_in_=A.shape[1];self.feature_names_in_=np.asarray(names if names is not None else [f"X{i}" for i in range(A.shape[1])],dtype=object)
        target=type_of_target(y);task=str(self.task).lower()
        if task=="auto":
            nunique=len(np.unique(y)); task="classification" if target in {"binary","multiclass"} and nunique<=max(20,int(np.sqrt(len(y)))) else "regression"
        params=dict(self.model_params or {});params.setdefault("population_size",int(self.population_size));params.setdefault("generations",int(self.generations));params.setdefault("max_depth",int(self.max_depth));params.setdefault("function_set",self.function_set);params.setdefault("selection_method",self.selection_method);params.setdefault("random_state",self.random_state);params.setdefault("n_jobs",self.n_jobs);params.setdefault("verbose",self.verbose)
        if np.isnan(A).any():params.setdefault("missing_value_strategy","median")
        if task=="classification": self.estimator_=FastSymbolicClassifier(**params).fit(A,y,sample_weight=sample_weight)
        elif task=="regression": params.setdefault("uncertainty","off"); self.estimator_=FastSymbolicRegressor(**params).fit(A,np.asarray(y,float),sample_weight=sample_weight)
        else: raise ValueError("task must be auto, classification, or regression")
        self.task_=task
        candidates=[]
        if getattr(self.estimator_,"_is_multiclass_",False):
            for e in self.estimator_.estimators_: candidates.extend(getattr(e,"hall_of_fame_",[]) or [e.best_program_])
        else: candidates.extend(getattr(self.estimator_,"hall_of_fame_",[]) or [self.estimator_.best_program_])
        unique={}
        for p in candidates:
            key=p.structural_hash();old=unique.get(key)
            if old is None or p.validation_loss<old.validation_loss:unique[key]=p
        programs=sorted(unique.values(),key=lambda p:(p.validation_loss+p.constraint_penalty,p.size,p.depth))
        outputs=[];valid=[]
        for p in programs:
            z=p.raw(A)
            if np.isfinite(z).all() and np.std(z)>1e-12:outputs.append(z);valid.append(p)
        if not valid: valid=[self.estimator_.best_program_];outputs=[valid[0].raw(A)]
        M=np.column_stack(outputs);rel=self._relevance(M,y,task);idx=self._select_components(M,rel,valid)
        self.programs_=[valid[i].clone() for i in idx];S=M[:,idx];self.component_relevance_=rel[idx]
        self.symbolic_mean_=S.mean(axis=0);self.symbolic_scale_=np.where(S.std(axis=0)>1e-12,S.std(axis=0),1.0);self.n_components_=len(self.programs_)
        sym=[f"symbolic_{i:03d}" for i in range(self.n_components_)]
        self.feature_names_out_=np.asarray((list(self.feature_names_in_)+sym) if self.include_original_features else sym,dtype=object)
        self.component_report_=[{"name":sym[i],"relevance":float(self.component_relevance_[i]),"nodes":p.size,"dag_nodes":p.dag_size,"depth":p.depth,"expression":p.to_string(self.feature_names_in_)} for i,p in enumerate(self.programs_)]
        return self
    def _relevance(self,M,y,task):
        try:return np.asarray(mutual_info_classif(M,y,random_state=self.random_state) if task=="classification" else mutual_info_regression(M,np.asarray(y,float),random_state=self.random_state),float)
        except Exception:
            yn=np.asarray(y,float);return np.asarray([0.0 if not np.isfinite(np.corrcoef(M[:,j],yn)[0,1]) else abs(np.corrcoef(M[:,j],yn)[0,1]) for j in range(M.shape[1])])
    def _select_components(self,M,rel,programs):
        remaining=set(range(M.shape[1]));selected=[];method=str(self.component_selection).lower()
        while remaining and len(selected)<int(self.n_components):
            best=None;score=-np.inf
            for i in remaining:
                cors=[abs(float(np.corrcoef(M[:,i],M[:,j])[0,1])) for j in selected] if selected else []
                cors=[0.0 if not np.isfinite(c) else c for c in cors]
                if cors and max(cors)>float(self.max_correlation):continue
                red=float(np.mean(cors)) if cors else 0.0
                if method=="pareto":s=-float(programs[i].validation_loss)-1e-3*programs[i].dag_size
                elif method in {"diversity_validation","diversity"}:s=-float(programs[i].validation_loss)-float(self.redundancy_penalty)*red
                elif method=="mrmr":s=float(rel[i])-float(self.redundancy_penalty)*red
                else:raise ValueError("component_selection must be pareto, diversity_validation, or mrmr")
                if s>score:best,score=i,s
            if best is None:best=max(remaining,key=lambda i:float(rel[i]))
            selected.append(best);remaining.remove(best)
        return selected
    def transform(self,X):
        check_is_fitted(self,"programs_");A=np.asarray(X,float)
        if A.ndim!=2 or A.shape[1]!=self.n_features_in_:raise ValueError(f"Expected {self.n_features_in_} features")
        if np.isnan(A).any():
            fill=getattr(self.estimator_,"missing_fill_values_",None)
            A=np.nan_to_num(A,nan=0.0) if fill is None else np.where(np.isnan(A),np.asarray(fill,float),A)
        S=np.column_stack([p.raw(A) for p in self.programs_])
        if self.standardize_outputs:S=(S-self.symbolic_mean_)/self.symbolic_scale_
        return np.column_stack((A,S)) if self.include_original_features else S
    def get_feature_names_out(self,input_features=None):check_is_fitted(self,"feature_names_out_");return self.feature_names_out_.copy()
    def get_expressions(self):check_is_fitted(self,"programs_");return [p.to_string(self.feature_names_in_) for p in self.programs_]
    def get_component_report(self):check_is_fitted(self,"component_report_");return list(self.component_report_)
