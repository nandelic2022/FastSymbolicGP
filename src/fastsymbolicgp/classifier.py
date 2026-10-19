from __future__ import annotations
import os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.utils.validation import check_is_fitted
from .base import BaseFastSymbolicGP
from .metrics import classification_loss, fit_logistic_scaler, fit_logistic_scaler_approx, sigmoid
from .primitives import validate_function_set
from .serialization import program_to_latex, save_model_json, load_model_json
from .calibration import fit_calibrator, apply_calibrator, TemperatureScaler
from ._version import __version__
from .backend import (NUMBA_AVAILABLE, SUPPORTED as NUMBA_SUPPORTED_PRIMITIVES,
                      FUSED_CLASSIFICATION_METRICS, score_population_classification_postfix,
                      estimate_fused_population_memory_mb, choose_population_batch_size)

class FastSymbolicClassifier(ClassifierMixin, BaseFastSymbolicGP):
    def __init__(self, *, fitness_metric="log_loss", metric=None, link_function="sigmoid", fitness_scaling="elite_logistic",
                 fitness_scaling_top_k=12, fitness_scaling_interval=2, fitness_scaling_regularization=1e-4,
                 fitness_scaling_warm_start=True, fitness_scaling_fallback="rank", validation_metric=None,
                 class_weight=None, probability_calibration="none", calibration_cv=3, probability_clip=1e-6,
                 threshold_strategy="mcc", threshold_validation_fraction=None, multiclass_strategy="ovr",
                 multiclass_calibration="none", shared_n_components=12, shared_max_correlation=.95, shared_regularization=1.0,
                 prediction_mode="best", ensemble_size=5, ensemble_weighting="optimized", max_ensemble_correlation=.97,
                 prediction_sets=True, calibration_fraction=.15,
                 population_size=500,generations=50,tournament_size=20,function_set=("add","sub","mul","div"),
                 init_depth=(2,6),max_depth=12,max_nodes=255,p_crossover=.70,p_subtree_mutation=.15,p_point_mutation=.10,
                 const_range=(-1.0,1.0),const_probability=.15,constant_probability=None,max_samples=1.0,parsimony_coefficient=0.0,
                 validation_fraction=.20,patience=20,min_delta=1e-6,elitism=2,selection_tolerance=.02,exact_champion=True,
                 exact_champion_candidates=4,selection_method="tournament",epsilon_lexicase_epsilon="mad",
                 random_immigrants=.05,restart_strategy="adaptive",restart_fraction=.25,diversity_threshold=.08,
                 simplification="safe",final_simplification="certified",simplification_tolerance=1e-10,simplification_reference_points=2048,constraints=None,feature_units=None,target_unit=None,unit_policy="off",constant_units=None,
                 optimization="scalar",pareto_archive_size=256,pareto_objectives=("validation_loss","nodes","constraint_penalty"),n_jobs=-1,time_budget=None,evaluation_budget=None,
                 memory_budget_mb=None,evaluation_backend="auto",fitness_engine="auto",evolution_model="panmictic",
                 n_islands=1,migration_interval=10,migration_size=2,island_parallel=True,evaluation_cache=True,fitness_cache_max_entries=100000,typed_generation="auto",
                 semantic_diversity=False,semantic_archive_size=64,semantic_sample_size=128,semantic_neighbors=5,semantic_weight=0.05,semantic_mutation_rate=0.0,semantic_mutation_trials=4,semantic_step_target=0.50,
                 age_management="off",age_gap=10,age_layers=4,memetic_search="off",memetic_elites=2,memetic_interval=5,memetic_budget=120,memetic_method="hybrid",model_selection="best_validation",
                 ovr_parallel=True,ovr_n_jobs="auto",low_memory=False,missing_value_strategy="error",
                 missing_value_constant=0.0,preset=None,preset_override_policy="defaults_only",deterministic=True,
                 checkpoint_path=None,checkpoint_interval=0,resume_from_checkpoint=False,warm_start=False,
                 random_state=None,verbose=1,display="dashboard",profile_fit=True):
        if constant_probability is not None: const_probability=float(constant_probability)
        super().__init__(population_size=population_size,generations=generations,tournament_size=tournament_size,
            function_set=function_set,init_depth=init_depth,max_depth=max_depth,max_nodes=max_nodes,p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,p_point_mutation=p_point_mutation,const_range=const_range,
            const_probability=const_probability,max_samples=max_samples,parsimony_coefficient=parsimony_coefficient,
            validation_fraction=validation_fraction,patience=patience,min_delta=min_delta,elitism=elitism,
            selection_tolerance=selection_tolerance,exact_champion=exact_champion,exact_champion_candidates=exact_champion_candidates,
            selection_method=selection_method,epsilon_lexicase_epsilon=epsilon_lexicase_epsilon,random_immigrants=random_immigrants,
            restart_strategy=restart_strategy,restart_fraction=restart_fraction,diversity_threshold=diversity_threshold,
            simplification=simplification,final_simplification=final_simplification,simplification_tolerance=simplification_tolerance,simplification_reference_points=simplification_reference_points,constraints=constraints,feature_units=feature_units,target_unit=target_unit,
            unit_policy=unit_policy,constant_units=constant_units,optimization=optimization,pareto_archive_size=pareto_archive_size,pareto_objectives=pareto_objectives,n_jobs=n_jobs,
            time_budget=time_budget,evaluation_budget=evaluation_budget,memory_budget_mb=memory_budget_mb,
            evaluation_backend=evaluation_backend,fitness_engine=fitness_engine,evolution_model=evolution_model,n_islands=n_islands,
            migration_interval=migration_interval,migration_size=migration_size,island_parallel=island_parallel,evaluation_cache=evaluation_cache,fitness_cache_max_entries=fitness_cache_max_entries,typed_generation=typed_generation,semantic_diversity=semantic_diversity,semantic_archive_size=semantic_archive_size,semantic_sample_size=semantic_sample_size,semantic_neighbors=semantic_neighbors,semantic_weight=semantic_weight,semantic_mutation_rate=semantic_mutation_rate,semantic_mutation_trials=semantic_mutation_trials,semantic_step_target=semantic_step_target,age_management=age_management,age_gap=age_gap,age_layers=age_layers,memetic_search=memetic_search,memetic_elites=memetic_elites,memetic_interval=memetic_interval,memetic_budget=memetic_budget,memetic_method=memetic_method,model_selection=model_selection,low_memory=low_memory,
            missing_value_strategy=missing_value_strategy,missing_value_constant=missing_value_constant,preset=preset,
            preset_override_policy=preset_override_policy,deterministic=deterministic,checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,resume_from_checkpoint=resume_from_checkpoint,warm_start=warm_start,
            random_state=random_state,verbose=verbose,display=display,profile_fit=profile_fit)
        self.constant_probability=constant_probability
        self.fitness_metric=fitness_metric;self.metric=metric;self.link_function=link_function;self.fitness_scaling=fitness_scaling
        self.fitness_scaling_top_k=fitness_scaling_top_k;self.fitness_scaling_interval=fitness_scaling_interval
        self.fitness_scaling_regularization=fitness_scaling_regularization;self.fitness_scaling_warm_start=fitness_scaling_warm_start;self.fitness_scaling_fallback=fitness_scaling_fallback
        self.validation_metric=validation_metric;self.class_weight=class_weight;self.probability_calibration=probability_calibration;self.calibration_cv=calibration_cv;self.probability_clip=probability_clip
        self.threshold_strategy=threshold_strategy;self.threshold_validation_fraction=threshold_validation_fraction;self.multiclass_strategy=multiclass_strategy;self.multiclass_calibration=multiclass_calibration
        self.shared_n_components=shared_n_components;self.shared_max_correlation=shared_max_correlation;self.shared_regularization=shared_regularization
        self.prediction_mode=prediction_mode;self.ensemble_size=ensemble_size;self.ensemble_weighting=ensemble_weighting;self.max_ensemble_correlation=max_ensemble_correlation
        self.prediction_sets=prediction_sets;self.calibration_fraction=calibration_fraction
        self.ovr_parallel=ovr_parallel;self.ovr_n_jobs=ovr_n_jobs

    def _resolve_fitness_engine(self, requested_engine):
        metric=str(self.metric or self.fitness_metric).lower()
        val_metric=metric if self.validation_metric is None else str(self.validation_metric).lower()
        scaling=str(self.fitness_scaling or "elite_logistic").lower()
        metric_ok=metric in FUSED_CLASSIFICATION_METRICS and val_metric in FUSED_CLASSIFICATION_METRICS
        scaling_ok=scaling in {"approx_logistic","elite_logistic","auto"}
        primitives_ok=all(str(name) in NUMBA_SUPPORTED_PRIMITIVES for name in getattr(self,"function_set_",self.function_set))
        native_missing=str(self.missing_value_strategy or "error").lower()=="native"
        compatible=bool(NUMBA_AVAILABLE and metric_ok and scaling_ok and primitives_ok and not native_missing)
        if compatible and requested_engine in {"auto","fused"}: return "fused",None
        reasons=[]
        if not NUMBA_AVAILABLE: reasons.append("Numba is unavailable")
        if not metric_ok: reasons.append(f"metric {metric!r}/validation metric {val_metric!r} is not fused-compatible")
        if not scaling_ok: reasons.append(f"scaling {scaling!r} is not fused-compatible")
        if not primitives_ok: reasons.append("function_set contains custom/non-Numba primitives")
        if native_missing: reasons.append("native missing-value primitives require materialized NumPy evaluation")
        reason="; ".join(reasons) or "materialized engine explicitly requested"
        if requested_engine=="fused": raise ValueError("Forced fused classification is incompatible: "+reason+". Use fitness_engine='auto' for safe fallback.")
        return "materialized",(reason if requested_engine=="auto" else None)

    def _evaluate_population(self,pop,Xtr,ytr,Xv,yv,swtr=None,swv=None):
        if self.resolved_config_.get("fitness_engine")!="fused":
            return super()._evaluate_population(pop,Xtr,ytr,Xv,yv,swtr,swv)
        missing=self._apply_fitness_cache(pop,Xtr,ytr,Xv,yv)
        if missing:
            metric=self.metric or self.fitness_metric; val_metric=metric if self.validation_metric is None else self.validation_metric
            scales,intercepts,train_losses,val_losses=score_population_classification_postfix(
                Xtr,ytr,Xv,yv,missing,metric=metric,validation_metric=val_metric,scaling=self.fitness_scaling,
                regularization=float(self.fitness_scaling_regularization),sample_weight_train=swtr,sample_weight_val=swv,
                n_jobs=self.n_jobs,memory_budget_mb=self.memory_budget_mb)
            missing_ids={id(p) for p in missing}
            for i,p in enumerate(missing):
                p.scale=float(scales[i]);p.intercept=float(intercepts[i]);p.loss=float(train_losses[i]);p.validation_loss=float(val_losses[i])
                p.constraint_penalty=float(self._constraint_penalty(p,Xtr));p.metadata["evaluation_backend"]="numba";p.metadata["fitness_engine"]="fused";p.metadata["scaling_phase"]="approximate"
            self.evaluation_count_+=len(missing)
            # Fused approximate screening followed by exact logistic refinement of
            # newly-evaluated elite candidates. Cached candidates are already exact.
            if str(self.fitness_scaling or "").lower() in {"elite_logistic","auto"}:
                k=max(1,min(int(self.fitness_scaling_top_k),len(pop)))
                provisional=sorted(pop,key=lambda p:(p.validation_loss+p.constraint_penalty,p.size))[:k]
                for p in provisional:
                    if id(p) not in missing_ids: continue
                    raw=p.raw(Xtr);p.scale,p.intercept=fit_logistic_scaler(raw,ytr,swtr,regularization=float(self.fitness_scaling_regularization))
                    p.loss=float(self._loss(ytr,self._prediction_from_program(p,Xtr),swtr))
                    p.validation_loss=float(self._validation_loss(yv,self._prediction_from_program(p,Xv),swv)) if Xv is not None else p.loss
                    p.constraint_penalty=float(self._constraint_penalty(p,Xtr));p.metadata["scaling_phase"]="elite_exact"
            for p in missing: self._store_fitness_cache(p)
        estimate=estimate_fused_population_memory_mb(missing or pop,n_jobs=self.n_jobs);batch=choose_population_batch_size(missing or pop,self.memory_budget_mb,n_jobs=self.n_jobs)
        self.execution_plan_["estimated_fused_population_memory_mb"]=float(estimate);self.execution_plan_["population_batch_size"]=int(batch);self.execution_plan_["memory_budget_enforced"]=bool(self.memory_budget_mb is not None and batch<len(missing or pop))
        self.execution_plan_["duplicate_programs_skipped_last_batch"]=int(len(pop)-len(missing))
        return pop

    def validate_configuration(self):
        self._validate_common_configuration();scaling=str(self.fitness_scaling or "logistic").lower()
        if scaling not in {"logistic","auto","approx_logistic","elite_logistic","exact_logistic"}: raise ValueError("Classification fitness_scaling must be logistic/auto/approx_logistic/elite_logistic")
        if str(self.multiclass_strategy).lower() not in {"ovr","one_vs_rest","one-vs-rest"}: raise ValueError("V1.1.4 supports multiclass_strategy='ovr'")
        return {"valid":True,"fitness_scaling":scaling,"fitness_metric":self.metric or self.fitness_metric}

    def _weights(self,y):
        if self.class_weight is None:return np.ones(len(y),float)
        if self.class_weight=="balanced":
            vals,c=np.unique(y,return_counts=True);mp={v:len(y)/(len(vals)*n) for v,n in zip(vals,c)};return np.asarray([mp[v] for v in y],float)
        if isinstance(self.class_weight,dict):return np.asarray([float(self.class_weight.get(v,1)) for v in y],float)
        raise ValueError("class_weight must be None, 'balanced', or dict")
    def _loss(self,y,proba,sample_weight=None):return classification_loss(y,proba,self.metric or self.fitness_metric,sample_weight)
    def _validation_loss(self,y,proba,sample_weight=None):return classification_loss(y,proba,self.validation_metric or self.metric or self.fitness_metric,sample_weight)
    def _per_case_loss(self,y,proba):
        p=np.clip(np.asarray(proba,float),1e-12,1-1e-12);y=np.asarray(y,float);return -(y*np.log(p)+(1-y)*np.log1p(-p))
    def _fit_program_scaler(self,p,X,y,sw=None):
        mode=str(self.fitness_scaling or "logistic").lower();raw=p.raw(X)
        if mode=="approx_logistic":p.scale,p.intercept=fit_logistic_scaler_approx(raw,y,sw,regularization=float(self.fitness_scaling_regularization))
        else:p.scale,p.intercept=fit_logistic_scaler(raw,y,sw,regularization=float(self.fitness_scaling_regularization))
    def _prediction_from_program(self,p,X):return sigmoid(p.scale*p.raw(X)+p.intercept)

    def _threshold_score(self,y,pred,weights):
        m=str(self.threshold_strategy).lower()
        if m in {"balanced_accuracy","bacc"}:return balanced_accuracy_score(y,pred,sample_weight=weights)
        if m in {"f1","f1_score"}:return f1_score(y,pred,sample_weight=weights,zero_division=0)
        if m in {"accuracy","acc"}:return accuracy_score(y,pred,sample_weight=weights)
        return matthews_corrcoef(y,pred,sample_weight=weights)

    def _fit_binary(self,X,y,sample_weight=None):
        self._fit_n_samples_=len(X);self.function_set_=validate_function_set(self.function_set);self._resolve_preset();self.decision_threshold_=0.5
        self._console_start("SYMBOLIC CLASSIFICATION")
        idx=np.arange(len(X));vf=float(self.validation_fraction)
        if vf>0:
            tr,va=train_test_split(idx,test_size=vf,random_state=self.random_state,stratify=y);Xtr,ytr=X[tr],y[tr];Xv,yv=X[va],y[va];swtr=None if sample_weight is None else sample_weight[tr];swv=None if sample_weight is None else sample_weight[va]
        else:Xtr,ytr=X,y;Xv=yv=None;swtr=sample_weight;swv=None
        self._evolve(Xtr,ytr,Xv,yv,swtr,swv)
        # Refit scaler on all available binary data, then choose threshold on validation when available.
        self._fit_program_scaler(self.best_program_,X,y,sample_weight);self.scale_=self.best_program_.scale;self.intercept_=self.best_program_.intercept
        Xt,yt,wt=(Xv,yv,swv) if Xv is not None else (X,y,sample_weight);self.threshold_source_="validation" if Xv is not None else "training_fallback"
        raw_p=self._prediction_from_program(self.best_program_,Xt);candidates=np.unique(np.quantile(raw_p,np.linspace(.02,.98,99)));best=(.5,-np.inf)
        for t in candidates:
            s=self._threshold_score(yt,(raw_p>=t).astype(int),wt)
            if s>best[1]:best=(float(t),float(s))
        self.decision_threshold_=best[0]
        # Optional cross-fitted probability calibration. Calibrate on validation if available to avoid pure resubstitution.
        self.calibration_model_=None;self.calibration_report_={"requested_method":self.probability_calibration,"selected_method":"none"}
        if str(self.probability_calibration).lower() not in {"none","off","identity"}:
            cp=self._prediction_from_program(self.best_program_,Xt);self.calibration_model_,self.calibration_report_,_=fit_calibrator(cp,yt,method=self.probability_calibration,sample_weight=wt,cv=self.calibration_cv,random_state=self.random_state)
        return self

    def _resolve_ovr_workers(self,n_classes):
        if not bool(self.ovr_parallel) or int(n_classes)<=1: return 1
        value=self.ovr_n_jobs
        if value is None or str(value).lower()=="auto":
            return max(1,min(int(n_classes),int(os.cpu_count() or 1)))
        value=int(value)
        if value<0: return max(1,min(int(n_classes),int(os.cpu_count() or 1)))
        return max(1,min(int(n_classes),value))

    def fit(self,X,y,sample_weight=None):
        self.validate_configuration();X=self._prepare_X(X,fit=True);y=np.asarray(y)
        if sample_weight is not None:
            sample_weight=np.asarray(sample_weight,float)
            if sample_weight.ndim!=1 or len(sample_weight)!=len(X):raise ValueError("sample_weight must be one-dimensional with length n_samples")
        self.classes_=np.unique(y)
        if len(self.classes_)<2:raise ValueError("At least two classes required")
        if len(self.classes_)>2:
            self._is_multiclass_=True; workers=self._resolve_ovr_workers(len(self.classes_)); wall_start=time.perf_counter()
            def _fit_one(item):
                i,c=item; params=self.get_params(deep=False).copy()
                params["random_state"]=(self.random_state or 0)+i*1009;params["verbose"]=max(0,int(self.verbose)-1);params["multiclass_calibration"]="none"
                params["ovr_parallel"]=False; params["ovr_n_jobs"]=1
                if workers>1: params["n_jobs"]=1
                if self.checkpoint_path:
                    from pathlib import Path as _P
                    cp=_P(self.checkpoint_path); params["checkpoint_path"]=str(cp.with_name(cp.stem+f".class_{i}"+cp.suffix))
                e=FastSymbolicClassifier(**params);e._prepare_X(X,fit=True);yy=(y==c).astype(int);cw=e._weights(yy);sw=cw if sample_weight is None else cw*np.asarray(sample_weight,float);e._fit_binary(X,yy,sw);e._finalize_fit(X,yy);return e
            items=list(enumerate(self.classes_))
            if workers>1:
                with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="fsgp-ovr") as pool:
                    self.estimators_=list(pool.map(_fit_one,items))
            else:
                self.estimators_=[_fit_one(item) for item in items]
            wall=time.perf_counter()-wall_start
            self.best_program_=self.estimators_[0].best_program_;self.best_expression_={str(c):e.get_expression() for c,e in zip(self.classes_,self.estimators_)};self.history_=[];self.pareto_archive_=[]
            self.profile_={"version":__version__,"strategy":"ovr","classes":len(self.classes_),"runtime_seconds":float(wall),"child_runtime_seconds_sum":sum(e.profile_["runtime_seconds"] for e in self.estimators_),"ovr_workers":int(workers),"nested_child_n_jobs":1 if workers>1 else self.n_jobs}
            self.resolved_config_={"version":__version__,"preset":self.preset or "custom","multiclass_strategy":"ovr","ovr_workers":int(workers),"ovr_parallel":bool(workers>1)}
            self.multiclass_calibrator_=None
            if str(self.multiclass_calibration).lower()=="temperature":
                P=self._predict_proba_uncalibrated(X);self.multiclass_calibrator_=TemperatureScaler().fit(P,y,sample_weight=sample_weight)
            self._finalize_fit(X,y.astype(str));return self
        self._is_multiclass_=False;mapping={self.classes_[0]:0,self.classes_[1]:1};yy=np.asarray([mapping[v] for v in y],int);cw=self._weights(yy);sw=cw if sample_weight is None else cw*np.asarray(sample_weight,float);self._fit_binary(X,yy,sw);self._finalize_fit(X,yy);self._console_finish();return self

    def _predict_proba_uncalibrated(self,A):
        P=np.column_stack([e.predict_proba(A)[:,1] for e in self.estimators_]);P=np.clip(P,self.probability_clip,None);P/=P.sum(axis=1,keepdims=True);return P
    def predict_proba(self,X):
        check_is_fitted(self,"best_program_");A=self._prepare_X(X)
        if getattr(self,"_is_multiclass_",False):
            P=self._predict_proba_uncalibrated(A);return self.multiclass_calibrator_.predict_proba(P) if getattr(self,"multiclass_calibrator_",None) is not None else P
        p=self._prediction_from_program(self.best_program_,A);p=apply_calibrator(getattr(self,"calibration_model_",None),p);p=np.clip(p,self.probability_clip,1-self.probability_clip);return np.c_[1-p,p]
    def predict(self,X):
        P=self.predict_proba(X)
        if getattr(self,"_is_multiclass_",False):return self.classes_[np.argmax(P,axis=1)]
        return self.classes_[(P[:,1]>=self.decision_threshold_).astype(int)]
    def score(self,X,y,sample_weight=None):return accuracy_score(y,self.predict(X),sample_weight=sample_weight)
    def predict_set(self,X,confidence=.95):
        P=self.predict_proba(X);target=float(confidence);sets=[]
        for row in P:
            order=np.argsort(-row);cum=0.;keep=[]
            for j in order:
                keep.append(self.classes_[j]);cum+=row[j]
                if cum>=target:break
            sets.append(tuple(keep))
        return sets
    def get_expression(self):
        if getattr(self,"_is_multiclass_",False):return {str(c):e.get_expression() for c,e in zip(self.classes_,self.estimators_)}
        return super().get_expression()
    def to_latex(self):
        if getattr(self,"_is_multiclass_",False):return {str(c):program_to_latex(e.best_program_,e._symbolic_feature_names_) for c,e in zip(self.classes_,self.estimators_)}
        return program_to_latex(self.best_program_,self._symbolic_feature_names_)
    def save_model_json(self,path):return save_model_json(self,path)
    @classmethod
    def load_model_json(cls,path):return load_model_json(path,expected_class=cls)

FastSymbolicMultiClassifier=FastSymbolicClassifier
