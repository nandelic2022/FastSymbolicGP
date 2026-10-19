from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from .base import BaseFastSymbolicGP
from .metrics import regression_loss, fit_linear_scaler
from .backend import (NUMBA_AVAILABLE, SUPPORTED as NUMBA_SUPPORTED_PRIMITIVES, FUSED_REGRESSION_METRICS,
                      fused_regression_supported, score_population_regression_postfix,
                      estimate_fused_population_memory_mb, choose_population_batch_size)
from .program import Program
from .canonical import canonicalize_program
from .simplification import safe_simplify_program, certified_simplify_program
from .serialization import program_to_latex, save_model_json, load_model_json

class FastSymbolicRegressor(RegressorMixin, BaseFastSymbolicGP):
    def __init__(self, *, fitness_metric="rmse", metric=None, fitness_scaling="linear",
                 validation_metric=None, quantile=0.5, huber_delta=1.0,
                 uncertainty="split_conformal", confidence_levels=(0.80,0.90,0.95), calibration_fraction=0.15,
                 constant_optimization="champion", constant_optimization_rounds=2,
                 population_size=500, generations=50, tournament_size=20, function_set=("add","sub","mul","div"),
                 init_depth=(2,6), max_depth=12, max_nodes=255, p_crossover=.70, p_subtree_mutation=.15, p_point_mutation=.10,
                 const_range=(-1.0,1.0), const_probability=.15, constant_probability=None, max_samples=1.0, parsimony_coefficient=0.0,
                 validation_fraction=.20, patience=20, min_delta=1e-6, elitism=2, selection_tolerance=.02, exact_champion=True, exact_champion_candidates=4, selection_method="tournament",
                 epsilon_lexicase_epsilon="mad", random_immigrants=.05, restart_strategy="adaptive", restart_fraction=.25,
                 diversity_threshold=.08, simplification="safe", final_simplification="certified", simplification_tolerance=1e-10, simplification_reference_points=2048, constraints=None, feature_units=None, target_unit=None,
                 unit_policy="off", constant_units=None, optimization="scalar", pareto_archive_size=256, pareto_objectives=("validation_loss","nodes","constraint_penalty"), n_jobs=-1, time_budget=None,
                 evaluation_budget=None, memory_budget_mb=None, evaluation_backend="auto", fitness_engine="auto", evolution_model="panmictic", n_islands=1, migration_interval=10, migration_size=2, island_parallel=True, evaluation_cache=True, fitness_cache_max_entries=100000, typed_generation="auto", semantic_diversity=False, semantic_archive_size=64, semantic_sample_size=128, semantic_neighbors=5, semantic_weight=0.05, semantic_mutation_rate=0.0, semantic_mutation_trials=4, semantic_step_target=0.50, age_management="off", age_gap=10, age_layers=4, memetic_search="off", memetic_elites=2, memetic_interval=5, memetic_budget=120, memetic_method="hybrid", model_selection="best_validation", low_memory=False, missing_value_strategy="error", missing_value_constant=0.0, preset=None, preset_override_policy="defaults_only",
                 deterministic=True, checkpoint_path=None, checkpoint_interval=0, resume_from_checkpoint=False,
                 warm_start=False, random_state=None, verbose=1, display="dashboard", profile_fit=True):
        if constant_probability is not None: const_probability=float(constant_probability)
        super().__init__(population_size=population_size,generations=generations,tournament_size=tournament_size,function_set=function_set,
            init_depth=init_depth,max_depth=max_depth,max_nodes=max_nodes,p_crossover=p_crossover,p_subtree_mutation=p_subtree_mutation,
            p_point_mutation=p_point_mutation,const_range=const_range,const_probability=const_probability,max_samples=max_samples,
            parsimony_coefficient=parsimony_coefficient,validation_fraction=validation_fraction,patience=patience,min_delta=min_delta,
            elitism=elitism,selection_tolerance=selection_tolerance,exact_champion=exact_champion,exact_champion_candidates=exact_champion_candidates,selection_method=selection_method,epsilon_lexicase_epsilon=epsilon_lexicase_epsilon,random_immigrants=random_immigrants,
            restart_strategy=restart_strategy,restart_fraction=restart_fraction,diversity_threshold=diversity_threshold,simplification=simplification,final_simplification=final_simplification,simplification_tolerance=simplification_tolerance,simplification_reference_points=simplification_reference_points,
            constraints=constraints,feature_units=feature_units,target_unit=target_unit,unit_policy=unit_policy,constant_units=constant_units,optimization=optimization,
            pareto_archive_size=pareto_archive_size,pareto_objectives=pareto_objectives,n_jobs=n_jobs,time_budget=time_budget,evaluation_budget=evaluation_budget,
            memory_budget_mb=memory_budget_mb,evaluation_backend=evaluation_backend,fitness_engine=fitness_engine,evolution_model=evolution_model,n_islands=n_islands,migration_interval=migration_interval,migration_size=migration_size,island_parallel=island_parallel,evaluation_cache=evaluation_cache,fitness_cache_max_entries=fitness_cache_max_entries,typed_generation=typed_generation,semantic_diversity=semantic_diversity,semantic_archive_size=semantic_archive_size,semantic_sample_size=semantic_sample_size,semantic_neighbors=semantic_neighbors,semantic_weight=semantic_weight,semantic_mutation_rate=semantic_mutation_rate,semantic_mutation_trials=semantic_mutation_trials,semantic_step_target=semantic_step_target,age_management=age_management,age_gap=age_gap,age_layers=age_layers,memetic_search=memetic_search,memetic_elites=memetic_elites,memetic_interval=memetic_interval,memetic_budget=memetic_budget,memetic_method=memetic_method,model_selection=model_selection,low_memory=low_memory,missing_value_strategy=missing_value_strategy,missing_value_constant=missing_value_constant,preset=preset,preset_override_policy=preset_override_policy,deterministic=deterministic,
            checkpoint_path=checkpoint_path,checkpoint_interval=checkpoint_interval,resume_from_checkpoint=resume_from_checkpoint,
            warm_start=warm_start,random_state=random_state,verbose=verbose,display=display,profile_fit=profile_fit)
        self.constant_probability=constant_probability
        self.fitness_metric=fitness_metric; self.metric=metric; self.fitness_scaling=fitness_scaling; self.validation_metric=validation_metric
        self.quantile=quantile; self.huber_delta=huber_delta; self.uncertainty=uncertainty; self.confidence_levels=confidence_levels; self.calibration_fraction=calibration_fraction
        self.constant_optimization=constant_optimization; self.constant_optimization_rounds=constant_optimization_rounds

    def _resolve_fitness_engine(self, requested_engine):
        metric=str(self.metric or self.fitness_metric).lower()
        val_metric=metric if self.validation_metric is None else str(self.validation_metric).lower()
        scaling="none" if self.fitness_scaling is None else str(self.fitness_scaling).lower()
        metric_ok=metric in FUSED_REGRESSION_METRICS and val_metric in FUSED_REGRESSION_METRICS
        scaling_ok=scaling in {"linear","auto","none","identity"}
        primitives_ok=all(str(name) in NUMBA_SUPPORTED_PRIMITIVES for name in getattr(self,"function_set_",self.function_set))
        compatible=bool(NUMBA_AVAILABLE and metric_ok and scaling_ok and primitives_ok)
        if compatible and requested_engine in {"auto","fused"}:
            return "fused", None
        reasons=[]
        if not NUMBA_AVAILABLE: reasons.append("Numba is unavailable")
        if not metric_ok: reasons.append(f"metric {metric!r}/validation metric {val_metric!r} is not fused-compatible")
        if not scaling_ok: reasons.append(f"scaling {scaling!r} is not fused-compatible")
        if not primitives_ok: reasons.append("function_set contains custom/non-Numba primitives")
        reason="; ".join(reasons) or "materialized engine explicitly requested"
        if requested_engine == "fused":
            raise ValueError("Forced fused regression is incompatible: "+reason+". Use fitness_engine='auto' for safe fallback.")
        return "materialized", (reason if requested_engine=="auto" else None)

    def _evaluate_population(self,pop,Xtr,ytr,Xv,yv,swtr=None,swv=None):
        if self.resolved_config_.get("fitness_engine") != "fused":
            return super()._evaluate_population(pop,Xtr,ytr,Xv,yv,swtr,swv)
        missing=self._apply_fitness_cache(pop,Xtr,ytr,Xv,yv)
        if missing:
            metric=self.metric or self.fitness_metric
            val_metric=metric if self.validation_metric is None else self.validation_metric
            metric_name=str(metric).lower()
            self.execution_plan_["fused_training_passes"]=1 if metric_name in {"mse","mean_squared_error","rmse","root_mean_squared_error","r2","r2_score"} else 2
            self.execution_plan_["fused_single_pass_squared_error"]=bool(self.execution_plan_["fused_training_passes"]==1)
            scales,intercepts,train_losses,val_losses=score_population_regression_postfix(
                Xtr,ytr,Xv,yv,missing,metric=metric,validation_metric=val_metric,
                scaling=self.fitness_scaling,sample_weight_train=swtr,sample_weight_val=swv,
                n_jobs=self.n_jobs,huber_delta=self.huber_delta,memory_budget_mb=self.memory_budget_mb,
            )
            for i,p in enumerate(missing):
                p.scale=float(scales[i]); p.intercept=float(intercepts[i])
                p.loss=float(train_losses[i]); p.validation_loss=float(val_losses[i])
                if not self.constraints and str(self.unit_policy).lower() not in {"enforce","penalty"}:
                    p.constraint_penalty=0.0; p.metadata["constraint_issues"]=[]; p.metadata["feasible"]=True
                else:
                    p.constraint_penalty=float(self._constraint_penalty(p,Xtr))
                p.metadata["evaluation_backend"]="numba"; p.metadata["fitness_engine"]="fused"; p.metadata["n_jobs"]=self.n_jobs
                self._store_fitness_cache(p)
            self.evaluation_count_ += len(missing)
        estimate=estimate_fused_population_memory_mb(missing or pop,n_jobs=self.n_jobs)
        batch=choose_population_batch_size(missing or pop,self.memory_budget_mb,n_jobs=self.n_jobs)
        self.execution_plan_["estimated_fused_population_memory_mb"]=float(estimate)
        self.execution_plan_["population_batch_size"]=int(batch)
        self.execution_plan_["memory_budget_enforced"]=bool(self.memory_budget_mb is not None and batch < len(missing or pop))
        self.execution_plan_["duplicate_programs_skipped_last_batch"]=int(len(pop)-len(missing))
        return pop

    def validate_configuration(self):
        self._validate_common_configuration()
        scaling=str(self.fitness_scaling).lower() if self.fitness_scaling is not None else "none"
        if scaling not in {"linear","auto","none","identity"}: raise ValueError("Regression fitness_scaling must be linear, auto, or none")
        if str(self.selection_method).lower() not in {"tournament","semantic_tournament","lexicase","epsilon_lexicase","epsilon-lexicase"}: raise ValueError("Unsupported selection_method")
        return {"valid":True,"fitness_scaling":scaling,"fitness_metric":self.metric or self.fitness_metric}

    def _loss(self,y,pred,sample_weight=None):
        return regression_loss(y,pred,self.metric or self.fitness_metric,sample_weight,quantile=self.quantile,huber_delta=self.huber_delta)
    def _per_case_loss(self,y,pred): return np.abs(np.asarray(pred)-np.asarray(y))
    def _fit_program_scaler(self,p,X,y,sw=None):
        mode=str(self.fitness_scaling).lower() if self.fitness_scaling is not None else "none"; raw=p.raw(X)
        if mode in {"linear","auto"}: p.scale,p.intercept=fit_linear_scaler(raw,y,sw)
        else: p.scale,p.intercept=1.0,0.0
    def _prediction_from_program(self,p,X): return p.predict(X)

    def _optimize_champion_constants(self, program, X, y, sample_weight=None):
        mode=str(self.constant_optimization or "off").lower()
        paths=[path for path in program.root.paths() if program.root.get(path).kind=="const"]
        if mode in {"off","none","false"} or not paths:
            return program, {"enabled":mode not in {"off","none","false"},"constants":len(paths),"evaluations":0,"improved":False}
        q=program.clone(); self._fit_program_scaler(q,X,y,sample_weight)
        q.constraint_penalty=float(self._constraint_penalty(q,X))
        best=float(self._loss(y,q.predict(X),sample_weight))+q.constraint_penalty
        initial=best; evaluations=0; span=max(float(self.const_range[1])-float(self.const_range[0]),1e-12)
        for rnd in range(max(1,int(self.constant_optimization_rounds))):
            step=span*(0.20/(2**rnd))
            for path in paths:
                node=q.root.get(path); current=float(node.value); best_value=current; lo,hi=self._constant_bounds(node); local_step=max(step,0.10*(hi-lo)/(2**rnd))
                candidates=(current-local_step,current+local_step,0.0 if getattr(node,"unit",None) in {None,"","1",()} else current)
                for value in candidates:
                    value=float(np.clip(value,lo,hi)); cand=q.clone(); cand.root.get(path).value=value; cand.invalidate_caches()
                    self._fit_program_scaler(cand,X,y,sample_weight); cand.constraint_penalty=float(self._constraint_penalty(cand,X))
                    score=float(self._loss(y,cand.predict(X),sample_weight))+cand.constraint_penalty; evaluations+=1
                    if score < best-1e-15: best=score; best_value=value; q=cand
                q.root.get(path).value=best_value; q.invalidate_caches()
        return q, {"enabled":True,"method":"deterministic_coordinate","constants":len(paths),"evaluations":evaluations,"initial_objective":initial,"final_objective":best,"improved":bool(best<initial-1e-15)}

    def fit(self,X,y,sample_weight=None):
        self.validate_configuration(); X=self._prepare_X(X,fit=True); y=np.asarray(y,float).reshape(-1)
        if len(y)!=len(X): raise ValueError("X and y length mismatch")
        weights=None if sample_weight is None else np.asarray(sample_weight,float)
        if weights is not None and (weights.ndim!=1 or len(weights)!=len(X)):
            raise ValueError("sample_weight must be a one-dimensional array with length n_samples")
        self._fit_n_samples_=len(X); self.function_set_=__import__('fastsymbolicgp.primitives',fromlist=['validate_function_set']).validate_function_set(self.function_set); self._resolve_preset()
        self._console_start("SYMBOLIC REGRESSION")

        # Split conformal must reserve calibration observations before symbolic search.
        conformal=str(self.uncertainty).lower() in {"split_conformal","conformal","auto"} and 0<float(self.calibration_fraction)<.5 and len(X)>=30
        if conformal:
            model_idx,cal_idx=train_test_split(np.arange(len(X)),test_size=float(self.calibration_fraction),random_state=(self.random_state or 0)+7919)
            Xm,ym=X[model_idx],y[model_idx]; wm=None if weights is None else weights[model_idx]
        else:
            model_idx=np.arange(len(X)); cal_idx=None; Xm,ym=X,y; wm=weights

        vf=float(self.validation_fraction); local=np.arange(len(Xm))
        if vf>0:
            tr,va=train_test_split(local,test_size=vf,random_state=self.random_state)
            Xtr,ytr=Xm[tr],ym[tr]; Xv,yv=Xm[va],ym[va]; swtr=None if wm is None else wm[tr]; swv=None if wm is None else wm[va]
        else:
            Xtr,ytr,swtr=Xm,ym,wm; Xv=yv=swv=None
        self._evolve(Xtr,ytr,Xv,yv,swtr,swv)

        self.best_program_,self.constant_optimization_report_=self._optimize_champion_constants(self.best_program_,Xm,ym,wm)
        # One final post-memetic expression-quality pass.  Constant optimisation
        # can create exact/near-exact zeros and ones that were not available to
        # the earlier structural simplifier.
        final_mode=str(self._cfg("final_simplification") or "off").lower()
        pre_final=self.best_program_.clone()
        if final_mode in {"certified","aggressive_certified"}:
            q,final_report=certified_simplify_program(self.best_program_,Xm,atol=float(self.simplification_tolerance),rtol=float(self.simplification_tolerance),max_reference_points=int(self.simplification_reference_points),admissible=self._admissible)
        elif final_mode=="canonical":
            q=canonicalize_program(self.best_program_,"canonical"); final_report={"mode":"canonical","scope":"global_rules","passed":True,"reference_points":0,"original_nodes":pre_final.size,"final_nodes":q.size,"nodes_removed":pre_final.size-q.size,"reduction_fraction":float((pre_final.size-q.size)/max(1,pre_final.size)),"max_abs_prediction_delta":0.0,"accepted_rewrites":0}
        elif final_mode in {"safe","aggressive_safe"}:
            q=safe_simplify_program(self.best_program_); final_report={"mode":final_mode,"scope":"global_rules","passed":True,"reference_points":0,"original_nodes":pre_final.size,"final_nodes":q.size,"nodes_removed":pre_final.size-q.size,"reduction_fraction":float((pre_final.size-q.size)/max(1,pre_final.size)),"max_abs_prediction_delta":0.0,"accepted_rewrites":0}
        else:
            q=self.best_program_; final_report={"mode":"off","scope":"none","passed":True,"reference_points":0,"original_nodes":pre_final.size,"final_nodes":pre_final.size,"nodes_removed":0,"reduction_fraction":0.0,"max_abs_prediction_delta":0.0,"accepted_rewrites":0}
        if self._admissible(q): self.best_program_=q
        self._fit_program_scaler(self.best_program_,Xm,ym,wm)
        final_report["final_nodes"]=int(self.best_program_.size); final_report["nodes_removed"]=int(pre_final.size-self.best_program_.size); final_report["reduction_fraction"]=float((pre_final.size-self.best_program_.size)/max(1,pre_final.size)); self.simplification_report_=final_report
        self.best_expression_=self.get_expression()
        self.scale_=self.best_program_.scale; self.intercept_=self.best_program_.intercept
        if hasattr(self,"profile_"):
            self.profile_["best_nodes"]=int(self.best_program_.size); self.profile_["best_depth"]=int(self.best_program_.depth); self.profile_["simplification"]=dict(final_report)
        if conformal:
            resid=np.abs(y[cal_idx]-self.best_program_.predict(X[cal_idx])); self.conformal_scores_=np.sort(resid); self.calibration_size_=len(cal_idx); self.calibration_indices_=np.asarray(cal_idx,int)
        else:
            self.conformal_scores_=None; self.calibration_size_=0; self.calibration_indices_=np.empty(0,dtype=int)
        self._finalize_fit(X,y); self._console_finish(); return self

    def get_raw_expression(self):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self,"best_program_"); return self.best_program_.to_string(self._symbolic_feature_names_)

    def get_expression(self, include_scaling=True):
        """Return the fitted regression equation.

        V1.2.4 includes the affine scaler used by prediction.  Pass
        ``include_scaling=False`` (or use ``get_raw_expression``) for the raw GP
        tree only.
        """
        raw=self.get_raw_expression()
        if not include_scaling: return raw
        a=float(self.best_program_.scale); b=float(self.best_program_.intercept)
        if abs(a-1.0)<=1e-15 and abs(b)<=1e-15: return raw
        if abs(b)<=1e-15: return f"({a:.10g} * ({raw}))"
        sign="+" if b>=0 else "-"
        return f"({a:.10g} * ({raw}) {sign} {abs(b):.10g})"

    def predict(self,X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self,"best_program_"); return self.best_program_.predict(self._prepare_X(X))
    def score(self,X,y,sample_weight=None): return r2_score(y,self.predict(X),sample_weight=sample_weight)
    def predict_interval(self,X,confidence=.95):
        pred=self.predict(X)
        if self.conformal_scores_ is None or len(self.conformal_scores_)==0: raise ValueError("Conformal uncertainty was not calibrated")
        n=len(self.conformal_scores_); q=min(1.0,np.ceil((n+1)*float(confidence))/n); radius=float(np.quantile(self.conformal_scores_,q,method="higher"))
        return pred,pred-radius,pred+radius
    def coverage_report(self,X,y,confidence=.95):
        pred,lo,hi=self.predict_interval(X,confidence); y=np.asarray(y,float); return {"confidence":float(confidence),"empirical_coverage":float(np.mean((y>=lo)&(y<=hi))),"mean_width":float(np.mean(hi-lo)),"n":len(y)}
    def to_latex(self): return program_to_latex(self.best_program_,self._symbolic_feature_names_)
    def save_model_json(self,path): return save_model_json(self,path)
    @classmethod
    def load_model_json(cls,path): return load_model_json(path,expected_class=cls)
