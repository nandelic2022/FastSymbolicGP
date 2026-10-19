from __future__ import annotations
import hashlib, json, os, platform, time, warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from ._version import __version__
from .primitives import validate_function_set, get_primitive
from .program import (Node, Program, random_tree, subtree_crossover, subtree_mutation, point_mutation, subtree_crossover_flat, subtree_mutation_flat, point_mutation_flat, subtree_crossover_genome, subtree_mutation_genome, point_mutation_genome, simplify_program,
                      typed_random_tree, typed_subtree_crossover, typed_subtree_mutation, typed_point_mutation, infer_unit)
from .constraints import structural_constraint_penalty, behavioral_constraint_penalty, unit_constraint_penalty, build_constraint_points
from .pareto import update_archive, assign_nsga2, normalize_objectives, objective_values
from .backend import NUMBA_AVAILABLE
from .semantics import assign_semantic_novelty, update_semantic_archive, semantic_distance, semantic_archive_report
from .canonical import canonicalize_program
from .memetic import optimize_constants as memetic_optimize_constants, constant_paths as memetic_constant_paths
from .model_selection import select_final_model
from .simplification import safe_simplify_program, certified_simplify_program

PRESETS={
 "superfast":dict(population_size=40,generations=30,max_depth=6,selection_method="tournament"),
 "fast":dict(population_size=64,generations=60,max_depth=8,selection_method="tournament"),
 "balanced":dict(population_size=96,generations=120,max_depth=10,selection_method="epsilon_lexicase"),
 "accurate":dict(population_size=160,generations=200,max_depth=12,selection_method="epsilon_lexicase"),
 "interpretable":dict(population_size=80,generations=120,max_depth=7,parsimony_coefficient=0.01,selection_method="lexicase",final_simplification="certified"),
 "publication":dict(population_size=160,generations=200,max_depth=10,selection_method="epsilon_lexicase",optimization="pareto",deterministic=True,final_simplification="safe"),
 "large_dataset":dict(population_size=64,generations=100,max_depth=8,max_samples=0.35,selection_method="tournament"),
}



def _tournament_min_cdf(n, k):
    """CDF of the winner index for best-of-k sampling without replacement.

    Population must already be sorted best-first. The winner is the minimum
    sampled index. This is exactly the marginal distribution of the classic
    tournament used by FastSymbolicGP, but can be sampled in vectorized batches.
    """
    n=int(n); k=max(1,min(int(k),n))
    mmax=n-k
    probs=np.empty(mmax+1,dtype=np.float64)
    probs[0]=float(k)/float(n)
    for m in range(mmax):
        den=float(n-m-1)
        probs[m+1]=0.0 if den<=0 else probs[m]*float(n-m-k)/den
    cdf=np.cumsum(probs)
    cdf[-1]=1.0
    return cdf


class _BatchedTournamentScheduler:
    """Vectorized exact-distribution tournament winner scheduler."""
    __slots__=("rng","cdf","batch_size","buffer","cursor","draws")
    def __init__(self, rng, n, k, batch_size=4096):
        self.rng=rng; self.cdf=_tournament_min_cdf(n,k); self.batch_size=max(64,int(batch_size))
        self.buffer=np.empty(0,dtype=np.int32); self.cursor=0; self.draws=0
    def _refill(self):
        u=self.rng.random(self.batch_size)
        self.buffer=np.searchsorted(self.cdf,u,side="left").astype(np.int32,copy=False)
        self.cursor=0
    def next(self):
        if self.cursor>=len(self.buffer): self._refill()
        out=int(self.buffer[self.cursor]); self.cursor+=1; self.draws+=1
        return out

class BaseFastSymbolicGP(BaseEstimator):
    def __init__(self, *, population_size=500, generations=50, tournament_size=20,
                 function_set=("add","sub","mul","div"), init_depth=(2,6), max_depth=12,
                 max_nodes=255, p_crossover=.70, p_subtree_mutation=.15, p_point_mutation=.10,
                 const_range=(-1.0,1.0), const_probability=.15, max_samples=1.0,
                 parsimony_coefficient=0.0, validation_fraction=.20, patience=20, min_delta=1e-6,
                 elitism=2, selection_tolerance=.02, exact_champion=True, exact_champion_candidates=4,
                 selection_method="tournament", epsilon_lexicase_epsilon="mad",
                 random_immigrants=.05, restart_strategy="adaptive", restart_fraction=.25,
                 diversity_threshold=.08, simplification="safe", final_simplification="certified", simplification_tolerance=1e-10, simplification_reference_points=2048, constraints=None,
                 feature_units=None, target_unit=None, unit_policy="off", constant_units=None, optimization="scalar",
                 pareto_archive_size=256, pareto_objectives=("validation_loss","nodes","constraint_penalty"), n_jobs=-1, time_budget=None, evaluation_budget=None,
                 memory_budget_mb=None, evaluation_backend="auto", fitness_engine="auto",
                 evolution_model="panmictic", n_islands=1, migration_interval=10, migration_size=2,
                 island_parallel=True, evaluation_cache=True, fitness_cache_max_entries=100000, typed_generation="auto",
                 semantic_diversity=False, semantic_archive_size=64, semantic_sample_size=128, semantic_neighbors=5,
                 semantic_weight=0.05, semantic_mutation_rate=0.0, semantic_mutation_trials=4, semantic_step_target=0.50,
                 age_management="off", age_gap=10, age_layers=4,
                 memetic_search="off", memetic_elites=2, memetic_interval=5, memetic_budget=120, memetic_method="hybrid",
                 model_selection="best_validation",
                 low_memory=False, missing_value_strategy="error", missing_value_constant=0.0,
                 preset=None, preset_override_policy="defaults_only", deterministic=True,
                 checkpoint_path=None, checkpoint_interval=0, resume_from_checkpoint=False,
                 warm_start=False, random_state=None, verbose=1, display="dashboard", profile_fit=True):
        self.population_size=population_size; self.generations=generations; self.tournament_size=tournament_size
        self.function_set=function_set; self.init_depth=init_depth; self.max_depth=max_depth; self.max_nodes=max_nodes
        self.p_crossover=p_crossover; self.p_subtree_mutation=p_subtree_mutation; self.p_point_mutation=p_point_mutation
        self.const_range=const_range; self.const_probability=const_probability; self.max_samples=max_samples
        self.parsimony_coefficient=parsimony_coefficient; self.validation_fraction=validation_fraction; self.patience=patience; self.min_delta=min_delta
        self.elitism=elitism; self.selection_tolerance=selection_tolerance; self.exact_champion=exact_champion; self.exact_champion_candidates=exact_champion_candidates
        self.selection_method=selection_method; self.epsilon_lexicase_epsilon=epsilon_lexicase_epsilon
        self.random_immigrants=random_immigrants; self.restart_strategy=restart_strategy; self.restart_fraction=restart_fraction; self.diversity_threshold=diversity_threshold
        self.simplification=simplification; self.final_simplification=final_simplification; self.simplification_tolerance=simplification_tolerance; self.simplification_reference_points=simplification_reference_points; self.constraints=constraints; self.feature_units=feature_units; self.target_unit=target_unit; self.unit_policy=unit_policy; self.constant_units=constant_units
        self.optimization=optimization; self.pareto_archive_size=pareto_archive_size; self.pareto_objectives=pareto_objectives; self.n_jobs=n_jobs; self.time_budget=time_budget; self.evaluation_budget=evaluation_budget
        self.memory_budget_mb=memory_budget_mb; self.evaluation_backend=evaluation_backend; self.fitness_engine=fitness_engine
        self.evolution_model=evolution_model; self.n_islands=n_islands; self.migration_interval=migration_interval; self.migration_size=migration_size
        self.island_parallel=island_parallel; self.evaluation_cache=evaluation_cache; self.fitness_cache_max_entries=fitness_cache_max_entries; self.typed_generation=typed_generation
        self.semantic_diversity=semantic_diversity; self.semantic_archive_size=semantic_archive_size; self.semantic_sample_size=semantic_sample_size; self.semantic_neighbors=semantic_neighbors
        self.semantic_weight=semantic_weight; self.semantic_mutation_rate=semantic_mutation_rate; self.semantic_mutation_trials=semantic_mutation_trials; self.semantic_step_target=semantic_step_target
        self.age_management=age_management; self.age_gap=age_gap; self.age_layers=age_layers
        self.memetic_search=memetic_search; self.memetic_elites=memetic_elites; self.memetic_interval=memetic_interval; self.memetic_budget=memetic_budget; self.memetic_method=memetic_method
        self.model_selection=model_selection
        self.low_memory=low_memory; self.missing_value_strategy=missing_value_strategy; self.missing_value_constant=missing_value_constant
        self.preset=preset; self.preset_override_policy=preset_override_policy; self.deterministic=deterministic
        self.checkpoint_path=checkpoint_path; self.checkpoint_interval=checkpoint_interval; self.resume_from_checkpoint=resume_from_checkpoint; self.warm_start=warm_start
        self.random_state=random_state; self.verbose=verbose; self.display=display; self.profile_fit=profile_fit

    def _console_enabled(self):
        return bool(int(getattr(self,"verbose",0) or 0)>0 and str(getattr(self,"display","dashboard") or "dashboard").lower() not in {"off","none","silent","false","0"})

    @staticmethod
    def _console_time(seconds):
        seconds=max(0,int(seconds or 0)); h,rem=divmod(seconds,3600); m,sec=divmod(rem,60)
        return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    def _console_start(self, task="SYMBOLIC LEARNING"):
        if not self._console_enabled(): return
        cfg=getattr(self,"resolved_config_",{}) or {}; plan=getattr(self,"execution_plan_",{}) or {}
        pop=int(cfg.get("population_size",self.population_size)); gens=int(cfg.get("generations",self.generations)); budget=pop*gens
        backend=str(cfg.get("evaluation_backend",self.evaluation_backend)).upper(); engine=str(cfg.get("fitness_engine",self.fitness_engine)).upper()
        selection=str(cfg.get("selection_method",self.selection_method)); opt=str(self.optimization); cpus=int(plan.get("cpu_count",os.cpu_count() or 1))
        sem="ON" if bool(self.semantic_diversity) else "OFF"; mem="ON" if str(self.memetic_search).lower() not in {"off","none","false"} else "OFF"; age="ON" if str(self.age_management).lower() not in {"off","none"} else "OFF"
        typed="ON" if self._typed_generation_enabled() else "OFF"; units=str(self.unit_policy).upper()
        print()
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print(f"║  ⚡ FastSymbolicGP {__version__:<8} │ {str(task).upper():<47} ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  DATA       {int(getattr(self,'_fit_n_samples_',0)):>8,} samples  │ {int(getattr(self,'n_features_in_',0)):>4} features  │ workload {str(plan.get('workload','?')):<8} ║")
        print(f"║  SEARCH     pop {pop:<6} × gen {gens:<5} │ budget {budget:>9,} │ {selection:<16} ║")
        print(f"║  ENGINE     {backend:<8} + {engine:<11} │ CPU {cpus:<3} │ jobs {str(self.n_jobs):<5} │ islands {int(self.n_islands):<3} ║")
        print(f"║  MODE       optimization {opt:<10} │ semantic {sem:<3} │ memetic {mem:<3} │ age {age:<3}       ║")
        print(f"║  SCIENCE    typed {typed:<3} │ units {units:<7} │ simplification {str(self._cfg('final_simplification')):<16} ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print("║  🚀 Evolution started                                                        ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        self._console_wall_start_=time.perf_counter(); self._console_best_seen_=np.inf

    def _console_generation(self, gen, generations, gen_best, diversity, elapsed):
        if not self._console_enabled(): return
        verbose=int(self.verbose or 0); emit=(verbose>=2 or gen==1 or gen%max(1,generations//10)==0 or gen==generations)
        if not emit: return
        pct=100.0*gen/max(1,generations); width=20; filled=min(width,int(round(width*gen/max(1,generations)))); bar="█"*filled+"░"*(width-filled)
        eta=(elapsed/max(gen,1))*max(0,generations-gen); eps=int(self.evaluation_count_/max(elapsed,1e-12))
        val=float(gen_best.validation_loss); previous=float(getattr(self,"_console_best_seen_",np.inf)); is_best=val<previous-1e-15
        if val<previous: self._console_best_seen_=val
        badge="★ BEST" if is_best else "      "
        print(f"⚡ {badge} │ G {gen:>4}/{generations:<4} │{bar}│ {pct:5.1f}% │ val {val:9.4g} │ train {float(gen_best.loss):9.4g} │ N {gen_best.size:>3} D {gen_best.depth:>2} │ div {diversity:5.1%} │ {eps:>5,} prog/s │ ETA {self._console_time(eta)}")

    def _console_finish(self):
        if not self._console_enabled() or not hasattr(self,"profile_"): return
        p=self.profile_; stages=dict(p.get("stage_seconds",{})); bottleneck=max(stages,key=stages.get) if stages else "n/a"; bottleneck_s=float(stages.get(bottleneck,0.0))
        expr=""
        try: expr=str(self.get_expression())
        except Exception: pass
        if len(expr)>112: expr=expr[:109]+"..."
        print()
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║  🏁 FastSymbolicGP • TRAINING COMPLETE                                      ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  Runtime      {float(p.get('runtime_seconds',0.0)):>9.3f} s │ Evaluations {int(p.get('actual_program_evaluations',0)):>9,} │ Speed {float(p.get('programs_per_second',0.0)):>8.0f} prog/s ║")
        print(f"║  Champion     nodes {int(p.get('best_nodes',0)):>4} │ depth {int(p.get('best_depth',0)):>3} │ generations {int(p.get('generations',0)):>4} │ stop {str(p.get('stop_reason','?')):<12} ║")
        print(f"║  Engine       {str(self.resolved_config_.get('evaluation_backend','?')).upper():<8} + {str(self.resolved_config_.get('fitness_engine','?')).upper():<10} │ bottleneck {bottleneck:<24} {bottleneck_s:>7.3f}s ║")
        simpl=dict(p.get("simplification",{})); removed=int(simpl.get("nodes_removed",0) or 0)
        print(f"║  Quality      simplification removed {removed:>3} nodes │ cache hit {100.0*float(p.get('fitness_cache_hit_rate',0.0)):>5.1f}% │ duplicates reused {int(p.get('batch_duplicate_reuse',0)):>6,} ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        if expr:
            print("║  🧬 Final equation                                                           ║")
            for i in range(0,len(expr),74): print(f"║  {expr[i:i+74]:<74}  ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")

    def set_params(self, **params):
        result=super().set_params(**params)
        explicit=set(getattr(self,"_explicit_search_params_",()))
        explicit.update(params)
        self._explicit_search_params_=explicit
        return result

    def _cfg(self,name):
        return getattr(self,"_resolved_params_",{}).get(name,getattr(self,name))

    def _resolve_preset(self):
        self.pareto_objectives_=normalize_objectives(self.pareto_objectives)
        # V1.1.4 never mutates constructor parameters during fit.  Presets live
        # in a separate resolved configuration used only by the runtime.
        self._resolved_params_={k:getattr(self,k) for k in self.get_params(deep=False) if hasattr(self,k)}
        p=str(self.preset or "none").lower(); applied={}; skipped={}
        if p not in {"none","off","custom"}:
            if p=="auto": p="balanced" if self._fit_n_samples_<10000 else "large_dataset"
            if p not in PRESETS: raise ValueError(f"Unknown preset {self.preset!r}")
            defaults=self.__class__().get_params(deep=False)
            explicit=set(getattr(self,"_explicit_search_params_",()))
            for k,v in PRESETS[p].items():
                if not hasattr(self,k): continue
                current=getattr(self,k)
                should=str(self.preset_override_policy).lower()=="always" or (k not in explicit and current==defaults.get(k))
                if should: self._resolved_params_[k]=v; applied[k]=v
                else: skipped[k]=current
        self.preset_=p if p not in {"none","off"} else "custom"
        self.preset_parameters_=applied; self.preset_skipped_parameters_=skipped

        requested_backend=str(self.evaluation_backend or "auto").lower()
        allowed={"auto","numpy","vectorized_numpy","numba","population_numba","postfix_numba"}
        if requested_backend not in allowed: raise ValueError("evaluation_backend must be auto, numpy, vectorized_numpy, or numba")
        resolved_backend=("numba" if NUMBA_AVAILABLE else "numpy") if requested_backend=="auto" else requested_backend
        if resolved_backend in {"population_numba","postfix_numba"}: resolved_backend="numba"
        backend_fallback=None
        if resolved_backend=="numba" and not NUMBA_AVAILABLE: resolved_backend="numpy"; backend_fallback="Numba is unavailable"
        if resolved_backend=="vectorized_numpy": resolved_backend="numpy"
        if str(self.missing_value_strategy or "error").lower()=="native" and resolved_backend=="numba":
            resolved_backend="numpy"; backend_fallback="native missing-value primitives require NumPy evaluation"

        requested_engine=str(self.fitness_engine or "auto").lower()
        if requested_engine not in {"auto","materialized","fused"}: raise ValueError("fitness_engine must be auto, materialized, or fused")
        resolved_engine,engine_fallback=self._resolve_fitness_engine(requested_engine)
        pop=int(self._cfg("population_size")); gens=int(self._cfg("generations")); depth=int(self._cfg("max_depth")); sel=str(self._cfg("selection_method"))
        self.execution_plan_={
            "workload":"small" if self._fit_n_samples_<5000 else "medium" if self._fit_n_samples_<100000 else "large",
            "n_samples":int(self._fit_n_samples_),"cpu_count":int(os.cpu_count() or 1),
            "requested_backend":self.evaluation_backend,"resolved_backend":resolved_backend,"backend_fallback_reason":backend_fallback,
            "requested_fitness_engine":self.fitness_engine,"resolved_fitness_engine":resolved_engine,"fitness_engine_fallback_reason":engine_fallback,
            "memory_budget_mb":self.memory_budget_mb,"numba_available":bool(NUMBA_AVAILABLE),"checkpoint_schema":4,
            "resolved_n_islands":int(max(1,self.n_islands)),"resolved_evolution_model":str(self.evolution_model),
            "cross_island_population_batching":bool(max(1,int(self.n_islands))>1),
        }
        self.resolved_config_={
            "version":__version__,"preset":self.preset_,"population_size":pop,"generations":gens,"max_depth":depth,
            "max_nodes":int(self.max_nodes),"selection_method":sel,"constraints":self.constraints or {},"unit_policy":self.unit_policy,"constant_units":self.constant_units or {},"pareto_objectives":tuple(self.pareto_objectives),
            "n_jobs":self.n_jobs,"n_islands":int(max(1,self.n_islands)),"evolution_model":str(self.evolution_model),
            "island_parallel":bool(self.island_parallel),"evaluation_cache":bool(self.evaluation_cache),
            "fitness_cache_max_entries":int(self.fitness_cache_max_entries),"typed_generation":str(self.typed_generation),
            "semantic_diversity":bool(self.semantic_diversity),"semantic_archive_size":int(self.semantic_archive_size),"semantic_sample_size":int(self.semantic_sample_size),"semantic_neighbors":int(self.semantic_neighbors),"semantic_weight":float(self.semantic_weight),"semantic_mutation_rate":float(self.semantic_mutation_rate),
            "age_management":str(self.age_management),"age_gap":int(self.age_gap),"age_layers":int(self.age_layers),
            "memetic_search":str(self.memetic_search),"memetic_elites":int(self.memetic_elites),"memetic_interval":int(self.memetic_interval),"memetic_budget":int(self.memetic_budget),"memetic_method":str(self.memetic_method),"model_selection":str(self.model_selection),"simplification":str(self.simplification),"final_simplification":str(self._cfg("final_simplification")),"simplification_tolerance":float(self.simplification_tolerance),"simplification_reference_points":int(self.simplification_reference_points),
            "evaluation_backend":resolved_backend,"fitness_engine":resolved_engine,"requested_evaluation_backend":self.evaluation_backend,
            "requested_fitness_engine":self.fitness_engine,"exact_champion_candidates":int(self.exact_champion_candidates),
            "constructor_params_mutated":False,
        }

    def _resolve_fitness_engine(self, requested_engine):
        if requested_engine=="fused": return "materialized","This estimator has no fused objective kernel; using materialized scoring"
        return "materialized",None

    def _validate_common_configuration(self):
        if int(self.population_size)<4: raise ValueError("population_size must be >= 4")
        if int(self.generations)<1: raise ValueError("generations must be >= 1")
        if int(self.tournament_size)<2: raise ValueError("tournament_size must be >= 2")
        if int(self.max_depth)<1 or int(self.max_nodes)<1: raise ValueError("max_depth and max_nodes must be positive")
        if not (0<=float(self.validation_fraction)<1): raise ValueError("validation_fraction must be in [0, 1)")
        if not (0<=float(self.random_immigrants)<1): raise ValueError("random_immigrants must be in [0, 1)")
        if str(self.selection_method).lower() not in {"tournament","semantic_tournament","lexicase","epsilon_lexicase","epsilon-lexicase"}: raise ValueError("Unsupported selection_method")
        if str(self.unit_policy).lower() not in {"off","none","penalty","enforce"}: raise ValueError("unit_policy must be off, penalty, or enforce")
        if str(self.preset_override_policy).lower() not in {"defaults_only","always"}: raise ValueError("preset_override_policy must be defaults_only or always")
        if int(self.n_islands)<1: raise ValueError("n_islands must be >= 1")
        if int(self.fitness_cache_max_entries)<0: raise ValueError("fitness_cache_max_entries must be >= 0")
        if str(self.typed_generation).lower() not in {"auto","on","off","true","false","enabled","disabled"}: raise ValueError("typed_generation must be auto/on/off")
        if int(self.migration_interval)<1: raise ValueError("migration_interval must be >= 1")
        if int(self.migration_size)<0: raise ValueError("migration_size must be >= 0")
        if int(self.semantic_archive_size)<0 or int(self.semantic_sample_size)<2 or int(self.semantic_neighbors)<1: raise ValueError("invalid semantic archive/sample/neighbor settings")
        if not (0.0<=float(self.semantic_mutation_rate)<=1.0): raise ValueError("semantic_mutation_rate must be in [0, 1]")
        if int(self.semantic_mutation_trials)<1: raise ValueError("semantic_mutation_trials must be >= 1")
        if str(self.age_management).lower() not in {"off","none","layered","alps","on"}: raise ValueError("age_management must be off or layered")
        if int(self.age_gap)<1 or int(self.age_layers)<1: raise ValueError("age_gap and age_layers must be positive")
        if str(self.memetic_search).lower() not in {"off","none","false","champion","elites","on","hybrid"}: raise ValueError("memetic_search must be off, champion, or elites")
        if int(self.memetic_elites)<1 or int(self.memetic_interval)<1 or int(self.memetic_budget)<1: raise ValueError("invalid memetic search settings")
        if str(self.model_selection).lower() not in {"best","best_validation","validation","smallest","smallest_within_tolerance","interpretable","pareto_knee","knee","semantic_consensus","consensus"}: raise ValueError("unsupported model_selection")
        if str(self.final_simplification).lower() not in {"off","none","safe","canonical","aggressive_safe","certified","aggressive_certified"}: raise ValueError("unsupported final_simplification")
        if float(self.simplification_tolerance)<0 or int(self.simplification_reference_points)<16: raise ValueError("invalid simplification tolerance/reference point settings")
        self.pareto_objectives_=normalize_objectives(self.pareto_objectives)
        if self.constant_units is not None:
            items=self.constant_units.keys() if isinstance(self.constant_units,dict) else self.constant_units
            from .units import parse_unit
            for unit in items: parse_unit(unit)
        return True

    def get_resolved_config(self): return dict(getattr(self,"resolved_config_",{}))

    def _prepare_X(self,X,fit=False):
        names=getattr(X,"columns",None); strategy=str(self.missing_value_strategy or "error").lower(); allow_nan=strategy in {"median","constant","native"}
        A=validate_data(self,X,dtype=float,ensure_2d=True,ensure_all_finite=("allow-nan" if allow_nan else True),reset=fit)
        if fit:
            self.n_features_in_=A.shape[1]; self._symbolic_feature_names_=np.asarray(names if names is not None else [f"X{i}" for i in range(A.shape[1])],dtype=object)
            if strategy=="median":
                fill=np.nanmedian(A,axis=0); self.missing_fill_values_=np.where(np.isfinite(fill),fill,0.0)
            elif strategy=="constant": self.missing_fill_values_=np.full(A.shape[1],float(self.missing_value_constant))
            elif strategy in {"native","error"}: self.missing_fill_values_=None
            else: raise ValueError("missing_value_strategy must be error, median, constant, or native")
        elif A.shape[1]!=self.n_features_in_: raise ValueError(f"Expected {self.n_features_in_} features, got {A.shape[1]}")
        if strategy in {"median","constant"}: A=np.where(np.isnan(A),np.asarray(self.missing_fill_values_,float),A)
        elif strategy=="native": A=np.asarray(A,float)
        return np.ascontiguousarray(A)

    def _feature_units_list(self):
        if self.feature_units is None: return None
        if isinstance(self.feature_units,dict):
            return [self.feature_units.get(str(n),self.feature_units.get(i,"1")) for i,n in enumerate(self._symbolic_feature_names_)]
        return list(self.feature_units)

    def _hard_constraint_violations(self,p):
        issues=[]
        if p.depth>int(self._cfg("max_depth")): issues.append("max_depth")
        if p.size>int(self.max_nodes): issues.append("max_nodes")
        if str(self.unit_policy).lower()=="enforce" and self.feature_units is not None and self.target_unit is not None:
            pen,msg=unit_constraint_penalty(p,self._feature_units_list(),self.target_unit)
            if pen>0: issues.append(msg or "unit constraint")
        c=self.constraints or {}
        if str(c.get("mode",c.get("constraint_mode","soft"))).lower()=="hard":
            pen,struct=structural_constraint_penalty(p,c,self._symbolic_feature_names_)
            if pen>0: issues.extend(struct)
        return issues

    def _admissible(self,p): return not self._hard_constraint_violations(p)

    def _typed_generation_enabled(self):
        mode=str(self.typed_generation).lower()
        if mode in {"off","false","disabled"}: return False
        if mode in {"on","true","enabled"}: return True
        return bool(str(self.unit_policy).lower()=="enforce" and self.feature_units is not None and self.target_unit is not None)

    def _make_program(self,rng):
        max_depth=int(self._cfg("max_depth")); lo=max(1,int(self.init_depth[0])); hi=min(int(self.init_depth[1]),max_depth)
        typed=self._typed_generation_enabled() and self.feature_units is not None and self.target_unit is not None
        for _ in range(120):
            desired=int(rng.integers(lo,hi+1)) if hi>=lo else max_depth
            try:
                if typed:
                    root=typed_random_tree(rng,self.n_features_in_,self.function_set_,desired,self._feature_units_list(),self.target_unit,self.const_range,self.const_probability,constant_units=self.constant_units)
                    p=Program(root); p.metadata["typed_generated"]=True
                else:
                    p=Program(random_tree(rng,self.n_features_in_,self.function_set_,max(0,desired-1),self.const_range,self.const_probability))
                p.metadata["age"]=0
                if str(self.simplification).lower() in {"canonical","aggressive","algebraic"}: p=canonicalize_program(p,self.simplification)
                if self._admissible(p): return p
            except ValueError:
                continue
        for i in range(self.n_features_in_):
            p=Program(Node("feature",i)); p.metadata["age"]=0
            if self._admissible(p): return p
        raise ValueError("Unable to generate an individual satisfying hard depth/node/unit/structural constraints")

    def _case_losses(self,p,X,y): return self._per_case_loss(y,self._prediction_from_program(p,X))

    def _prepare_lexicase_cache(self,pop,X,y):
        method=str(self._cfg("selection_method")).lower()
        if method in {"tournament","semantic_tournament"}: self._case_loss_cache_={}; return
        self._case_loss_cache_={p.structural_hash():np.asarray(self._case_losses(p,X,y),float) for p in pop}

    def _select(self,pop,rng,X,y):
        method=str(self._cfg("selection_method")).lower()
        # The common tournament path is intentionally zero-copy.  V1.2.1 used
        # ``list(pop)`` for every parent selection, which copied hundreds of
        # references tens of thousands of times per fit.
        candidates=pop if not self._age_management_enabled() else list(pop)
        if self._age_management_enabled() and candidates:
            layers={self._age_layer(int(p.metadata.get("age",0))) for p in candidates}
            target=int(rng.choice(sorted(layers)))
            local=[p for p in candidates if self._age_layer(int(p.metadata.get("age",0)))==target]
            if len(local)>=2: candidates=local
        if method in {"tournament","semantic_tournament"}:
            k=max(2,min(int(self.tournament_size),len(candidates))); ids=rng.choice(len(candidates),size=k,replace=False)
            best=candidates[int(ids[0])]
            if method=="semantic_tournament":
                best_key=(self._selection_key(best),-float(best.metadata.get("semantic_novelty",0.0)))
                for idx in ids[1:]:
                    cand=candidates[int(idx)]; key=(self._selection_key(cand),-float(cand.metadata.get("semantic_novelty",0.0)))
                    if key<best_key: best,best_key=cand,key
            else:
                best_key=self._selection_key(best)
                for idx in ids[1:]:
                    cand=candidates[int(idx)]; key=self._selection_key(cand)
                    if key<best_key: best,best_key=cand,key
            return best
        candidates=list(candidates)
        order=rng.permutation(X.shape[0]); cache=getattr(self,"_case_loss_cache_",{})
        for case in order:
            case_vals=[]
            for p in candidates:
                key=p.structural_hash(); arr=cache.get(key)
                if arr is None:
                    arr=np.asarray(self._case_losses(p,X,y),float); cache[key]=arr
                case_vals.append(arr[case])
            vals=np.asarray(case_vals); best=float(np.nanmin(vals)); eps=0.0
            if method in {"epsilon_lexicase","epsilon-lexicase"}:
                if self.epsilon_lexicase_epsilon=="mad": eps=float(np.nanmedian(np.abs(vals-np.nanmedian(vals))))
                else: eps=float(self.epsilon_lexicase_epsilon)
            candidates=[p for p,v in zip(candidates,vals) if v<=best+eps+1e-15]
            if len(candidates)<=1: break
        return candidates[int(rng.integers(0,len(candidates)))]

    def _selection_key(self,p):
        novelty=float(p.metadata.get("semantic_novelty",0.0)) if bool(self.semantic_diversity) else 0.0
        if str(self.optimization).lower() in {"pareto","nsga2","multiobjective"} and "pareto_rank" in p.metadata:
            return (int(p.metadata["pareto_rank"]),-float(p.metadata.get("crowding_distance",0.0)),-float(self.semantic_weight)*novelty,p.validation_loss,p.size)
        cached=p.metadata.get("_scalar_selection_key")
        if cached is not None and not bool(self.semantic_diversity):
            return cached
        key=(p.validation_loss+p.constraint_penalty+float(self._cfg("parsimony_coefficient"))*p.size-float(self.semantic_weight)*novelty,p.size)
        if not bool(self.semantic_diversity): p.metadata["_scalar_selection_key"]=key
        return key

    def _diversity(self,pop): return len({p.structural_hash() for p in pop})/max(len(pop),1)

    def _constraint_penalty(self,p,X):
        s,issues=structural_constraint_penalty(p,self.constraints,self._symbolic_feature_names_)
        u,ui=unit_constraint_penalty(p,self._feature_units_list(),self.target_unit) if str(self.unit_policy).lower() in {"enforce","penalty"} else (0,None)
        b,bi,breport=behavioral_constraint_penalty(p,X,self._symbolic_feature_names_,self.constraints,return_report=True)
        all_issues=issues+([ui] if ui else [])+bi
        p.metadata["constraint_issues"]=all_issues
        p.metadata["behavioral_constraint_report"]=breport
        hard=str((self.constraints or {}).get("mode",(self.constraints or {}).get("constraint_mode","soft"))).lower()=="hard"
        unit_hard=str(self.unit_policy).lower()=="enforce"
        infeasible=bool((hard and (issues or bi)) or (unit_hard and ui))
        p.metadata["feasible"]=not infeasible
        total=float(s+u+b)
        if infeasible:
            # Hard scientific constraints are feasibility conditions, not just large
            # scalar penalties. Infinite fitness prevents a violating candidate from
            # becoming a champion even if constraint_penalty is omitted from Pareto objectives.
            p.loss=float("inf"); p.validation_loss=float("inf")
            total=max(total,1e12)
        return total

    @staticmethod
    def _eval_token(X,y):
        h=hashlib.blake2b(digest_size=12); h.update(str(X.shape).encode()); h.update(str(y.shape).encode())
        if X.size:
            flat=X.ravel(); ids=np.linspace(0,len(flat)-1,min(64,len(flat)),dtype=int); h.update(np.ascontiguousarray(flat[ids]).tobytes())
        if y.size:
            ids=np.linspace(0,len(y)-1,min(64,len(y)),dtype=int); h.update(np.ascontiguousarray(y[ids]).tobytes())
        return h.hexdigest()

    def _fitness_cache_context(self,Xtr,ytr,Xv,yv):
        token=self._eval_token(Xtr,ytr)+(":"+self._eval_token(Xv,yv) if Xv is not None else "")
        if getattr(self,"_fitness_cache_token_",None)!=token:
            self._fitness_cache_token_=token; self._fitness_cache_={}
        if not hasattr(self,"_fitness_cache_hits_"): self._fitness_cache_hits_=0
        if not hasattr(self,"_fitness_cache_misses_"): self._fitness_cache_misses_=0
        return token

    def _apply_fitness_cache(self,pop,Xtr,ytr,Xv,yv):
        self._fitness_cache_context(Xtr,ytr,Xv,yv)
        self._pending_fitness_aliases_={}
        if not bool(self.evaluation_cache) or int(self.fitness_cache_max_entries)==0:
            self._fitness_cache_misses_ += len(pop); return list(pop)
        missing=[]; pending={}
        for p in pop:
            key=p.structural_hash(); cached=self._fitness_cache_.get(key)
            if cached is not None:
                p.scale,p.intercept,p.loss,p.validation_loss,p.constraint_penalty=cached
                p.metadata.pop("_scalar_selection_key",None)
                p.metadata["fitness_cache_hit"]=True; self._fitness_cache_hits_+=1; continue
            if key in pending:
                pending[key].append(p); p.metadata["fitness_batch_duplicate"]=True
                self._fitness_cache_hits_+=1
                self._duplicate_fitness_reuse_=int(getattr(self,"_duplicate_fitness_reuse_",0))+1
                continue
            pending[key]=[p]; missing.append(p); self._fitness_cache_misses_+=1
        self._pending_fitness_aliases_=pending
        return missing

    def _store_fitness_cache(self,p):
        p.metadata.pop("_scalar_selection_key",None)
        key=p.structural_hash(); value=(p.scale,p.intercept,p.loss,p.validation_loss,p.constraint_penalty)
        if bool(self.evaluation_cache) and int(self.fitness_cache_max_entries)!=0:
            cache=self._fitness_cache_; cache[key]=value
            limit=max(0,int(self.fitness_cache_max_entries))
            while limit and len(cache)>limit: cache.pop(next(iter(cache)))
        # Reuse the just-computed result for structural duplicates in the same batch.
        aliases=getattr(self,"_pending_fitness_aliases_",{}).get(key,())
        for alias in aliases:
            if alias is p: continue
            alias.scale,alias.intercept,alias.loss,alias.validation_loss,alias.constraint_penalty=value
            alias.metadata["fitness_cache_hit"]=True; alias.metadata["fitness_batch_duplicate"]=True


    def _evaluate_population(self,pop,Xtr,ytr,Xv,yv,swtr=None,swv=None):
        missing=self._apply_fitness_cache(pop,Xtr,ytr,Xv,yv)
        for p in missing:
            p.metadata["evaluation_backend"]=self.resolved_config_.get("evaluation_backend","numpy"); p.metadata["n_jobs"]=self.n_jobs
            self._fit_program_scaler(p,Xtr,ytr,swtr)
            p.loss=float(self._loss(ytr,self._prediction_from_program(p,Xtr),swtr))
            p.validation_loss=float(self._validation_loss(yv,self._prediction_from_program(p,Xv),swv)) if Xv is not None else p.loss
            p.constraint_penalty=float(self._constraint_penalty(p,Xtr)); p.metadata.pop("_scalar_selection_key",None); self.evaluation_count_+=1
            self._store_fitness_cache(p)
        return pop

    def _validation_loss(self,y,pred,sample_weight=None): return self._loss(y,pred,sample_weight)

    def _dataset_fingerprint(self,X,y):
        h=hashlib.sha256(); h.update(np.ascontiguousarray(X).tobytes()); h.update(np.ascontiguousarray(y).tobytes()); return h.hexdigest()

    def _flat_genome_hotpath_enabled(self):
        """Whether the standard search can run rootless during evolution."""
        try:
            from .backend import NUMBA_AVAILABLE, SUPPORTED
            functions_ok=all(str(f) in SUPPORTED for f in self.function_set_)
        except Exception:
            return False
        hard=bool(self.constraints and str((self.constraints or {}).get("mode",(self.constraints or {}).get("constraint_mode","soft"))).lower()=="hard")
        return bool(
            NUMBA_AVAILABLE and functions_ok
            and str(self._cfg("selection_method")).lower()=="tournament"
            and not self._age_management_enabled()
            and not self._typed_generation_enabled()
            and str(self.unit_policy).lower() in {"off","none","false"}
            and not self.constraints
            and not hard
            and not self._semantic_enabled()
            and str(self.memetic_search or "off").lower() in {"off","none","false"}
            and str(self.simplification or "off").lower() not in {"canonical","aggressive","algebraic"}
        )

    def _initial_populations(self,rng):
        n_islands=max(1,int(self.n_islands)); total=int(self._cfg("population_size")); flat=self._flat_genome_hotpath_enabled()
        def make_one():
            p=self._make_program(rng)
            return p.as_flat_only() if flat else p
        if str(self.evolution_model).lower() not in {"island","islands"} and n_islands<=1:
            return [[make_one() for _ in range(total)]]
        n_islands=min(n_islands,total); sizes=[total//n_islands]*n_islands
        for i in range(total%n_islands): sizes[i]+=1
        return [[make_one() for _ in range(sz)] for sz in sizes]

    def _migrate(self,islands):
        if len(islands)<=1 or int(self.migration_size)<=0: return islands
        migrants=[]
        for pop in islands:
            pop.sort(key=self._selection_key); migrants.append([p.clone() for p in pop[:min(int(self.migration_size),len(pop))]])
        for i,pop in enumerate(islands):
            incoming=migrants[(i-1)%len(islands)]
            pop.sort(key=self._selection_key); keep=max(0,len(pop)-len(incoming)); islands[i]=pop[:keep]+[p.clone() for p in incoming]
        return islands

    def _semantic_enabled(self):
        return bool(self.semantic_diversity) or str(self._cfg("selection_method")).lower()=="semantic_tournament" or float(self.semantic_mutation_rate)>0 or any(str(x).lower() in {"semantic_redundancy","semantic_similarity"} for x in getattr(self,"pareto_objectives_",()))

    def _age_management_enabled(self):
        return str(self.age_management).lower() in {"layered","alps","on"}

    def _age_layer(self,age):
        if not self._age_management_enabled(): return 0
        age=max(0,int(age)); gap=max(1,int(self.age_gap)); layers=max(1,int(self.age_layers))
        threshold=gap
        for layer in range(layers-1):
            if age<=threshold: return layer
            threshold*=2
        return layers-1

    def _assign_semantics(self,pop,X):
        if not self._semantic_enabled() or len(pop)<1: return
        assign_semantic_novelty(pop,X,getattr(self,"semantic_archive_",[]),self.semantic_sample_size,self.semantic_neighbors)

    def _semantic_guided_mutation(self,parent,rng,X,typed,feature_units):
        trials=[]; n=max(1,int(self.semantic_mutation_trials))
        for _ in range(n):
            if rng.random()<0.65:
                child=(typed_subtree_mutation(parent,rng,self.n_features_in_,self.function_set_,int(self._cfg("max_depth")),feature_units,self.const_range,self.const_probability,self.constant_units)
                       if typed else subtree_mutation(parent,rng,self.n_features_in_,self.function_set_,int(self._cfg("max_depth")),self.const_range,self.const_probability))
            else:
                child=typed_point_mutation(parent,rng,self.function_set_,self.n_features_in_,feature_units,self.const_range,self.constant_units) if typed else point_mutation(parent,rng,self.function_set_,self.n_features_in_,self.const_range)
            child.metadata["age"]=int(parent.metadata.get("age",0))+1
            if str(self.simplification).lower() in {"canonical","aggressive","algebraic"}: child=canonicalize_program(child,self.simplification)
            if self._admissible(child):
                try: trials.append((abs(semantic_distance(parent,child,X,self.semantic_sample_size)-float(self.semantic_step_target)),child))
                except Exception: pass
        if trials:
            trials.sort(key=lambda x:x[0]); trials[0][1].metadata["semantic_mutation"]=True; trials[0][1].metadata["semantic_step_error"]=float(trials[0][0]); return trials[0][1]
        return None

    def _memetic_refine(self,program,X,y,sw,Xv=None,yv=None,swv=None):
        paths=memetic_constant_paths(program)
        if not paths: return program.clone(),{"enabled":True,"constants":0,"evaluations":0,"improved":False}
        bounds=[self._constant_bounds(program.root.get(path)) for path in paths]
        def objective(cand):
            self._fit_program_scaler(cand,X,y,sw); cand.constraint_penalty=float(self._constraint_penalty(cand,X))
            return float(self._loss(y,self._prediction_from_program(cand,X),sw))+float(cand.constraint_penalty)
        q,report=memetic_optimize_constants(program,objective,bounds,method=self.memetic_method,rounds=3,budget=self.memetic_budget,random_state=self.random_state or 0)
        self._fit_program_scaler(q,X,y,sw); q.loss=float(self._loss(y,self._prediction_from_program(q,X),sw)); q.validation_loss=float(self._validation_loss(yv,self._prediction_from_program(q,Xv),swv)) if Xv is not None else q.loss; q.constraint_penalty=float(self._constraint_penalty(q,X))
        q.metadata["memetic_refined"]=bool(report.get("improved",False)); q.metadata["memetic_report"]=dict(report)
        return q,report

    def _apply_memetic_search(self,islands,X,y,sw,Xv,yv,swv,generation):
        mode=str(self.memetic_search or "off").lower()
        if mode in {"off","none","false"} or generation%max(1,int(self.memetic_interval))!=0: return islands,[]
        reports=[]; n=1 if mode=="champion" else max(1,int(self.memetic_elites))
        for ii,pop in enumerate(islands):
            order=sorted(range(len(pop)),key=lambda i:self._selection_key(pop[i]))[:min(n,len(pop))]
            for i in order:
                q,rep=self._memetic_refine(pop[i],X,y,sw,Xv,yv,swv); reports.append(rep)
                if float(q.loss+q.constraint_penalty)<float(pop[i].loss+pop[i].constraint_penalty)-1e-15:
                    q.metadata["age"]=int(pop[i].metadata.get("age",0)); pop[i]=q
        return islands,reports

    def _make_offspring_population(self,pop,rng,Xe,ye):
        pop=sorted(pop,key=self._selection_key); target=len(pop)
        elite=[p.evolution_clone() for p in pop[:max(1,min(int(self.elitism),target))]]
        for p in elite: p.metadata["age"]=int(p.metadata.get("age",0))+1
        new=elite
        div=getattr(self,"_current_diversity_for_variation_",None)
        if div is None: div=self._diversity(pop)
        stagnant=int(getattr(self,"_stagnant_",0))
        restart=(str(self.restart_strategy).lower()=="adaptive" and div<float(self.diversity_threshold) and stagnant>max(3,int(self.patience)//3))
        target_imm=float(self.restart_fraction if restart else self.random_immigrants); immigrants=max(0,min(target-len(new),int(target*target_imm)))
        typed=self._typed_generation_enabled() and self.feature_units is not None and self.target_unit is not None
        feature_units=self._feature_units_list() if typed else None
        method=str(self._cfg("selection_method")).lower()
        fast_tournament=(method=="tournament" and not self._age_management_enabled())
        tournament_k=max(2,min(int(self.tournament_size),target))
        max_depth_fast=int(self._cfg("max_depth")); max_nodes_fast=int(self.max_nodes)
        basic_admissible=(not typed and str(self.unit_policy).lower()!="enforce" and not (self.constraints and str((self.constraints or {}).get("mode",(self.constraints or {}).get("constraint_mode","soft"))).lower()=="hard"))
        # V1.2.8 batched tournament scheduler. Population is sorted best-first,
        # therefore the classic best-of-k-without-replacement winner is the
        # minimum sampled index. We sample that exact winner distribution in
        # vectorized batches using a dedicated RNG so tournament scheduling no
        # longer dominates Python orchestration.
        scheduler=None
        if fast_tournament:
            sel_rng=getattr(self,"_selection_rng_",None)
            if sel_rng is None:
                seed=(0 if self.random_state is None else int(self.random_state)) ^ 0x5F3759DF
                sel_rng=np.random.default_rng(seed); self._selection_rng_=sel_rng
            scheduler=_BatchedTournamentScheduler(sel_rng,target,tournament_k,batch_size=max(1024,target*4))
        def select_parent():
            if not fast_tournament:
                return self._select(pop,rng,Xe,ye)
            return pop[scheduler.next()]
        attempts=0
        while len(new)<target-immigrants and attempts<max(200,target*100):
            attempts+=1; self._variation_attempts_=int(getattr(self,"_variation_attempts_",0))+1; r=rng.random(); a=select_parent(); child=None
            if float(self.semantic_mutation_rate)>0 and rng.random()<float(self.semantic_mutation_rate):
                child=self._semantic_guided_mutation(a,rng,Xe,typed,feature_units)
            if child is None and r<float(self.p_crossover):
                b=select_parent()
                child=typed_subtree_crossover(a,b,rng,feature_units) if typed else (subtree_crossover_genome(a,b,rng,max_depth_fast,max_nodes_fast) if basic_admissible and a.root is None and b.root is None else (subtree_crossover_flat(a,b,rng) if basic_admissible else subtree_crossover(a,b,rng)))
                if child is not None: child.metadata["age"]=max(int(a.metadata.get("age",0)),int(b.metadata.get("age",0)))+1
            elif child is None and r<float(self.p_crossover)+float(self.p_subtree_mutation):
                child=(typed_subtree_mutation(a,rng,self.n_features_in_,self.function_set_,int(self._cfg("max_depth")),feature_units,self.const_range,self.const_probability,self.constant_units)
                       if typed else (subtree_mutation_genome(a,rng,self.n_features_in_,self.function_set_,int(self._cfg("max_depth")),max_nodes_fast,self.const_range,self.const_probability) if basic_admissible and a.root is None else (subtree_mutation_flat(a,rng,self.n_features_in_,self.function_set_,int(self._cfg("max_depth")),self.const_range,self.const_probability) if basic_admissible else subtree_mutation(a,rng,self.n_features_in_,self.function_set_,int(self._cfg("max_depth")),self.const_range,self.const_probability))))
                if child is not None: child.metadata["age"]=int(a.metadata.get("age",0))+1
            elif child is None:
                child=typed_point_mutation(a,rng,self.function_set_,self.n_features_in_,feature_units,self.const_range,self.constant_units) if typed else (point_mutation_genome(a,rng,self.function_set_,self.n_features_in_,self.const_range) if basic_admissible and a.root is None else (point_mutation_flat(a,rng,self.function_set_,self.n_features_in_,self.const_range) if basic_admissible else point_mutation(a,rng,self.function_set_,self.n_features_in_,self.const_range)))
                child.metadata["age"]=int(a.metadata.get("age",0))+1
            if child is None:
                self._variation_rejections_=int(getattr(self,"_variation_rejections_",0))+1
                continue
            if typed: child.metadata["typed_variation"]=True
            if str(self.simplification).lower() in {"canonical","aggressive","algebraic"}: child=canonicalize_program(child,self.simplification)
            admissible=((child.depth<=max_depth_fast and child.size<=max_nodes_fast) if basic_admissible else self._admissible(child))
            if admissible: new.append(child)
            else: self._variation_rejections_=int(getattr(self,"_variation_rejections_",0))+1
        while len(new)<target:
            self._variation_fallbacks_=int(getattr(self,"_variation_fallbacks_",0))+1
            q=self._make_program(rng); q.metadata["age"]=0
            if pop and pop[0].root is None: q=q.as_flat_only()
            new.append(q)
        return new

    def _restore_resume_state(self,Xtr,ytr):
        state=None
        if bool(self.resume_from_checkpoint) and self.checkpoint_path and Path(self.checkpoint_path).exists():
            saved=self.load_checkpoint(self.checkpoint_path); state=getattr(saved,"evolution_state_",None)
        elif bool(self.warm_start): state=getattr(self,"evolution_state_",None)
        if state and state.get("dataset_fingerprint")!=self._dataset_fingerprint(Xtr,ytr):
            raise ValueError("Checkpoint/warm-start dataset fingerprint does not match current training data")
        return state

    def _pareto_enabled(self):
        return str(self.optimization).lower() in {"pareto","nsga2","multiobjective"}

    def _needs_pareto_archive(self):
        # Scalar search should not pay for a Pareto archive unless the final
        # model selector explicitly requests Pareto geometry.
        return self._pareto_enabled() or str(self.model_selection).lower() in {"pareto_knee","knee","pareto"}

    def _should_snapshot_state(self, terminal=False, checkpoint_due=False):
        # A full evolution snapshot clones every program.  It is needed at the
        # end of a fit (for compatibility/warm-start), or immediately before a
        # requested checkpoint, but not on every ordinary generation.
        return bool(terminal or checkpoint_due)

    def _store_evolution_state(self,islands,rng,generation,best,best_val,stagnant,elapsed,Xtr,ytr):
        self._state_snapshot_count_=int(getattr(self,"_state_snapshot_count_",0))+1
        self.evolution_state_={
            "schema":4,"generation":int(generation),"islands":[[p.evolution_clone() for p in pop] for pop in islands],
            "rng_state":rng.bit_generator.state,"best":None if best is None else best.clone(),"best_val":float(best_val),
            "stagnant":int(stagnant),"elapsed_seconds":float(elapsed),"dataset_fingerprint":self._dataset_fingerprint(Xtr,ytr),
            "evaluation_count":int(self.evaluation_count_),"history":list(self.history_),"pareto_archive":[p.evolution_clone() for p in self.pareto_archive_],
            "semantic_archive":[p.evolution_clone() for p in getattr(self,"semantic_archive_",[])],
            "memetic_objective_evaluations":int(getattr(self,"memetic_objective_evaluations_",0)),
            "selection_rng_state":None if getattr(self,"_selection_rng_",None) is None else self._selection_rng_.bit_generator.state,
        }
        if best is not None:
            self.best_program_=best.clone(); self.best_expression_=best.to_string(self._symbolic_feature_names_)

    def _evolve(self,Xtr,ytr,Xv,yv,swtr=None,swv=None):
        state=self._restore_resume_state(Xtr,ytr); rng=np.random.default_rng(self.random_state); start=time.perf_counter()
        if state:
            rng.bit_generator.state=state["rng_state"]; islands=[[p.clone() for p in pop] for pop in state["islands"]]
            best=None if state["best"] is None else state["best"].clone(); best_val=float(state["best_val"]); stagnant=int(state["stagnant"])
            self.history_=list(state.get("history",[])); self.pareto_archive_=[p.clone() for p in state.get("pareto_archive",[])]; self.semantic_archive_=[p.clone() for p in state.get("semantic_archive",[])]
            self.evaluation_count_=int(state.get("evaluation_count",0)); self.memetic_objective_evaluations_=int(state.get("memetic_objective_evaluations",0)); start_gen=int(state["generation"])+1; elapsed_before=float(state.get("elapsed_seconds",0.0))
            sel_seed=None if self.random_state is None else (int(self.random_state)^0x5F3759DF)
            self._selection_rng_=np.random.default_rng(sel_seed)
            if state.get("selection_rng_state") is not None: self._selection_rng_.bit_generator.state=state["selection_rng_state"]
        else:
            sel_seed=None if self.random_state is None else (int(self.random_state)^0x5F3759DF)
            self._selection_rng_=np.random.default_rng(sel_seed)
            islands=self._initial_populations(rng); best=None; best_val=np.inf; stagnant=0; self.history_=[]; self.pareto_archive_=[]; self.semantic_archive_=[]; self.evaluation_count_=0; self.memetic_objective_evaluations_=0; start_gen=1; elapsed_before=0.0
        self._rootless_fast_used_=bool(islands and islands[0] and islands[0][0].root is None)
        generations=int(self._cfg("generations")); self.stop_reason_="complete"
        self._state_snapshot_count_=0; self._nsga2_calls_=0; self._pareto_archive_updates_=0; self._variation_attempts_=0; self._variation_rejections_=0; self._variation_fallbacks_=0
        self._stage_seconds_={"evaluation":0.0,"memetic":0.0,"semantic":0.0,"nsga2":0.0,"pareto_archive":0.0,"lexicase_cache_and_sort":0.0,"variation":0.0,"migration":0.0,"state_snapshot":0.0,"checkpoint":0.0}
        for gen in range(start_gen,generations+1):
            elapsed=elapsed_before+(time.perf_counter()-start)
            if self.time_budget is not None and elapsed>=float(self.time_budget): self.stop_reason_="time_budget"; break
            if self.evaluation_budget is not None and self.evaluation_count_>=int(self.evaluation_budget): self.stop_reason_="evaluation_budget"; break
            if 0<float(self._cfg("max_samples"))<1:
                n=max(32,int(len(Xtr)*float(self._cfg("max_samples")))); ids=rng.choice(len(Xtr),size=min(n,len(Xtr)),replace=False); Xe,ye=Xtr[ids],ytr[ids]; swe=None if swtr is None else swtr[ids]
            else: Xe,ye,swe=Xtr,ytr,swtr
            all_pop=[p for pop in islands for p in pop]
            t_eval=time.perf_counter()
            if len(islands)>1 and bool(self.island_parallel):
                self._evaluate_population(all_pop,Xe,ye,Xv,yv,swe,swv); island_eval_mode="cross_island_population_batch"
            else:
                for pop in islands: self._evaluate_population(pop,Xe,ye,Xv,yv,swe,swv)
                island_eval_mode="per_island"
            eval_seconds=time.perf_counter()-t_eval; self.execution_plan_["island_evaluation_mode"]=island_eval_mode

            # Lamarckian local refinement of selected elites. The refined program
            # keeps the lineage age and re-enters Pareto/semantic ranking below.
            tm=time.perf_counter(); islands,memetic_reports=self._apply_memetic_search(islands,Xe,ye,swe,Xv,yv,swv,gen); memetic_seconds=time.perf_counter()-tm
            self.memetic_objective_evaluations_ += int(sum(int(r.get("evaluations",0)) for r in memetic_reports))
            all_pop=[p for pop in islands for p in pop]

            ts=time.perf_counter()
            if self._semantic_enabled():
                self._assign_semantics(all_pop,Xe)
                self.semantic_archive_=update_semantic_archive(self.semantic_archive_,all_pop,self.semantic_archive_size)
            semantic_seconds=time.perf_counter()-ts

            # Rank the generation at most once.  The same Program objects live
            # inside the islands and all_pop, so global ranks are immediately
            # visible to island-local selection without repeated O(P^2) sorts.
            t_nsga=time.perf_counter()
            if self._pareto_enabled():
                assign_nsga2(all_pop,self.pareto_objectives_); self._nsga2_calls_+=1
            nsga_seconds=time.perf_counter()-t_nsga

            t_select=time.perf_counter()
            for pop in islands:
                self._prepare_lexicase_cache(pop,Xe,ye); pop.sort(key=self._selection_key)
            all_pop.sort(key=self._selection_key); gen_best=all_pop[0]
            selection_prepare_seconds=time.perf_counter()-t_select
            self._stage_seconds_["evaluation"]+=eval_seconds; self._stage_seconds_["memetic"]+=memetic_seconds; self._stage_seconds_["semantic"]+=semantic_seconds; self._stage_seconds_["nsga2"]+=nsga_seconds; self._stage_seconds_["lexicase_cache_and_sort"]+=selection_prepare_seconds

            t_archive=time.perf_counter()
            hof_limit=min(100,max(10,int(self.pareto_archive_size)))
            if self._needs_pareto_archive():
                self.pareto_archive_=update_archive(self.pareto_archive_,all_pop,self.pareto_archive_size,self.pareto_objectives_); self._pareto_archive_updates_+=1
                # Archive already contains the trade-off surface; only merge the
                # current top slice rather than re-hashing the whole population.
                hof_source=self.pareto_archive_+all_pop[:hof_limit]
            else:
                self.pareto_archive_=[]
                # Scalar HOF is top-K by definition. Merge two bounded top-K sets
                # instead of cloning/sorting the complete population each generation.
                hof_source=list(getattr(self,"hall_of_fame_",[]))+all_pop[:hof_limit]
            unique_hof={}
            for q in hof_source:
                key=q.structural_hash(); old_q=unique_hof.get(key)
                if old_q is None or self._selection_key(q)<self._selection_key(old_q): unique_hof[key]=q
            self.hall_of_fame_=list(sorted(unique_hof.values(),key=self._selection_key)[:hof_limit])
            archive_seconds=time.perf_counter()-t_archive
            self._stage_seconds_["pareto_archive"]+=archive_seconds
            if gen_best.validation_loss<best_val-float(self.min_delta): best=gen_best.clone(); best_val=gen_best.validation_loss; stagnant=0
            else: stagnant+=1
            if best is not None:
                self.exact_validation_champion_=best.clone(); self.best_validation_program_=best.clone()
            self._stagnant_=stagnant; div=float(np.mean([self._diversity(pop) for pop in islands])); ages=[int(p.metadata.get("age",0)) for p in all_pop]
            elapsed=elapsed_before+(time.perf_counter()-start)
            sem_report=semantic_archive_report(self.semantic_archive_) if self._semantic_enabled() else {"size":0,"mean_novelty":0.0,"max_novelty":0.0,"unique_semantics":0}
            self.history_.append({"generation":gen,"training_loss":gen_best.loss,"validation_loss":gen_best.validation_loss,"best_exact_validation_fitness":None if best is None else float(best_val),"constraint_penalty":gen_best.constraint_penalty,"nodes":gen_best.size,"depth":gen_best.depth,"diversity":div,"semantic_novelty":float(gen_best.metadata.get("semantic_novelty",0.0)),"semantic_archive_size":int(sem_report["size"]),"mean_age":float(np.mean(ages)) if ages else 0.0,"max_age":int(max(ages)) if ages else 0,"memetic_evaluations":int(sum(int(r.get("evaluations",0)) for r in memetic_reports)),"evaluations":self.evaluation_count_,"elapsed_seconds":elapsed,"islands":len(islands),"unique_structures":len({p.structural_hash() for p in all_pop}),"fitness_cache_hits":int(getattr(self,"_fitness_cache_hits_",0)),"fitness_cache_misses":int(getattr(self,"_fitness_cache_misses_",0)),"evaluation_seconds_generation":float(eval_seconds),"memetic_seconds_generation":float(memetic_seconds),"semantic_seconds_generation":float(semantic_seconds),"selection_prepare_seconds_generation":float(selection_prepare_seconds),"nsga2_seconds_generation":float(nsga_seconds),"pareto_archive_seconds_generation":float(archive_seconds)})
            self._console_generation(gen,generations,gen_best,div,elapsed)
            terminal=(stagnant>=int(self.patience)) or (gen>=generations)
            if not terminal:
                if len(islands)>1 and gen%int(self.migration_interval)==0:
                    tmig=time.perf_counter(); islands=self._migrate(islands); self._stage_seconds_["migration"]+=time.perf_counter()-tmig
                tv=time.perf_counter(); self._current_diversity_for_variation_=div
                try: islands=[self._make_offspring_population(pop,rng,Xe,ye) for pop in islands]
                finally: self._current_diversity_for_variation_=None
                self._stage_seconds_["variation"]+=time.perf_counter()-tv
            checkpoint_due=bool(self.checkpoint_path and int(self.checkpoint_interval)>0 and gen%int(self.checkpoint_interval)==0)
            if self._should_snapshot_state(terminal=terminal,checkpoint_due=checkpoint_due):
                tsnap=time.perf_counter(); self._store_evolution_state(islands,rng,gen,best,best_val,stagnant,elapsed,Xtr,ytr); self._stage_seconds_["state_snapshot"]+=time.perf_counter()-tsnap
            elif best is not None:
                # Preserve immediate predictability without cloning the full population.
                self.best_program_=best.clone(); self.best_expression_=best.to_string(self._symbolic_feature_names_)
            if checkpoint_due:
                tc=time.perf_counter(); self._atomic_checkpoint(self.checkpoint_path); self._stage_seconds_["checkpoint"]+=time.perf_counter()-tc
            if stagnant>=int(self.patience): self.stop_reason_="patience"; break
            if gen>=generations: break
        if best is None:
            all_pop=[p for island in islands for p in island]; self._evaluate_population(all_pop,Xtr,ytr,Xv,yv,swtr,swv); all_pop.sort(key=self._selection_key); best=all_pop[0].clone()

        # Preserve the public Pareto-reporting API in fast scalar mode without
        # paying archive maintenance on every generation. Build it once from
        # the compact hall of fame and the terminal population.
        if not getattr(self,"pareto_archive_",None):
            t_archive=time.perf_counter()
            report_pool=list(getattr(self,"hall_of_fame_",[]))+list(all_pop)
            self.pareto_archive_=update_archive([],report_pool,self.pareto_archive_size,self.pareto_objectives_); self._pareto_archive_updates_+=1
            self._stage_seconds_["pareto_archive"]+=time.perf_counter()-t_archive

        candidates=[best]+[p.clone() for p in getattr(self,"pareto_archive_",[])]+[p.clone() for p in getattr(self,"hall_of_fame_",[])]
        selected,self.model_selection_report_=select_final_model(candidates,self.model_selection,self.selection_tolerance,X_validation=Xv)
        best=selected
        # V1.2.4 final expression-quality pass.  Evolution-time simplification
        # remains separately controlled by ``simplification`` so this does not
        # slow the hot search loop.
        reference_X = Xtr if Xv is None else np.vstack((Xtr, Xv))
        mode=str(self._cfg("final_simplification") or "off").lower()
        original_best=best.clone()
        report={"mode":mode,"passed":True,"original_nodes":int(best.size),"final_nodes":int(best.size),"nodes_removed":0,"reduction_fraction":0.0,"max_abs_prediction_delta":0.0,"reference_points":int(min(len(reference_X),int(self.simplification_reference_points)))}
        if mode not in {"off","none"}:
            if mode in {"certified","aggressive_certified"}:
                q,report=certified_simplify_program(best,reference_X,atol=float(self.simplification_tolerance),rtol=float(self.simplification_tolerance),max_reference_points=int(self.simplification_reference_points),admissible=self._admissible)
            elif mode in {"canonical"}:
                q=canonicalize_program(best,"canonical")
                report.update(final_nodes=int(q.size),nodes_removed=int(best.size-q.size),reduction_fraction=float((best.size-q.size)/max(1,best.size)))
            else:
                q=safe_simplify_program(best)
                report.update(final_nodes=int(q.size),nodes_removed=int(best.size-q.size),reduction_fraction=float((best.size-q.size)/max(1,best.size)))
            if self._admissible(q): best=q
            else: best=original_best
            self._fit_program_scaler(best,Xtr,ytr,swtr); best.loss=float(self._loss(ytr,self._prediction_from_program(best,Xtr),swtr)); best.validation_loss=float(self._validation_loss(yv,self._prediction_from_program(best,Xv),swv)) if Xv is not None else best.loss; best.constraint_penalty=float(self._constraint_penalty(best,Xtr))
        report["selected_nodes_before_simplification"]=int(original_best.size); report["selected_nodes_after_simplification"]=int(best.size); report["selected_validation_loss_before_simplification"]=float(original_best.validation_loss); report["selected_validation_loss_after_simplification"]=float(best.validation_loss)
        self.simplification_report_=report
        self.best_program_=best; self.best_expression_=best.to_string(self._symbolic_feature_names_); self.n_generations_=len(self.history_)
        self.run_time_seconds_=elapsed_before+(time.perf_counter()-start)
        total_cache=max(1,int(getattr(self,"_fitness_cache_hits_",0))+int(getattr(self,"_fitness_cache_misses_",0)))
        sem_report=semantic_archive_report(getattr(self,"semantic_archive_",[]))
        tracked_seconds=float(sum(self._stage_seconds_.values()))
        _fast_variation_profile=(str(self._cfg("selection_method")).lower()=="tournament" and not self._age_management_enabled() and not self._typed_generation_enabled() and str(self.unit_policy).lower()!="enforce" and not (self.constraints and str((self.constraints or {}).get("mode",(self.constraints or {}).get("constraint_mode","soft"))).lower()=="hard"))
        self.profile_={"version":__version__,"runtime_seconds":self.run_time_seconds_,"tracked_stage_seconds":tracked_seconds,"untracked_overhead_seconds":max(0.0,float(self.run_time_seconds_-tracked_seconds)),"generations":self.n_generations_,"evaluations":self.evaluation_count_,"actual_program_evaluations":self.evaluation_count_,"memetic_objective_evaluations":int(getattr(self,"memetic_objective_evaluations_",0)),"programs_per_second":self.evaluation_count_/max(self.run_time_seconds_,1e-12),"best_nodes":best.size,"best_depth":best.depth,"stop_reason":self.stop_reason_,"fitness_cache_entries":len(getattr(self,"_fitness_cache_",{})),"fitness_cache_hits":int(getattr(self,"_fitness_cache_hits_",0)),"fitness_cache_misses":int(getattr(self,"_fitness_cache_misses_",0)),"fitness_cache_hit_rate":float(getattr(self,"_fitness_cache_hits_",0)/total_cache),"batch_duplicate_reuse":int(getattr(self,"_duplicate_fitness_reuse_",0)),"n_islands":len(islands),"cross_island_population_batching":bool(len(islands)>1 and self.island_parallel),"island_evaluation_mode":self.execution_plan_.get("island_evaluation_mode","per_island"),"variation_engine":("numba_rootless_genome_v2" if _fast_variation_profile and bool(getattr(self,"_rootless_fast_used_",False)) else ("flat_genome_inheritance_v1" if _fast_variation_profile else "generic_safe")),"tournament_sampling":"batched_exact_distribution" if _fast_variation_profile else "configured_generic","tournament_scheduler":"vectorized_cdf_v1" if _fast_variation_profile else "generic","semantic_archive":sem_report,"age_management":str(self.age_management),"model_selection":dict(self.model_selection_report_),"nsga2_calls":int(getattr(self,"_nsga2_calls_",0)),"pareto_archive_updates":int(getattr(self,"_pareto_archive_updates_",0)),"state_snapshot_count":int(getattr(self,"_state_snapshot_count_",0)),"variation_attempts":int(getattr(self,"_variation_attempts_",0)),"variation_rejections":int(getattr(self,"_variation_rejections_",0)),"variation_fallbacks":int(getattr(self,"_variation_fallbacks_",0)),"stage_seconds":dict(self._stage_seconds_)}
        self.feature_usage_=self._feature_usage(best); self.function_usage_=self._function_usage(best)

    def _constant_bounds(self,node):
        label=getattr(node,"unit",None)
        if label is not None and self.constant_units:
            items=self.constant_units.items() if isinstance(self.constant_units,dict) else ((u,None) for u in self.constant_units)
            from .units import parse_unit
            desired=parse_unit(label)
            for unit,spec in items:
                if parse_unit(unit)==desired:
                    if isinstance(spec,(tuple,list)) and len(spec)==2: return float(spec[0]),float(spec[1])
                    if isinstance(spec,dict): return float(spec.get("min",self.const_range[0])),float(spec.get("max",self.const_range[1]))
        return float(self.const_range[0]),float(self.const_range[1])

    def _feature_usage(self,p):
        vals=p.root.feature_indices(); return {str(self._symbolic_feature_names_[i]):{"count":vals.count(i),"fraction":vals.count(i)/max(len(vals),1),"index":i} for i in sorted(set(vals))}
    def _function_usage(self,p):
        vals=p.root.function_names(); return {f:vals.count(f) for f in sorted(set(vals))}

    def _atomic_checkpoint(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); prev=path.with_suffix(path.suffix+".prev")
        self.checkpoint_schema_version_=3; self.checkpoint_library_version_=__version__; self.checkpoint_saved_at_=time.time()
        joblib.dump(self,tmp,compress=3); probe=joblib.load(tmp)
        if int(getattr(probe,"checkpoint_schema_version_",0))!=3: tmp.unlink(missing_ok=True); raise RuntimeError("Checkpoint validation failed")
        if path.exists():
            try:
                if prev.exists(): prev.unlink()
                os.replace(path,prev)
            except OSError: pass
        os.replace(tmp,path)
    def save_checkpoint(self,path): self._atomic_checkpoint(path); return str(path)
    @classmethod
    def load_checkpoint(cls,path):
        path=Path(path); candidates=[path,path.with_suffix(path.suffix+".prev")]; errors=[]
        for candidate in candidates:
            if not candidate.exists(): continue
            try:
                obj=joblib.load(candidate); schema=int(getattr(obj,"checkpoint_schema_version_",1))
                if schema not in {1,2,3}: raise RuntimeError(f"Unsupported checkpoint schema {schema}")
                obj.checkpoint_recovered_from_=str(candidate); return obj
            except Exception as exc: errors.append(f"{candidate}: {exc}")
        raise RuntimeError("Could not load checkpoint. "+" | ".join(errors))

    def get_expression(self): check_is_fitted(self,"best_program_"); return self.best_program_.to_string(self._symbolic_feature_names_)
    def get_expression_stats(self):
        check_is_fitted(self,"best_program_"); p=self.best_program_; return {"nodes":p.size,"dag_nodes":p.dag_size,"depth":p.depth,"unique_variables":len(set(p.root.feature_indices())),"nonlinear_operators":sum(get_primitive(f).nonlinear for f in p.root.function_names())}
    def get_simplification_report(self):
        check_is_fitted(self,"best_program_"); return dict(getattr(self,"simplification_report_",{"mode":"off","passed":True,"original_nodes":self.best_program_.size,"final_nodes":self.best_program_.size,"nodes_removed":0}))
    def get_pareto_front(self): return [p.clone() for p in getattr(self,"pareto_archive_",[])]
    def get_pareto_front_dataframe(self):
        objs=getattr(self,"pareto_objectives_",normalize_objectives(self.pareto_objectives)); rows=[]
        for p in getattr(self,"pareto_archive_",[]):
            row={"expression":p.to_string(self._symbolic_feature_names_),"validation_loss":p.validation_loss,"nodes":p.size,"constraint_penalty":p.constraint_penalty,"feasible":bool(p.metadata.get("feasible",True))}
            for name,value in zip(objs,objective_values(p,objs)): row[f"objective:{name}"]=value
            rows.append(row)
        try:
            import pandas as pd; return pd.DataFrame(rows)
        except Exception: return rows
    def get_constraint_report(self,X=None):
        check_is_fitted(self,"best_program_")
        if X is None:
            report=dict(getattr(self,"constraint_report_",{}))
            if report: return report
            raise ValueError("X is required because no fitted constraint report is available")
        A=self._prepare_X(X); s,si=structural_constraint_penalty(self.best_program_,self.constraints,self._symbolic_feature_names_)
        u,ui=unit_constraint_penalty(self.best_program_,self._feature_units_list(),self.target_unit) if str(self.unit_policy).lower() in {"enforce","penalty"} else (0,None)
        b,bi,br=behavioral_constraint_penalty(self.best_program_,A,self._symbolic_feature_names_,self.constraints,return_report=True)
        issues=si+([ui] if ui else [])+bi
        return {"passed":not issues,"feasible":not issues,"issues":issues,"structural_penalty":float(s),"unit_penalty":float(u),"behavioral_penalty":float(b),"behavioral":br,"pareto_objectives":list(getattr(self,"pareto_objectives_",normalize_objectives(self.pareto_objectives)))}
    def get_unit_report(self):
        check_is_fitted(self,"best_program_")
        from .units import parse_unit, unit_to_string
        feature_units=self._feature_units_list()
        if feature_units is None or self.target_unit is None:
            return {"enabled":False,"passed":True,"feature_units":feature_units,"target_unit":self.target_unit}
        fu=[parse_unit(u) for u in feature_units]; expected=parse_unit(self.target_unit)
        try: got=infer_unit(self.best_program_.root,fu); passed=(got==expected); error=None
        except Exception as exc: got=None; passed=False; error=str(exc)
        constants=[]
        for path,node in self.best_program_.root.iter_paths():
            if node.kind=="const": constants.append({"path":path,"value":float(node.value),"unit":getattr(node,"unit",None) or "1"})
        return {"enabled":True,"passed":bool(passed),"feature_units":[unit_to_string(parse_unit(u)) for u in feature_units],"target_unit":unit_to_string(expected),"expression_unit":None if got is None else unit_to_string(got),"typed_constants":constants,"error":error}
    def history_dataframe(self):
        try:
            import pandas as pd; return pd.DataFrame(self.history_)
        except Exception: return list(self.history_)
    def summary(self,print_output=True):
        check_is_fitted(self,"best_program_"); d={"version":__version__,"estimator":self.__class__.__name__,"expression":self.get_expression(),"expression_stats":self.get_expression_stats(),"resolved_config":self.get_resolved_config(),"profile":dict(self.profile_)}
        if print_output: print(json.dumps(d,indent=2,default=str))
        return d
    def model_manifest(self):
        check_is_fitted(self,"best_program_"); return {"format":"FastSymbolicGP-model-manifest-v3","version":__version__,"estimator":self.__class__.__name__,"n_features_in":int(self.n_features_in_),"feature_names":[str(x) for x in self._symbolic_feature_names_],"expression":self.get_expression(),"resolved_config":self.get_resolved_config(),"run_fingerprint":getattr(self,"run_fingerprint_",None),"constraint_report":dict(getattr(self,"constraint_report_",{})),"unit_report":self.get_unit_report(),"semantic_report":self.get_semantic_report(),"model_selection":dict(getattr(self,"model_selection_report_",{}))}
    def _make_fingerprint(self,X,y):
        h=hashlib.sha256(); h.update(np.ascontiguousarray(X).tobytes()); h.update(np.ascontiguousarray(y).tobytes()); h.update(json.dumps(self.get_params(deep=False),sort_keys=True,default=str).encode()); return h.hexdigest()
    def _finalize_fit(self,X,y):
        self.run_fingerprint_=self._make_fingerprint(X,y)
        if hasattr(self,"best_program_"):
            try:
                A=np.ascontiguousarray(np.asarray(X,float))
                s,si=structural_constraint_penalty(self.best_program_,self.constraints,self._symbolic_feature_names_)
                u,ui=unit_constraint_penalty(self.best_program_,self._feature_units_list(),self.target_unit) if str(self.unit_policy).lower() in {"enforce","penalty"} else (0,None)
                b,bi,br=behavioral_constraint_penalty(self.best_program_,A,self._symbolic_feature_names_,self.constraints,return_report=True)
                issues=si+([ui] if ui else [])+bi
                self.constraint_report_={"passed":not issues,"feasible":not issues,"issues":issues,"structural_penalty":float(s),"unit_penalty":float(u),"behavioral_penalty":float(b),"behavioral":br,"pareto_objectives":list(getattr(self,"pareto_objectives_",normalize_objectives(self.pareto_objectives)))}
            except Exception as exc: self.constraint_report_={"passed":False,"issues":[f"constraint report unavailable: {exc}"]}
        return self
    @staticmethod
    def validate_search_space(param_grid):
        issues=[]
        for k,v in dict(param_grid).items():
            vals=v if isinstance(v,(list,tuple,set)) else [v]
            if k=="population_size" and any(int(x)<4 for x in vals): issues.append("population_size must be >= 4")
            if k=="max_depth" and any(int(x)<1 for x in vals): issues.append("max_depth must be >= 1")
            if k=="selection_method" and any(str(x).lower() not in {"tournament","semantic_tournament","lexicase","epsilon_lexicase","epsilon-lexicase"} for x in vals): issues.append("unsupported selection_method")
        return {"valid":not issues,"issues":issues}
    def get_semantic_report(self):
        """Return Semantic GP 2.0 archive and current champion diagnostics."""
        check_is_fitted(self,"best_program_")
        report=semantic_archive_report(getattr(self,"semantic_archive_",[]))
        report.update({"enabled":bool(self._semantic_enabled()),"champion_novelty":float(self.best_program_.metadata.get("semantic_novelty",0.0)),"archive_size_limit":int(self.semantic_archive_size),"sample_size":int(self.semantic_sample_size),"neighbors":int(self.semantic_neighbors)})
        return report

    def get_evolution_diagnostics(self):
        """Compact diagnostics for diversity, age layers, memetic search and selection."""
        check_is_fitted(self,"best_program_")
        ages=[]
        state=getattr(self,"evolution_state_",{}) or {}
        for pop in state.get("islands",[]): ages.extend(int(p.metadata.get("age",0)) for p in pop)
        return {"age_management":str(self.age_management),"age_layers":int(self.age_layers),"age_gap":int(self.age_gap),"mean_age":float(np.mean(ages)) if ages else 0.0,"max_age":int(max(ages)) if ages else 0,"semantic":self.get_semantic_report(),"memetic_search":str(self.memetic_search),"memetic_objective_evaluations":int(getattr(self,"memetic_objective_evaluations_",0)),"model_selection":dict(getattr(self,"model_selection_report_",{})),"stage_seconds":dict(getattr(self,"_stage_seconds_",{}))}

    def explain(self):
        check_is_fitted(self,"best_program_"); return {"equation":self.get_expression(),"complexity":self.get_expression_stats(),"feature_usage":dict(self.feature_usage_),"function_usage":dict(self.function_usage_),"pareto_models":len(getattr(self,"pareto_archive_",[])),"constraint_issues":self.best_program_.metadata.get("constraint_issues",[]),"run_fingerprint":getattr(self,"run_fingerprint_",None),"constraint_report":dict(getattr(self,"constraint_report_",{})),"unit_report":self.get_unit_report(),"semantic_report":self.get_semantic_report(),"model_selection":dict(getattr(self,"model_selection_report_",{}))}
    def explain_prediction(self,X):
        A=self._prepare_X(X); pred=self.predict(A); grad=self.gradient(A); return {"prediction":np.asarray(pred).tolist(),"local_gradient":grad.tolist(),"features":[str(x) for x in self._symbolic_feature_names_]}
    def to_sympy(self):
        try: import sympy as sp
        except Exception as exc: raise ImportError("Install fastsymbolicgp[science] for SymPy export") from exc
        syms=[sp.Symbol(str(n)) for n in self._symbolic_feature_names_]
        def conv(n):
            if n.kind=="feature": return syms[int(n.value)]
            if n.kind=="const": return sp.Float(float(n.value))
            a=[conv(c) for c in n.children]; op=n.value
            mp={"add":lambda:a[0]+a[1],"sub":lambda:a[0]-a[1],"mul":lambda:a[0]*a[1],"div":lambda:a[0]/a[1],"sin":lambda:sp.sin(a[0]),"cos":lambda:sp.cos(a[0]),"tanh":lambda:sp.tanh(a[0]),"log":lambda:sp.log(sp.Abs(a[0])+sp.Float(1e-12)),"sqrt":lambda:sp.sqrt(sp.Abs(a[0])),"exp":lambda:sp.exp(a[0]),"abs":lambda:sp.Abs(a[0]),"neg":lambda:-a[0],"square":lambda:a[0]**2,"cube":lambda:a[0]**3,"inv":lambda:1/a[0]}
            return mp.get(op,lambda:sp.Function(str(op))(*a))()
        return sp.Float(self.best_program_.scale)*conv(self.best_program_.root)+sp.Float(self.best_program_.intercept)
    def predict_raw(self,X): check_is_fitted(self,"best_program_"); return self.best_program_.raw(self._prepare_X(X))
    def derivative(self,feature,X=None,eps=1e-5):
        check_is_fitted(self,"best_program_"); names=list(map(str,self._symbolic_feature_names_)); j=names.index(feature) if isinstance(feature,str) else int(feature)
        if X is None: raise ValueError("X is required for numerical derivative")
        A=self._prepare_X(X); scale=max(np.std(A[:,j]),1.0); h=eps*scale; Ap=A.copy(); Am=A.copy(); Ap[:,j]+=h; Am[:,j]-=h
        return (self.predict(Ap)-self.predict(Am))/(2*h)
    def gradient(self,X): return np.column_stack([self.derivative(i,X) for i in range(self.n_features_in_)])
    def doctor(self,X=None,y=None):
        from .diagnostics import doctor; return doctor(X,y,estimator=self)
    def branch(self,**parameter_changes):
        import copy
        check_is_fitted(self,"best_program_");b=copy.deepcopy(self);b.set_params(**parameter_changes);b.warm_start=True;b.resume_from_checkpoint=False;return b
    def save_history_csv(self,path):
        import csv
        rows=list(getattr(self,"history_",[])); path=Path(path)
        if not rows: path.write_text("",encoding="utf-8"); return str(path)
        fields=sorted({k for r in rows for k in r});
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        return str(path)
    def save_report_json(self,path):
        check_is_fitted(self,"best_program_");report={"version":__version__,"estimator":self.__class__.__name__,"parameters":self.get_params(deep=False),"n_features_in":int(self.n_features_in_),"n_generations":int(getattr(self,"n_generations_",0)),"run_time_seconds":float(getattr(self,"run_time_seconds_",0.0)),"stop_reason":getattr(self,"stop_reason_","complete"),"evaluation_count":int(getattr(self,"evaluation_count_",0)),"expression":self.get_expression(),"expression_stats":self.get_expression_stats(),"resolved_config":self.get_resolved_config(),"execution_plan":getattr(self,"execution_plan_",{}),"profile":getattr(self,"profile_",{}),"feature_usage":getattr(self,"feature_usage_",{}),"function_usage":getattr(self,"function_usage_",{}),"pareto_front":[{"loss":p.validation_loss,"nodes":p.size,"dag_nodes":p.dag_size,"depth":p.depth,"expression":p.to_string(self._symbolic_feature_names_)} for p in getattr(self,"pareto_archive_",[])]}
        Path(path).write_text(json.dumps(report,indent=2,default=str),encoding="utf-8");return str(path)
    def save_expression(self,path): Path(path).write_text(str(self.get_expression())+"\n",encoding="utf-8"); return str(path)
    def save_model(self,path): check_is_fitted(self,"best_program_");joblib.dump(self,path);return str(path)
    @classmethod
    def load_model(cls,path):
        obj=joblib.load(path)
        if not isinstance(obj,cls): raise TypeError(f"Stored object is {type(obj).__name__}, expected {cls.__name__}")
        return obj
    def save(self,path): return self.save_model(path)
    @classmethod
    def load(cls,path): return cls.load_model(path)
    def profile_prediction(self,X,repeats=100,warmup=5):
        import tracemalloc
        A=self._prepare_X(X);
        for _ in range(max(0,int(warmup))): self.predict(A)
        tracemalloc.start();t=time.perf_counter()
        for _ in range(max(1,int(repeats))): self.predict(A)
        elapsed=time.perf_counter()-t;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();calls=max(1,int(repeats));samples=max(1,A.shape[0]*calls)
        return {"repeats":calls,"batch_size":int(A.shape[0]),"seconds_total":elapsed,"milliseconds_per_batch":elapsed*1000/calls,"microseconds_per_sample":elapsed*1e6/samples,"peak_working_memory_bytes":int(peak),"tree_operations":self.best_program_.size,"dag_operations":self.best_program_.dag_size}
    def export_python(self,path):
        from .exporter import export_model;return export_model(self,path,"python")
    def export_c(self,path):
        from .exporter import export_model;return export_model(self,path,"c")
    def export_cpp(self,path):
        from .exporter import export_model;return export_model(self,path,"cpp")
    def export_java(self,path):
        from .exporter import export_model;return export_model(self,path,"java")
    def export_kotlin(self,path):
        from .exporter import export_model;return export_model(self,path,"kotlin")
    def export_javascript(self,path):
        from .exporter import export_model;return export_model(self,path,"javascript")
    def export_reproducibility_bundle(self,path,X=None,y=None):
        """Export reproducibility bundle v2 with certificate and content hashes."""
        check_is_fitted(self,"best_program_")
        from .publication import write_reproducibility_bundle_v2
        return write_reproducibility_bundle_v2(self,path,X=X,y=y)
    def get_scientific_certificate(self,X=None):
        check_is_fitted(self,"best_program_")
        from .publication import scientific_certificate
        return scientific_certificate(self,X=X)
    def save_scientific_certificate(self,path,X=None):
        cert=self.get_scientific_certificate(X=X); Path(path).write_text(json.dumps(cert,indent=2,default=str),encoding="utf-8"); return str(path)
    def robustness_report(self,X,noise_levels=(0.001,0.01,0.05),repeats=5,random_state=0):
        from .publication import robustness_report
        return robustness_report(self,X,noise_levels=noise_levels,repeats=repeats,random_state=random_state)
