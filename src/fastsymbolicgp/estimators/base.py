import numpy as np

from fastsymbolicgp.core.program import (
    random_program,
    point_mutation,
    subtree_mutation,
    subtree_crossover,
    hoist_mutation,
)
from fastsymbolicgp.core.evaluator import evaluate_program
from fastsymbolicgp.utils.export import save_elite_expressions_csv, save_history_csv


class BaseSymbolicEvolution:
    """
    Shared symbolic evolution engine.

    Subclasses must implement:
        _fitness_from_scores(scores, y_target)
    """

    def __init__(
        self,
        population_size=300,
        generations=30,
        max_depth=4,
        max_nodes=63,
        init_method="grow",
        function_set="default",
        elite_fraction=0.10,
        tournament_size=5,
        crossover_rate=0.45,
        subtree_mutation_rate=0.25,
        point_mutation_rate=0.20,
        hoist_mutation_rate=0.05,
        const_range=(-1.0, 1.0),
        parsimony=0.001,
        subsample=1.0,
        backend="auto",
        cache_programs=True,
        random_state=None,
        verbose=1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.init_method = init_method
        self.function_set = function_set
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.subtree_mutation_rate = subtree_mutation_rate
        self.point_mutation_rate = point_mutation_rate
        self.hoist_mutation_rate = hoist_mutation_rate
        self.const_range = const_range
        self.parsimony = parsimony
        self.subsample = subsample
        self.backend = backend
        self.cache_programs = cache_programs
        self.random_state = random_state
        self.verbose = verbose

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return np.ascontiguousarray(X)

    def _fitness_from_scores(self, scores, y_target):
        raise NotImplementedError

    def _evaluate_program_fitness(self, program, X, y_target, cache=None):
        if cache is not None:
            key = program.key()
            if key in cache:
                return cache[key]

        try:
            scores = evaluate_program(program.ops, program.args, X, backend=self.backend)

            if not np.all(np.isfinite(scores)):
                fitness = -1e18
            elif np.std(scores) < 1e-12:
                fitness = -1e18
            else:
                raw = self._fitness_from_scores(scores, y_target)
                fitness = float(raw) - float(self.parsimony) * float(program.size)

        except Exception:
            fitness = -1e18

        if cache is not None:
            cache[program.key()] = fitness

        return fitness

    def _tournament_select(self, population, rng):
        idx = rng.integers(0, len(population), size=self.tournament_size)
        best = None

        for i in idx:
            candidate = population[int(i)]
            if best is None or candidate.fitness_ > best.fitness_:
                best = candidate

        return best.copy()

    def _make_initial_population(self, rng, n_features):
        return [
            random_program(
                rng=rng,
                n_features=n_features,
                max_depth=self.max_depth,
                const_range=self.const_range,
                function_set=self.function_set,
                init_method=self.init_method,
            )
            for _ in range(self.population_size)
        ]

    def _make_child(self, population, rng, n_features):
        r = rng.random()

        if r < self.crossover_rate and len(population) >= 2:
            parent_a = self._tournament_select(population, rng)
            parent_b = self._tournament_select(population, rng)
            return subtree_crossover(parent_a, parent_b, rng, max_nodes=self.max_nodes)

        r -= self.crossover_rate

        if r < self.subtree_mutation_rate:
            parent = self._tournament_select(population, rng)
            child = subtree_mutation(
                parent,
                rng=rng,
                n_features=n_features,
                max_depth=self.max_depth,
                const_range=self.const_range,
                function_set=self.function_set,
            )
            if child.size <= self.max_nodes:
                return child
            return parent

        r -= self.subtree_mutation_rate

        if r < self.hoist_mutation_rate:
            parent = self._tournament_select(population, rng)
            return hoist_mutation(parent, rng)

        parent = self._tournament_select(population, rng)
        return point_mutation(
            parent,
            rng=rng,
            n_features=n_features,
            mutation_rate=self.point_mutation_rate,
            const_range=self.const_range,
            function_set=self.function_set,
        )

    def _evolve(self, X, y_target):
        rng = np.random.default_rng(self.random_state)
        X = self._prepare_X(X)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in the interval (0, 1].")

        n_samples, n_features = X.shape

        if self.subsample < 1.0:
            n_sub = max(10, int(n_samples * self.subsample))
            n_sub = min(n_sub, n_samples)
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            X_fit = X[idx]
            y_fit = np.asarray(y_target)[idx]
        else:
            X_fit = X
            y_fit = y_target

        # Warm-up evaluator.
        warm = random_program(
            rng=rng,
            n_features=n_features,
            max_depth=2,
            const_range=self.const_range,
            function_set=self.function_set,
        )
        _ = evaluate_program(warm.ops, warm.args, X_fit[: min(5, len(X_fit))], backend=self.backend)

        population = self._make_initial_population(rng, n_features)
        elite_count = max(1, int(self.population_size * self.elite_fraction))
        self.history_ = []

        for gen in range(self.generations):
            cache = {} if self.cache_programs else None

            for program in population:
                program.fitness_ = self._evaluate_program_fitness(program, X_fit, y_fit, cache=cache)

            population.sort(key=lambda p: p.fitness_, reverse=True)

            best = population[0]
            valid = [p.fitness_ for p in population if p.fitness_ > -1e17]
            mean_fit = float(np.mean(valid)) if valid else -1e18

            self.history_.append(
                {
                    "generation": gen + 1,
                    "best_fitness": float(best.fitness_),
                    "mean_fitness": mean_fit,
                    "best_size": int(best.size),
                    "best_depth": int(best.depth()),
                    "best_expression": best.to_string(),
                }
            )

            if self.verbose:
                print(
                    f"[{self.__class__.__name__}] Gen {gen + 1:03d}/{self.generations} | "
                    f"Best: {best.fitness_:.6f} | Mean: {mean_fit:.6f} | "
                    f"Size: {best.size} | Depth: {best.depth()}"
                )

            elites = [p.copy() for p in population[:elite_count]]
            new_population = elites.copy()

            while len(new_population) < self.population_size:
                child = self._make_child(population, rng, n_features)
                if child.size <= self.max_nodes and child.is_valid():
                    new_population.append(child)
                else:
                    new_population.append(self._tournament_select(population, rng))

            population = new_population

        # Final scoring on full data.
        final_cache = {} if self.cache_programs else None
        for program in population:
            program.fitness_ = self._evaluate_program_fitness(program, X, y_target, cache=final_cache)

        population.sort(key=lambda p: p.fitness_, reverse=True)

        self.population_ = population
        self.best_program_ = population[0].copy()

        return self

    def get_expression(self, feature_names=None):
        return self.best_program_.to_string(feature_names=feature_names)

    def get_latex_expression(self, feature_names=None):
        return self.best_program_.to_latex(feature_names=feature_names)

    def get_elite_expressions(self, n=20, feature_names=None):
        records = []
        for rank, program in enumerate(self.population_[:n], start=1):
            records.append(
                {
                    "rank": rank,
                    "fitness": program.fitness_,
                    "size": program.size,
                    "depth": program.depth(),
                    "expression": program.to_string(feature_names=feature_names),
                    "latex": program.to_latex(feature_names=feature_names),
                }
            )
        return records

    def save_elite_expressions(self, path, feature_names=None, n=20):
        return save_elite_expressions_csv(self.population_, path, feature_names=feature_names, n=n)

    def save_history(self, path):
        return save_history_csv(self.history_, path)
