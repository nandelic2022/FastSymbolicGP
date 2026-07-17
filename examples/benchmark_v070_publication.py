from fastsymbolicgp.benchmark import run_publication_benchmark

run_publication_benchmark(
    output_dir="benchmark_v070_results",
    datasets=("breast_cancer", "iris", "wine"),
    algorithms=("fastsymbolicgp", "logistic", "random_forest", "hist_gradient_boosting", "svm"),
    runs=3,
    folds=5,
    population_size=30,
    generations=120,
    backend="auto",
    seed=42,
    fastsymbolic_params={
        "evolution_model": "islands",
        "n_islands": 4,
        "migration_interval": 10,
    },
)
