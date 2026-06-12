import csv


def elite_records(programs, feature_names=None, n=20):
    records = []

    for rank, program in enumerate(programs[:n], start=1):
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


def save_elite_expressions_csv(programs, path, feature_names=None, n=20):
    records = elite_records(programs, feature_names=feature_names, n=n)

    fieldnames = ["rank", "fitness", "size", "depth", "expression", "latex"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    return path


def save_history_csv(history, path):
    if not history:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return path

    fieldnames = list(history[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    return path
