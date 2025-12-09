#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from opcd.experiments.off_policy import OffPolicyConfig, evaluate


RESULT_FIELDNAMES = ("k", "experiment", "path", "accuracy")


@dataclass(frozen=True)
class Experiment:
    k: int
    experiment: str
    path: str

    def key(self) -> Tuple[int, str, str]:
        return (self.k, self.experiment, self.path)


def load_experiments(csv_path: Path) -> List[Experiment]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiments CSV not found: {csv_path}")

    experiments: List[Experiment] = []
    with csv_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row or not row.get("k"):
                continue
            experiments.append(
                Experiment(
                    k=int(row["k"]),
                    experiment=row["experiment"],
                    path=row["path"],
                )
            )
    return experiments


def load_existing_results(results_path: Path) -> Dict[Tuple[int, str, str], float]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return {}

    existing: Dict[Tuple[int, str, str], float] = {}
    with results_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        missing_columns = set(RESULT_FIELDNAMES) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Results CSV at "
                f"{results_path} is missing expected columns: {missing_columns}"
            )
        for row in reader:
            if not row:
                continue
            try:
                key = (int(row["k"]), row["experiment"], row["path"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row in {results_path}: {row}") from exc
            existing[key] = float(row["accuracy"])
    return existing


def append_result(results_path: Path, experiment: Experiment, accuracy: float) -> None:
    file_exists = results_path.exists()
    should_write_header = not file_exists or results_path.stat().st_size == 0

    with results_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULT_FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(
            {
                "k": experiment.k,
                "experiment": experiment.experiment,
                "path": experiment.path,
                "accuracy": f"{accuracy:.6f}",
            }
        )


def evaluate_experiment(experiment: Experiment) -> Tuple[Experiment, float]:
    logging.info(
        "Starting evaluation for k=%s experiment=%s",
        experiment.k,
        experiment.experiment,
    )
    config = OffPolicyConfig(k=experiment.k, experiment=experiment.experiment)
    accuracy = evaluate(experiment.path, config)
    logging.info(
        "Finished evaluation for k=%s experiment=%s accuracy=%.4f",
        experiment.k,
        experiment.experiment,
        accuracy,
    )
    return experiment, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluate() for each experiment row, skipping any that already "
            "have results."
        )
    )
    parser.add_argument(
        "--experiments-csv",
        default="experiments.csv",
        type=Path,
        help="Path to the experiments CSV file.",
    )
    parser.add_argument(
        "--results-csv",
        default="results.csv",
        type=Path,
        help="Path to the results CSV file (created if missing).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Optional cap on parallel evaluations. Defaults to 20.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    experiments_csv = args.experiments_csv.expanduser().resolve()
    results_csv = args.results_csv.expanduser().resolve()

    experiments = load_experiments(experiments_csv)
    existing_results = load_existing_results(results_csv)

    pending = [exp for exp in experiments if exp.key() not in existing_results]
    if not pending:
        logging.info("All experiments already evaluated. Nothing to do.")
        return

    max_workers = args.max_workers or os.cpu_count() or 1
    max_workers = max(1, min(max_workers, len(pending)))

    logging.info(
        "Queuing %s experiment(s) with up to %s worker(s).",
        len(pending),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_experiment = {
            executor.submit(evaluate_experiment, exp): exp for exp in pending
        }

        for future in as_completed(future_to_experiment):
            experiment = future_to_experiment[future]
            try:
                _, accuracy = future.result()
            except Exception:  # noqa: BLE001
                logging.exception(
                    "Evaluation failed for k=%s experiment=%s path=%s",
                    experiment.k,
                    experiment.experiment,
                    experiment.path,
                )
                continue

            append_result(results_csv, experiment, accuracy)

    logging.info("All pending experiments processed.")


if __name__ == "__main__":
    main()

