# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write direct-EEF task rows from installed LIBERO benchmark initial states."""

import argparse
import json
from pathlib import Path


def parse_seeds(value: str) -> list[int]:
    """Parse nonnegative seed IDs and inclusive ranges without duplicates."""
    seeds = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) not in (1, 2) or not all(item.isdecimal() for item in bounds):
            raise ValueError(f"Expected nonnegative seeds/ranges, got {part!r}.")
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first:
            raise ValueError(f"Seed range ends before it starts: {part!r}.")
        seeds.update(range(first, last + 1))
    return sorted(seeds)


def load_tasks(libero_type: str, suites: list[str]) -> list[dict]:
    if libero_type == "pro":
        from liberopro.liberopro import benchmark
    else:
        from libero.libero import benchmark

    tasks = []
    for suite in suites:
        bench = benchmark.get_benchmark(suite)()
        for task in range(bench.n_tasks):
            tasks.append(
                {
                    "suite": suite,
                    "task": task,
                    "language": bench.get_task(task).language,
                    "initial_states": len(bench.get_task_init_states(task)),
                }
            )
    return tasks


def build_rows(tasks: list[dict], seeds: list[int], split: str) -> list[dict]:
    """Build adaptive episodes with distinct, explicitly checked initial states."""
    rows = []
    for task in tasks:
        for seed in seeds:
            trials = task["initial_states"]
            if not 0 <= seed < trials:
                raise ValueError(
                    f"{task['suite']}/t{task['task']} has {trials} initial states; "
                    f"seed {seed} would alias seed {seed % trials if trials else '?'} "
                    "through RPent's seed modulo initial-state count."
                )
            answer = {
                "environment": "libero",
                "suite": task["suite"],
                "task": task["task"],
                "seed": seed,
                "init_state_id": seed,
            }
            rows.append(
                {
                    "prompt": task["language"],
                    "answer": json.dumps(answer, sort_keys=True),
                    "query_id": f"{split}-{task['suite']}-{task['task']}-{seed}",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-type", choices=("standard", "pro"), required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_goal", "libero_object", "libero_spatial"],
    )
    parser.add_argument(
        "--seeds", required=True, help="Initial-state indices, e.g. 0-31 or 0,4,9."
    )
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(
        load_tasks(args.libero_type, args.suites), parse_seeds(args.seeds), args.split
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.split}.jsonl"
    destination.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} episodes to {destination}")


if __name__ == "__main__":
    main()
