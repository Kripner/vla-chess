#!/usr/bin/env python3
"""Episode splits over SUCCESSES: validation = the first K successful episodes of the dataset,
"n demonstrations" = the next n successes, in collection order or, with --shuffle-seed (the
real datasets: collection order drifts with the operator's practice and the session), in a
seeded random order. Nested: every size is a prefix of the next. A success = grade good or mid
in meta/outcomes.json (a lenient placement) AND marked (the stage marks exist). Writes
<out>/<dataset>_n<size>.json, one file per size, read by finetune_real --episodes.

    python -m chess.train.splits my_sim_dataset --sizes 100,316,1000,3162,10000
    python -m chess.train.splits <namespace>/<real dataset> --val 0 --sizes 8,32,128,256 --shuffle-seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chess.real import outcomes, segments


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="dataset name (<namespace>/<name>, or a bare name under local/)")
    ap.add_argument("--val", type=int, default=30, help="validation = the first K successes")
    ap.add_argument("--sizes", default="100,316,1000,3162,10000")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle the successes with this seed before cutting val and the nested "
                         "prefixes (default: collection order)")
    a = ap.parse_args()
    root = outcomes.dataset_root(a.data)
    name = root.name
    n = int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])
    grades = outcomes.load(root).get("outcomes") or {}
    unmarked = {int(e) for e in segments.unmarked(root)}
    succ = [e for e in range(n) if grades.get(str(e)) in ("good", "mid") and e not in unmarked]
    print(f"{name}: {n} episodes, {len(succ)} successes (good|mid, marked), "
          f"{sum(grades.get(str(e)) == 'bad' for e in range(n))} bad, {len(unmarked)} unmarked")
    if a.shuffle_seed is not None:
        random.Random(a.shuffle_seed).shuffle(succ)
    order = (f"shuffled (seed {a.shuffle_seed})" if a.shuffle_seed is not None
             else "in collection order")
    val = succ[: a.val]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for size in (int(x) for x in a.sizes.split(",")):
        train = succ[a.val: a.val + size]
        if len(train) < size:
            raise SystemExit(f"only {len(train)} successes after the {a.val} validation ones; "
                             f"n={size} needs more collection")
        f = out / f"{name}_n{size}.json"
        f.write_text(json.dumps({
            "dataset": name, "rule": f"successes only (good|mid, marked), {order}; val = first "
                                     f"{a.val} successes, train = the next {size}",
            "val": val, "train": train,
            "source": {"episodes": n, "successes": len(succ)}}, indent=0) + "\n")
        print(f"  {f.name}: train {len(train)} episodes, val {len(val)} ({order})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
