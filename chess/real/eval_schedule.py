#!/usr/bin/env python3
"""Generate an eval schedule: a seeded chain of king moves (trial i's target is trial i+1's
source) with per-trial exact intra-cell offsets. easy = every offset (0, 0); hard =
seeded-uniform within task.OFFSET_MAX_MM. The schedules of the paper are committed in
schedules/ and never edited; an eval runs the first k trials, so any eval size is a prefix
and two evals are comparable iff (schedule id, k) match.

  python -m chess.real.eval_schedule --setup easy --n 1000 --seed 0 --out schedules/easy_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chess import task as T

REPO = Path(__file__).resolve().parents[2]
# Schedules are generated for the sim, so reachability is judged against the sim's nominal board.
BOARD_POSE = REPO / "chess" / "sim" / "board_pose.json"


def reachable_squares(board_pose: Path) -> list[tuple[int, int]]:
    """Squares whose hover the scripted planner can solve (the recorder's own filter)."""
    from chess.real.phases import Planner
    pl = Planner(str(board_pose))
    out = []
    for r in range(T.N):
        for c in range(T.N):
            try:
                pl.hover((r, c))
                out.append((r, c))
            except RuntimeError:
                continue
    return out


def generate(setup: str, n: int, seed: int, start: str, board_pose: Path) -> dict:
    rng = np.random.default_rng(seed)
    pool = reachable_squares(board_pose)
    if len(pool) < 2:
        raise SystemExit(f"only {len(pool)} reachable squares under {board_pose}")
    cur = T.cell_from_name(start)
    if cur not in pool:
        raise SystemExit(f"start square {start} is not reachable")
    trials, deck = [], []
    for i in range(n):
        # permutation decks over the reachable pool -> even square coverage (the recorder's
        # _chain_rounds pattern); skip the current square.
        while True:
            if not deck:
                deck = [pool[j] for j in rng.permutation(len(pool))]
            nxt = deck.pop()
            if nxt != cur:
                break
        if setup == "hard":
            dx, dy = (float(v) for v in rng.uniform(-T.OFFSET_MAX_MM, T.OFFSET_MAX_MM, 2))
        else:
            dx = dy = 0.0
        trials.append({"i": i, "source": T.cell_name(cur), "target": T.cell_name(nxt),
                       "dx_mm": round(dx, 2), "dy_mm": round(dy, 2)})
        cur = nxt
    return {"id": None, "setup": setup, "seed": seed, "piece": "king",
            "board": "empty-except-king", "board_pose": str(board_pose.relative_to(REPO)),
            "offset_max_mm": T.OFFSET_MAX_MM, "n": n, "trials": trials}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", required=True, choices=["easy", "hard"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="e4", help="first trial's source square")
    ap.add_argument("--out", required=True, help="e.g. schedules/easy_v1.json")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists — schedules are immutable; pick a new versioned name")
    sched = generate(a.setup, a.n, a.seed, a.start, BOARD_POSE)
    sched["id"] = out.stem
    # invariants before anything is written
    tr = sched["trials"]
    assert all(tr[i + 1]["source"] == tr[i]["target"] for i in range(len(tr) - 1)), "chain broken"
    assert all(t["source"] != t["target"] for t in tr), "self-move"
    assert all(abs(t["dx_mm"]) <= T.OFFSET_MAX_MM and abs(t["dy_mm"]) <= T.OFFSET_MAX_MM
               for t in tr), "offset out of bound"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sched, indent=1))
    srcs = {t["source"] for t in tr}
    print(f"{out}: {len(tr)} trials, {len(srcs)} distinct sources, setup {sched['setup']}, "
          f"seed {a.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
