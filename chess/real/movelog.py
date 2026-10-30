#!/usr/bin/env python3
"""Per-episode move log: the chess move each recorded episode is and the position it started from.
Reference sidecar at <dataset root>/meta/moves.json, keyed like meta/outcomes.json; training never
reads it. Entry: {task, piece, source, target (null on a pick), phase (null = monolithic | "pick" |
"place"), board ({"e7": "black pawn", ...} | null), board_source ("game": replayed truth, values
"<side> <type>"; "detected": the app's board, values red|teal|empty|uncertain), san, ply, game
(game-sourced only)}. board is the position when the recorded sub-action starts.
CLI: python -m chess.real.movelog my_dataset [--board EP]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess.real.outcomes import dataset_root      # same name -> same on-disk root resolution


def path(root: Path) -> Path:
    return root / "meta" / "moves.json"


def load(root: Path) -> dict:
    p = path(root)
    return json.loads(p.read_text()) if p.exists() else {"moves": {}}


def save(root: Path, data: dict) -> None:
    path(root).write_text(json.dumps(data, indent=1))


def moves(root: Path) -> dict[int, dict]:
    """{episode index: entry}."""
    return {int(k): v for k, v in (load(root).get("moves") or {}).items()}


def record(root: Path, episode: int, entry: dict) -> None:
    """Write one episode's entry; a re-record of the same index overwrites."""
    data = load(root)
    m = dict(data.get("moves") or {})
    m[str(episode)] = entry
    data["moves"] = m
    save(root, data)


def _ascii(board: dict) -> str:
    """8x8 grid, White at the bottom (the operator's view; the robot sits behind Black)."""
    def sym(v):
        if not v or v in ("empty", "uncertain"):
            return "." if v != "uncertain" else "?"
        if v in ("red", "teal"):
            return "O" if v == "red" else "x"
        side, kind = v.split()
        s = {"knight": "n"}.get(kind, kind[0])
        return s.upper() if side == "white" else s
    rows = []
    for rank in range(8, 0, -1):
        cells = " ".join(sym(board.get(f"{f}{rank}")) for f in "abcdefgh")
        rows.append(f"{rank}  {cells}")
    rows.append("   a b c d e f g h")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="dataset name (<namespace>/<name> or bare <name>)")
    ap.add_argument("--board", type=int, metavar="EP",
                    help="print the position this episode started from and exit")
    args = ap.parse_args()
    root = dataset_root(args.data)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"{root} not found")
    m = moves(root)
    if not m:
        print(f"{root.name}: no meta/moves.json (recorded without a move log, or not chess)")
        return 0
    if args.board is not None:
        e = m.get(args.board)
        if e is None:
            raise SystemExit(f"episode {args.board} is not in the move log")
        if not e.get("board"):
            raise SystemExit(f"episode {args.board} has no board (Detect was never run)")
        print(f"ep {args.board}: {e.get('task')}  [{e.get('board_source')}]")
        print(_ascii(e["board"]))
        return 0
    total = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
    print(f"{root.name}: {len(m)}/{total} episodes logged")
    for ep in sorted(m):
        e = m[ep]
        board = e.get("board") or {}
        n = sum(1 for v in board.values() if v not in ("empty", "uncertain"))
        extra = f"  {e['san']} (ply {e['ply']})" if e.get("san") else ""
        print(f"  ep {ep:4d}  {e.get('phase') or 'move':5s}  {e.get('piece'):6s} "
              f"{e.get('source') or '?'}->{e.get('target') or '-':4s} "
              f"{n:2d} pieces{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
