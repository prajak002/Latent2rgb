"""Experiment 6: does the sweep's (mode, epsilon, j) condition set have a
non-transitive "more stable than" relation -- i.e. a Condorcet-style cycle?

Two conditions A, B are compared by per-clip majority vote: A beats B if
A's pixel_error is lower than B's on more than half of clips that have both
(paired by clip_id). A transitive relation admits a total order consistent
with every pairwise result; brute-force search for a 3-cycle (A beats B,
B beats C, C beats A) is the direct test for a violation.
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=str, default="outputs/experiment4_results.csv")
    ap.add_argument("--out-json", type=str, default="outputs/experiment6_summary.json")
    ap.add_argument("--max-cycles-reported", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["condition"] = list(zip(df["mode"], df["epsilon"], df["j"]))
    pivot = df.pivot_table(index="clip_id", columns="condition", values="pixel_error")
    conditions = list(pivot.columns)
    print(f"{len(conditions)} conditions, {len(pivot)} clips")

    beats = {}  # (A, B) -> True if A beats B (only stored for A < B by index)
    for i, j in itertools.combinations(range(len(conditions)), 2):
        a, b = conditions[i], conditions[j]
        both = pivot[[a, b]].dropna()
        if both.empty:
            continue
        a_wins = (both[a] < both[b]).sum()
        b_wins = (both[b] < both[a]).sum()
        if a_wins == b_wins:
            continue  # tie, no clear majority
        beats[(i, j)] = a_wins > b_wins  # True: i beats j; False: j beats i

    def i_beats_j(i, j):
        if i == j:
            return None
        key, flip = ((i, j), False) if i < j else ((j, i), True)
        res = beats.get(key)
        if res is None:
            return None
        return (res if not flip else not res)

    cycles = []
    n = len(conditions)
    for i, j, k in itertools.combinations(range(n), 3):
        ij, jk, ki = i_beats_j(i, j), i_beats_j(j, k), i_beats_j(k, i)
        if ij is None or jk is None or ki is None:
            continue
        if ij and jk and ki:
            cycles.append((i, j, k))
        elif (not ij) and (not jk) and (not ki):
            cycles.append((k, j, i))
        if len(cycles) >= args.max_cycles_reported:
            break

    copeland = {i: 0 for i in range(n)}
    for (i, j), i_wins in beats.items():
        copeland[i if i_wins else j] += 1
    ranking = sorted(range(n), key=lambda i: -copeland[i])
    consistent_pairs = sum(
        1 for a_idx, i in enumerate(ranking) for j in ranking[a_idx + 1:]
        if i_beats_j(i, j) is True
    )
    total_decided_pairs = len(beats)

    print(f"\npairs with a clear majority: {total_decided_pairs} / {n * (n - 1) // 2}")
    print(f"3-cycles found (first {args.max_cycles_reported}): {len(cycles)}")
    for (i, j, k) in cycles[:args.max_cycles_reported]:
        print(f"  {conditions[i]} beats {conditions[j]} beats {conditions[k]} beats {conditions[i]}")
    print(f"\nCopeland-ranking consistency: {consistent_pairs}/{total_decided_pairs} decided pairs "
          f"agree with the ranking implied by total wins (100% would mean fully transitive)")

    with open(args.out_json, "w") as f:
        json.dump({
            "n_conditions": n,
            "n_clips": len(pivot),
            "n_decided_pairs": total_decided_pairs,
            "n_3cycles_found": len(cycles),
            "example_3cycles": [[str(conditions[i]), str(conditions[j]), str(conditions[k])] for (i, j, k) in cycles],
            "copeland_consistent_pairs": consistent_pairs,
            "copeland_total_decided_pairs": total_decided_pairs,
        }, f, indent=2)
    print(f"\nwrote {args.out_json}")
    print("\nRAW NUMBERS ONLY ABOVE.")


if __name__ == "__main__":
    main()
