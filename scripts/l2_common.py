"""Shared helpers for the Level-2 research scripts: experiment ledger."""

from __future__ import annotations

import csv
import fcntl
from datetime import datetime, timezone
from pathlib import Path

LOG = Path("experiment_log.csv")
FIELDS = [
    "exp_id", "timestamp_utc", "hypothesis", "partition", "underlying", "config",
    "scenario", "n", "metric", "value", "ci_lo", "ci_hi", "verdict", "notes",
]


def append_experiment_log(**kw) -> int:
    with open(LOG.parent / ".expLog.lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        new = not LOG.exists()
        if new:
            exp_id = 1
        else:
            with LOG.open() as f:
                exp_id = sum(1 for _ in f)  # header counts as row 1 -> ids start at 1
        row = {"exp_id": exp_id,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        row.update({k: kw.get(k, "") for k in FIELDS if k not in row})
        with LOG.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return exp_id
