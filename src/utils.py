"""
src/utils.py
------------
Shared utilities for the VolatilityRegimes pipeline.

Provides:
    atomic_write  — context manager for safe, crash-proof file writes
"""

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_write(path: Path):
    """
    Context manager for atomic file writes.

    Writes to a temporary .tmp file first. On success the .tmp file is
    renamed to the final path in a single OS operation (atomic on all
    POSIX systems). On failure the .tmp file is removed, leaving any
    previously existing output intact and uncorrupted.

    This prevents DVC from seeing a partial/corrupt output if a stage
    crashes mid-write — the final file either exists and is complete,
    or does not exist at all, so dvc repro correctly detects the stage
    as incomplete and re-runs it.

    Usage
    -----
        # CSV
        with atomic_write(ROOT / 'data/processed/returns.csv') as tmp:
            df.to_csv(tmp)

        # Pickle / model save
        with atomic_write(ROOT / 'data/processed/hmm.pkl') as tmp:
            model.save(tmp)

        # JSON
        with atomic_write(ROOT / 'metrics.json') as tmp:
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)

    Parameters
    ----------
    path : Path
        Final destination path for the file.

    Yields
    ------
    tmp_path : Path
        Temporary path to write to. Do not use the final path inside
        the context block — write only to tmp_path.
    """
    path    = Path(path)
    tmp     = path.with_suffix(path.suffix + '.tmp')
    try:
        yield tmp
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
