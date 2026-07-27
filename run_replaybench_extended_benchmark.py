#!/usr/bin/env python3
"""Public ReplayBench-PG entry point for the archived extended benchmark.

The implementation remains in ``run_fgcs_extended_benchmark.py`` so that
existing manifests, imports, and checksums continue to resolve. This wrapper
provides submission-aligned public naming without changing execution logic.
"""

from run_fgcs_extended_benchmark import main


if __name__ == "__main__":
    main()
