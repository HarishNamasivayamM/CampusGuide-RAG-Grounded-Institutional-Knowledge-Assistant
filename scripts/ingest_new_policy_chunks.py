#!/usr/bin/env python3
"""Backward-compatible alias for rebuilding the current policy index.

The former ``new_policy_chunks.json`` input is not part of this repository.
Use ``scripts/reingest_policies.py`` to rebuild ``iit_policies`` from the
checked-in policy dataset.
"""

from reingest_policies import main


if __name__ == "__main__":
    main()
