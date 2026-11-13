"""Step 2: run the Spark/SQL rollup from raw logs to the exposure table."""

from __future__ import annotations

import time

from bstr.aggregate import main

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"step seconds              : {time.time() - t0:.1f}")
