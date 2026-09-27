"""Rebuild every revisit result and figure from the committed data.

    cd revisit && uv run python run_all.py

External inputs download once into data/cache/ (hash-checked); nothing here
touches CDC WONDER, whose exports are already committed in data/wonder/.
"""
import runpy
import sys
import time

STEPS = [
    "r01_consistency", "r02_life_cycle", "r03_aian", "r04_age", "r05_supply",
    "r06_overlap", "r07_definitions", "r08_annual", "r09_recent_waves", "figures",
]

if __name__ == "__main__":
    for step in STEPS:
        t = time.time()
        print(f"\n=== {step} ===", flush=True)
        runpy.run_module(step, run_name="__main__")
        print(f"=== {step} done in {time.time() - t:.0f}s ===", flush=True)
    sys.exit(0)
