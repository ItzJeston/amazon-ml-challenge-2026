"""
run_full_benchmark.py — End-to-End Full Benchmark Pipeline
Amazon Business Entity Resolution Challenge

Sequentially executes:
1. Stage 1: Full Blocking (all 441,365 validation queries against full S2 & S3 pool)
2. Stage 2: Feature Engineering (Rapidfuzz features + negative subsampling)
3. Stage 3: LightGBM Classifier & Decision Threshold Search for Macro F0.5
"""

import os
import sys
import time
import subprocess
import logging

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
PYTHON = sys.executable

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def run_step(step_name: str, script_name: str, args: list = None):
    args = args or []
    script_path = os.path.join(SRC_DIR, script_name)
    cmd = [PYTHON, script_path] + args
    log.info(f"==================================================")
    log.info(f"STARTING {step_name}: {' '.join(cmd)}")
    log.info(f"==================================================")
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"FAILURE in {step_name} (exit code: {result.returncode})")
        sys.exit(result.returncode)
    elapsed = time.time() - t0
    log.info(f"COMPLETED {step_name} in {elapsed:.1f}s ({elapsed/60:.1f} min)")

def main():
    t_start = time.time()
    log.info("🚀 STARTING FULL DATASET BENCHMARK PIPELINE")
    
    # Step 1: Blocking
    run_step("STAGE 1: CANDIDATE GENERATION (BLOCKING)", "blocking.py", ["--full"])
    
    # Step 2: Feature Engineering
    run_step("STAGE 2: FEATURE ENGINEERING", "features.py", ["--full"])
    
    # Step 3: Classifier & Threshold Search
    run_step("STAGE 3: CLASSIFIER & METRIC EVALUATION", "classifier.py", ["--full"])
    
    total_elapsed = time.time() - t_start
    log.info("==================================================")
    log.info(f"🎉 FULL PIPELINE COMPLETED SUCCESSFULLY in {total_elapsed/60:.1f} minutes!")
    log.info("==================================================")

if __name__ == '__main__':
    main()
