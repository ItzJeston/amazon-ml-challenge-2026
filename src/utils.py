"""
Utility functions for Business Entity Resolution Pipeline.
Includes UTF-8 stdout setup, memory tracking, and logging utilities.
"""

import contextlib
import ctypes
from ctypes import wintypes
import io
import logging
import os
import sys
import time
import warnings
from typing import Generator, Optional

from src.config import MEMORY_BUDGET_GB, MEMORY_CEILING_GB

# Suppress benign warnings to avoid non-zero stderr exits on Windows/PowerShell
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. UTF-8 Stdout / Stderr Initialization
# ==============================================================================
def setup_utf8_stdout() -> None:
    """
    Enforce explicit UTF-8 stdout and stderr initialization.
    Prevents Windows cp1252 UnicodeEncodeError when printing Indic or French text.
    """
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except (io.UnsupportedOperation, ValueError, Exception):
            pass

    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except (io.UnsupportedOperation, ValueError, Exception):
            pass


# Execute immediately on module import
setup_utf8_stdout()


# ==============================================================================
# 2. Logging Setup
# ==============================================================================
def get_logger(name: str = "entity_resolution") -> logging.Logger:
    """Return a configured logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("utils")


# ==============================================================================
# 3. Memory Tracking & Hardware Profiling
# ==============================================================================
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_current_process_memory_mb() -> float:
    """
    Return current process working set memory (RSS) in Megabytes.
    Uses psutil if available, otherwise native Windows kernel32/psapi API.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    # Windows native ctypes fallback
    if sys.platform == "win32":
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass

    return 0.0


def get_current_process_memory_gb() -> float:
    """Return current process working set memory in Gigabytes."""
    return get_current_process_memory_mb() / 1024.0


@contextlib.contextmanager
def track_memory(stage_name: str) -> Generator[None, None, None]:
    """
    Context manager to track memory usage and execution time for a processing stage.
    Emits warnings if operational budget (6GB) is exceeded, and raises if ceiling (10GB) is breached.
    """
    start_time = time.time()
    start_mem_gb = get_current_process_memory_gb()
    logger.info(f"Starting stage: '{stage_name}' (Current RAM: {start_mem_gb:.2f} GB)")

    try:
        yield
    finally:
        end_time = time.time()
        end_mem_gb = get_current_process_memory_gb()
        elapsed_sec = end_time - start_time
        delta_mem_gb = end_mem_gb - start_mem_gb

        logger.info(
            f"Finished stage: '{stage_name}' | Elapsed: {elapsed_sec:.2f}s | "
            f"RAM: {end_mem_gb:.2f} GB (Delta: {delta_mem_gb:+.2f} GB)"
        )

        if end_mem_gb > MEMORY_CEILING_GB:
            raise MemoryError(
                f"Peak memory limit exceeded in stage '{stage_name}': "
                f"{end_mem_gb:.2f} GB > {MEMORY_CEILING_GB} GB ceiling!"
            )
        elif end_mem_gb > MEMORY_BUDGET_GB:
            logger.warning(
                f"Peak memory warning in stage '{stage_name}': "
                f"{end_mem_gb:.2f} GB exceeds target budget of {MEMORY_BUDGET_GB} GB"
            )
