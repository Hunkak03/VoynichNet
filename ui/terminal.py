"""
Terminal capabilities: Windows VT100 processing, column width, ANSI stripping.
"""
from __future__ import annotations

import re
import shutil
import sys
from typing import Tuple

# CSI (most sequences), OSC (title etc.), simple two-char escapes (e.g. ESC 7/8)
_ANSI_ESCAPE = re.compile(
    r"\x1b\[[\d;?]*[ -/]*[@-~]|"  # CSI
    r"\x1b\][^\x07]*\x07|"       # OSC
    r"\x1b[@-_]"                 # two-byte
)


def strip_ansi(text: str) -> str:
    """Visible width for layout (CSI OSC sequences removed)."""
    return _ANSI_ESCAPE.sub("", text)


def get_terminal_columns(default: int = 80) -> int:
    try:
        return max(40, shutil.get_terminal_size((default, 24)).columns)
    except (OSError, AttributeError, ValueError):
        return default


def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    try:
        sz = shutil.get_terminal_size(default)
        return max(40, sz.columns), max(1, sz.lines)
    except (OSError, AttributeError, ValueError):
        return default


def enable_vt_processing() -> bool:
    """
    Enable ANSI cursor positioning on Windows console (ConPTY / VT).
    No-op on Unix when stdout is a TTY (sequences already honoured).
    """
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:
        return False
