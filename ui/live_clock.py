"""
Wall-clock-synchronized live status clock for Voychinet.

Redraws one dedicated header row every second, aligned to OS wall seconds
(so HH:MM:SS flips with the real clock, not a drifting sleep(1) loop).

Requires a terminal that supports ANSI cursor position (CUP) and save/restore
cursor — enabled automatically on modern Windows Terminal / ConHost with VT.
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from ui.terminal import get_terminal_columns, strip_ansi

# 1-based row: must match VoychinetEngine._header() when the clock rail is printed
# (logo 1 + banner 7 + reset 1 + status 1 + THIS line + border)
CLOCK_RAIL_ROW = 11

SnapshotFn = Callable[[], Tuple[Dict[str, str], Dict[str, str]]]


def _sleep_until_next_wall_second() -> None:
    """Avoid cumulative drift from fixed sleep(1)."""
    now = time.time()
    frac = now % 1.0
    wait = 1.0 - frac
    if wait < 0.02:
        wait += 1.0
    end = time.time() + wait
    while time.time() < end:
        time.sleep(min(0.05, end - time.time()))


class StatusBarClock:
    """
    Daemon thread: paints a framed digital time + date on a fixed terminal row.
    Uses save/restore cursor so interactive prompts are not disturbed.
    """

    def __init__(self, row: int, snapshot: SnapshotFn):
        self._row = row
        self._snapshot = snapshot
        self._running = False
        self._suppressed = False
        self._thread: Optional[threading.Thread] = None
        self._supported = bool(sys.stdout.isatty())

    @property
    def supported(self) -> bool:
        return self._supported

    def set_supported(self, value: bool) -> None:
        self._supported = bool(value)

    def _build_segments(self, T: Dict[str, str], L: Dict[str, str], now: datetime) -> Tuple[str, str, int, str]:
        """
        Left block (date + label), right block (large HH:MM:SS with blink colon).
        Returns (left_ansi, right_ansi, visible_width_right, time_plain).
        """
        D, R = T.get("D", ""), T.get("R", "")
        key = T.get("key", "")
        clk = T.get("clock", "")
        B = T.get("B", "")

        label = L.get("clock_rail", "LOCAL TIME")
        dow = now.strftime("%a")
        date_s = now.strftime("%Y-%m-%d")

        blink_on = int(time.time()) % 2 == 0
        sep = ":" if blink_on else " "
        h, m, s = now.hour, now.minute, now.second
        time_plain = f"{h:02d}{sep}{m:02d}{sep}{s:02d}"

        left = (
            f"  {D}\u2502{R} {D}\u25cb{R} {B}{label}{R} {D}\u2500{R} "
            f"{key}{dow}{R} {D}\u00b7{R} {key}{date_s}{R}"
        )
        right = f" {clk}{B} {time_plain} {R} {D}\u2502{R} "

        rw = len(strip_ansi(right))
        return left, right, rw, time_plain

    def _render(self) -> None:
        if not self._supported:
            return
        try:
            T, L = self._snapshot()
        except Exception:
            return

        cols = min(max(52, get_terminal_columns()), 120)
        now = datetime.now()
        left, right, rw, time_plain = self._build_segments(T, L, now)

        lw = len(strip_ansi(left))
        right_col = min(cols, max(lw + 2, cols - rw + 1))
        clk = T.get("clock", "")
        B = T.get("B", "")
        R = T.get("R", "")
        if lw + rw + 3 > cols:
            compact = f"{clk}{B} {time_plain} {R}"
            seq = f"\033[s\033[{self._row};1H\033[2K{compact}\033[u"
        else:
            seq = (
                "\033[s"
                f"\033[{self._row};1H\033[2K"
                f"{left}"
                f"\033[{self._row};{right_col}H"
                f"{right}"
                "\033[u"
            )
        try:
            sys.stdout.write(seq)
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass

    def _loop(self) -> None:
        while self._running:
            if not self._suppressed and self._supported:
                self._render()
            if not self._running:
                break
            if self._suppressed:
                time.sleep(0.15)
            else:
                _sleep_until_next_wall_second()

    def start(self) -> None:
        if self._running or not self._supported:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="voychinet-clock", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def suppress(self) -> None:
        self._suppressed = True

    def resume(self) -> None:
        self._suppressed = False

    def refresh_now(self) -> None:
        """Draw immediately (e.g. right after _header paints the layout)."""
        if self._running and not self._suppressed and self._supported:
            self._render()

    @property
    def is_live(self) -> bool:
        return self._running and self._supported

    def placeholder_line(self, T: Dict[str, str], L: Dict[str, str]) -> str:
        """Static first paint: full-width rail so the row exists before the thread runs."""
        cols = min(max(52, get_terminal_columns()), 120)
        D, R = T.get("D", ""), T.get("R", "")
        key = T.get("key", "")
        label = L.get("clock_rail", "LOCAL TIME")
        now = datetime.now()
        B = T.get("B", "")
        left = (
            f"  {D}\u2502{R} {D}\u25cb{R} {B}{label}{R} {D}\u2500{R} "
            f"{key}{now.strftime('%a')}{R} {D}\u00b7{R} {key}{now.strftime('%Y-%m-%d')}{R}"
        )
        lw = len(strip_ansi(left))
        pad = max(0, cols - lw - 22)
        time_part = now.strftime("%H:%M:%S")
        right = f" {' ' * pad}{T.get('clock', '')}{T.get('B', '')} {time_part} {R} {D}\u2502{R}"
        return left + right
