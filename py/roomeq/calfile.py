"""miniDSP UMIK-1 calibration file (.txt): optional header with 'Sens Factor =-1.378dB, AGain =18dB, SERNO: 7000000',
then 'freq dB [phase]' lines."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Cal:
    freq: np.ndarray
    db: np.ndarray
    sens_factor: float | None = None
    name: str = ""
    serial: str | None = None
    orientation_deg: int | None = None     # 0 = aimed at the speakers, 90 = pointing up (from the file name)

    def at(self, f: np.ndarray) -> np.ndarray:
        """Mic deviation in dB at f (log-frequency interpolation, held flat outside the file range)."""
        return np.interp(np.log10(f), np.log10(self.freq), self.db)

    @property
    def label(self) -> str:
        return os.path.basename(self.name) if self.name else ""


def load(path: str) -> Cal:
    freqs, dbs, sens, serial = [], [], None, None
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            m = re.search(r"Sens\s*Factor\s*=\s*([+-]?[\d.]+)\s*dB", line, re.I)
            if m:
                sens = float(m.group(1))
            m = re.search(r"SERNO\s*[:=]\s*(\w+)", line, re.I)
            if m:
                serial = m.group(1)
            parts = re.split(r"[\s,;]+", line.strip().strip('"'))
            try:
                f, d = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            if f > 0:
                freqs.append(f)
                dbs.append(d)
    if len(freqs) < 10:
        raise ValueError(f"{path}: no usable calibration data found")
    order = np.argsort(freqs)
    base = os.path.basename(path)
    m = re.search(r"(\d+)\s*deg", base, re.I)
    if serial is None:
        m2 = re.match(r"(\d{6,})", base)
        serial = m2.group(1) if m2 else None
    return Cal(np.array(freqs)[order], np.array(dbs)[order], sens, path, serial, int(m.group(1)) if m else None)
