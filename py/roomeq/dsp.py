"""Spectrum analysis and parametric-EQ fitting. Pure numpy/scipy, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize

FS = 48000
PER_OCT = 12


def band_centers(fmin: float = 20.0, fmax: float = 20000.0, per_oct: int = PER_OCT) -> np.ndarray:
    k = np.arange(int(np.ceil(per_oct * np.log2(fmin / 1000))), int(np.floor(per_oct * np.log2(fmax / 1000))) + 1)
    return 1000.0 * 2.0 ** (k / per_oct)


CENTERS = band_centers()


def power_mean_db(levels_db: list[np.ndarray] | np.ndarray) -> np.ndarray:
    a = np.asarray(levels_db, dtype=float)
    return 10 * np.log10(np.mean(10 ** (a / 10), axis=0))


def smooth_db(level_db: np.ndarray, width_oct: float = 1 / 3) -> np.ndarray:
    """Rectangular power smoothing over `width_oct` octaves."""
    half = max(int(round(width_oct * PER_OCT / 2)), 0)
    pw = 10 ** (np.asarray(level_db) / 10)
    out = np.empty_like(pw)
    for i in range(len(pw)):
        out[i] = pw[max(0, i - half):i + half + 1].mean()
    return 10 * np.log10(out)


def target_curve(freqs: np.ndarray, tilt_db_per_oct: float = 0.0) -> np.ndarray:
    return tilt_db_per_oct * np.log2(freqs / 1000.0)


def q_to_bw(q: float) -> float:
    return float((2 / np.log(2)) * np.arcsinh(1 / (2 * q)))


def bw_to_q(bw: float) -> float:
    return float(1 / (2 * np.sinh(np.log(2) * bw / 2)))


def peq_db(freqs: np.ndarray, f0: float, q: float, gain_db: float, fs: int = FS) -> np.ndarray:
    """RBJ peaking biquad magnitude in dB."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * c, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * c, 1 - alpha / A])
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs) / fs)
    h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
    return 20 * np.log10(np.abs(h))


@dataclass
class Filter:
    freq: float
    q: float
    gain: float

    @property
    def bw(self) -> float:
        return q_to_bw(self.q)


@dataclass
class FitSettings:
    fmin: float = 63.0
    fmax: float = 10000.0
    max_filters: int = 8
    max_boost: float = 3.0
    max_cut: float = -12.0
    q_min: float = 0.7
    q_max_low: float = 8.0     # below q_split
    q_max_high: float = 1.5    # above q_split: broad, gentle only (runbook rule)
    max_cut_high: float = -6.0 # deepest cut above q_split
    low_filters: int = 3       # of max_filters, how many are reserved for per-channel room modes (< q_split)
    q_split: float = 300.0
    tol: float = 1.5           # stop adding filters once residual is inside +/- tol dB
    min_gain: float = 0.7      # drop filters smaller than this
    boost_weight: float = 0.6  # boosts count less than cuts
    boost_min_bw: float = 0.6  # octaves: only a shortfall at least this wide (at half depth) may be boosted; narrower dips are cancellations
    ref_lo: float = 250.0
    ref_hi: float = 4000.0
    tilt: float = 0.0          # target slope, dB/oct about 1 kHz
    smooth_oct: float = 1 / 3


def _offset(tgt: np.ndarray, m: np.ndarray, ref: np.ndarray) -> float:
    """Level (dB) that puts the measurement on the target: the MEDIAN over the reference band.

    A mean is dragged by a few deep notches or one strong resonance in the band, and the fit then boosts or cuts broad
    regions just to compensate. The median ignores features narrower than half the band, so only the shape is corrected.
    """
    return float(np.median(tgt[ref] - m[ref]))


def _total(freqs, filters):
    t = np.zeros(len(freqs))
    for f in filters:
        t += peq_db(freqs, f.freq, f.q, f.gain)
    return t


def _cut_limit(f, s: FitSettings):
    return s.max_cut if f < s.q_split else s.max_cut_high


def _q_max(f, s: FitSettings):
    return s.q_max_low if f < s.q_split else s.q_max_high


def evaluate(freqs: np.ndarray, measured_db: np.ndarray, valid: np.ndarray, s: FitSettings, filters: list[dict]) -> dict:
    """Predicted result of applying `filters` (dicts freq/q/gain) to a measured response."""
    fl = [Filter(f["freq"], f["q"], f["gain"]) for f in filters]
    m = smooth_db(measured_db, s.smooth_oct)
    tgt = target_curve(freqs, s.tilt)
    ref = (freqs >= s.ref_lo) & (freqs <= s.ref_hi) & valid
    sel = (freqs >= s.fmin) & (freqs <= s.fmax) & valid
    if ref.sum() < 4 or sel.sum() < 4:
        raise ValueError("Not enough usable data (signal well above the room noise) to evaluate this channel.")
    corr = _total(freqs, fl)
    off0 = _offset(tgt, m, ref)
    off1 = _offset(tgt, m + corr, ref)
    e0, e1 = tgt - (m + off0), tgt - (m + corr + off1)
    return {
        "filters": [dict(f, bw=q_to_bw(f["q"])) for f in filters],
        "offset_db": off0, "target": tgt, "smoothed": m + off0, "correction": corr, "predicted": m + corr + off1,
        "rms_before": float(np.sqrt(np.mean(e0[sel] ** 2))), "rms_after": float(np.sqrt(np.mean(e1[sel] ** 2))),
        "max_before": float(np.max(np.abs(e0[sel]))), "max_after": float(np.max(np.abs(e1[sel]))),
        "trim_db": float(-max(0.0, corr.max())),
    }


def fit_peq(freqs: np.ndarray, measured_db: np.ndarray, valid: np.ndarray, s: FitSettings,
            weights: np.ndarray | None = None) -> dict:
    """Fit up to s.max_filters peaking filters so measured -> target.

    Greedy placement at the largest weighted error, then joint least-squares refinement.
    `weights` (0..1 per band, e.g. from seat-to-seat consistency) lowers the importance of bands that only some seats
    show: the EQ should fix what is common to the listening area, not chase one position.
    Returns filters, the level offset used, predicted response and error stats.
    """
    wt = np.ones(len(freqs)) if weights is None else np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    m = smooth_db(measured_db, s.smooth_oct)
    tgt = target_curve(freqs, s.tilt)
    ref = (freqs >= s.ref_lo) & (freqs <= s.ref_hi) & valid
    if ref.sum() < 4:
        raise ValueError("Not enough valid data in the reference band to level-match; check signal level / SNR.")
    offset = _offset(tgt, m, ref)
    err = tgt - (m + offset)                     # +ve = needs boost
    mask = (freqs >= s.fmin) & (freqs <= s.fmax) & valid
    w = np.where(mask, wt, 0.0)

    def eff(r):  # perceived badness: boosts count less
        return np.where(r < 0, -r, r * s.boost_weight)

    filters: list[Filter] = []
    resid = err.copy()
    blocked = np.zeros(len(freqs), dtype=bool)         # features the rules forbid correcting (see below)
    for _ in range(s.max_filters + len(freqs)):        # every pass adds a filter or blocks a feature: always terminates
        if len(filters) >= s.max_filters:
            break
        score = np.where(mask & ~blocked, eff(resid) * wt, 0)
        for f in filters:  # never stack a new filter on an existing one
            score[np.abs(np.log2(freqs / f.freq)) < 0.25] = 0
        i = int(np.argmax(score))
        if score[i] < s.tol:
            break
        sign = np.sign(resid[i])
        half = abs(resid[i]) * 0.5
        lo = hi = i
        while lo > 0 and np.sign(resid[lo - 1]) == sign and abs(resid[lo - 1]) > half:
            lo -= 1
        while hi < len(freqs) - 1 and np.sign(resid[hi + 1]) == sign and abs(resid[hi + 1]) > half:
            hi += 1
        bw = max((hi - lo + 1) / PER_OCT, 1 / PER_OCT)
        f0 = float(freqs[i])
        # Runbook rules: never boost below the split (dips there are cancellations that boosting cannot fill) and never boost a
        # narrow dip anywhere (reflections); leave such a feature, and the skirts around it, alone.
        no_boost = sign > 0 and (s.max_boost < s.min_gain or f0 < s.q_split or bw < s.boost_min_bw)
        no_cut = sign < 0 and _cut_limit(f0, s) > -s.min_gain
        if no_boost or no_cut:
            a = b = i
            while a > 0 and np.sign(resid[a - 1]) == sign and eff(resid[a - 1]) >= s.tol:
                a -= 1
            while b < len(freqs) - 1 and np.sign(resid[b + 1]) == sign and eff(resid[b + 1]) >= s.tol:
                b += 1
            blocked[a:b + 1] = True
            continue
        q = float(np.clip(bw_to_q(bw), s.q_min, _q_max(f0, s)))
        g = float(np.clip(resid[i] * 0.9, _cut_limit(f0, s), s.max_boost))
        filters.append(Filter(f0, q, g))
        resid = err - _total(freqs, filters)

    if filters:
        filters = _refine(freqs, err, w, filters, s)
        filters = _merge_close(filters, s)
        filters = _cap_boost(freqs, filters, s)
        filters = [f for f in filters if abs(f.gain) >= s.min_gain]
    filters.sort(key=lambda f: f.freq)

    corr = _total(freqs, filters)
    pred = m + offset + corr
    after = tgt - pred
    sel = mask
    return {
        "filters": [dict(asdict(f), bw=f.bw) for f in filters],
        "offset_db": offset,
        "target": tgt,
        "smoothed": m + offset,
        "correction": corr,
        "predicted": pred,
        "rms_before": float(np.sqrt(np.mean(err[sel] ** 2))),
        "rms_after": float(np.sqrt(np.mean(after[sel] ** 2))),
        "max_before": float(np.max(np.abs(err[sel]))),
        "max_after": float(np.max(np.abs(after[sel]))),
        "trim_db": float(-max(0.0, corr.max())),
    }


def _cap_boost(freqs: np.ndarray, filters: list[Filter], s: FitSettings) -> list[Filter]:
    """The refinement only penalises boost above the limit; overlapping boosts can still add up to a little more.
    Scale the boosts down until the COMBINED correction never exceeds the limit (it is a headroom/amplifier-power rule)."""
    for _ in range(30):
        top = float(_total(freqs, filters).max())
        if top <= s.max_boost + 1e-6:
            break
        k = max(s.max_boost, 0.0) / top
        filters = [Filter(f.freq, f.q, f.gain * k if f.gain > 0 else f.gain) for f in filters]
    return filters


def _merge_close(filters: list[Filter], s: FitSettings) -> list[Filter]:
    """Refinement can pull two filters onto the same spot; fold such pairs into one (stays within limits)."""
    fl = sorted(filters, key=lambda f: f.freq)
    i = 0
    while i < len(fl) - 1:
        a, b = fl[i], fl[i + 1]
        if np.log2(b.freq / a.freq) < 1 / 6 and np.sign(a.gain) == np.sign(b.gain):
            g = a.gain + b.gain
            fl[i:i + 2] = [Filter(float(np.sqrt(a.freq * b.freq)), min(a.q, b.q),
                                  float(np.clip(g, _cut_limit(a.freq, s), s.max_boost)))]
        else:
            i += 1
    return fl


def _refine(freqs, err, w, filters, s: FitSettings):
    n = len(filters)
    x0 = np.concatenate([[np.log(f.freq), np.log(f.q), f.gain] for f in filters])
    lb, ub = [], []
    for f in filters:
        # stay within half an octave of where the greedy pass put it, so the region rules above still hold; a cut stays a
        # cut and a boost stays a boost (and above the split)
        boost = f.gain > 0
        lb += [np.log(max(s.fmin * 0.7, f.freq / 1.41, 20, s.q_split if boost else 0.0)), np.log(s.q_min),
               0.0 if boost else _cut_limit(f.freq, s)]
        ub += [np.log(min(s.fmax * 1.2, f.freq * 1.41)), np.log(_q_max(f.freq, s)), s.max_boost if boost else 0.0]
    x0 = np.clip(x0, lb, ub)

    def res(x):
        fl = [Filter(float(np.exp(x[3 * k])), float(np.exp(x[3 * k + 1])), float(x[3 * k + 2])) for k in range(n)]
        tot = _total(freqs, fl)
        r = err - tot
        r = np.where(r > 0, r * s.boost_weight, r)
        over = np.maximum(tot - s.max_boost, 0) * 3.0  # combined boost must respect the limit too
        # light regulariser: prefer smaller corrections
        return np.concatenate([r * w, over * w, 0.02 * x[2::3]])

    sol = optimize.least_squares(res, x0, bounds=(lb, ub), max_nfev=200)
    x = sol.x
    return [Filter(float(np.exp(x[3 * k])), float(np.exp(x[3 * k + 1])), float(x[3 * k + 2])) for k in range(n)]


def verify(freqs, measured_db, valid, s: FitSettings, offset_db: float | None = None) -> dict:
    """Score a (post-correction) measurement against the target."""
    m = smooth_db(measured_db, s.smooth_oct)
    tgt = target_curve(freqs, s.tilt)
    ref = (freqs >= s.ref_lo) & (freqs <= s.ref_hi) & valid
    sel = (freqs >= s.fmin) & (freqs <= s.fmax) & valid
    if (offset_db is None and ref.sum() < 4) or sel.sum() < 4:
        raise ValueError("Not enough usable data (signal well above the room noise) to score this channel.")
    off = _offset(tgt, m, ref) if offset_db is None else offset_db
    err = tgt - (m + off)
    return {
        "smoothed": m + off,
        "offset_db": off,
        "rms": float(np.sqrt(np.mean(err[sel] ** 2))),
        "max": float(np.max(np.abs(err[sel]))),
    }


def _normalised(level_db: np.ndarray, freqs: np.ndarray, s: FitSettings, valid: np.ndarray | None = None) -> np.ndarray:
    """One seat's smoothed response, level-matched over the reference band (so seats at different distances compare)."""
    m = smooth_db(level_db, s.smooth_oct)
    ref = (freqs >= s.ref_lo) & (freqs <= s.ref_hi)
    if valid is not None and (ref & valid).sum() >= 4:
        ref &= valid
    return m - float(np.median(m[ref]))


def consistency(level_list: list[np.ndarray], freqs: np.ndarray, s: FitSettings, valid: np.ndarray | None = None,
                sigma0: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Seat-to-seat spread (dB, standard deviation) per band, and the 0..1 weight derived from it.

    Weight = 1 / (1 + (sigma / sigma0)^2): bands where the seats agree count fully; bands where they differ by
    ~sigma0 dB count half. With fewer than 3 seats there is no meaningful spread, so every weight is 1.
    """
    if len(level_list) < 3:
        return np.zeros(len(freqs)), np.ones(len(freqs))
    norm = np.array([_normalised(l, freqs, s, valid) for l in level_list])
    sigma = np.std(norm, axis=0, ddof=1)
    return sigma, 1.0 / (1.0 + (sigma / sigma0) ** 2)


def evaluate_positions(freqs: np.ndarray, level_list: list[np.ndarray], valid: np.ndarray, s: FitSettings,
                       filters: list[dict]) -> list[dict]:
    """Predicted error vs the target at EACH seat, before and after the correction (rms dB over the correction range).

    The fit works on the seat average; this shows whether every individual seat benefits.
    """
    fl = [Filter(f["freq"], f["q"], f["gain"]) for f in filters]
    corr = _total(freqs, fl)
    tgt = target_curve(freqs, s.tilt)
    ref = (freqs >= s.ref_lo) & (freqs <= s.ref_hi) & valid
    sel = (freqs >= s.fmin) & (freqs <= s.fmax) & valid
    out = []
    for lv in level_list:
        m = smooth_db(lv, s.smooth_oct)
        e0 = tgt - (m + _offset(tgt, m, ref))
        e1 = tgt - (m + corr + _offset(tgt, m + corr, ref))
        out.append({"rms_before": float(np.sqrt(np.mean(e0[sel] ** 2))), "rms_after": float(np.sqrt(np.mean(e1[sel] ** 2)))})
    return out
