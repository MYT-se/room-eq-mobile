"""Log-sweep measurement: build the playback, deconvolve the recording to impulse responses, analyse.

One 'run' plays a sweep on each channel in turn (same clock, one recording), so arrival times are comparable
between channels. The output device and the microphone have DIFFERENT clocks (typically 20-100 ppm apart), which
would shift later channels by about 0.5 ms per 8 s of run time. Runs with two or more channels therefore end with a
copy of the first channel's sweep: the time between the two copies gives the clock ratio, and each channel's
segment is resampled onto the output clock before analysis.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len

from . import dsp

PRE = 1.0           # silence before the first sweep (s); also where the room noise is measured
TAIL = 1.0          # extra recording after the last sweep (s)
SEG_PRE = 0.1       # margin before each channel's expected arrival (s)
KEEP = 0.8          # how much IR (after SEG_PRE) is kept for alignment maths (s)
NOISE_GUARD = 0.1   # the noise window ends this long before the end of the gap (s)
FDW_TMIN = 0.005    # shortest analysis window (s) - the analysis never uses less than this after the peak
CLOCK_MAX = 5e-4    # clock ratios further from 1 than this are rejected as measurement errors (500 ppm)
DETECT = 12.0       # matched-filter peak must exceed this many times the median to count as heard


class NotHeard(RuntimeError):
    """The first speaker's sweep was not found in the recording, so nothing in this run can be placed in time."""


class Run(dict):
    """Per-channel results of one run (name -> dict). `.meta` holds run-level facts: clock offset, room noise, tail check."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.meta: dict = {}


# ---------------------------------------------------------------------------------------------------------------
# frequency weightings (for SPL read-outs, never for the EQ analysis)
# ---------------------------------------------------------------------------------------------------------------
def _weight_sos(fs: int, n_zeros: int, poles_hz: list[float]) -> np.ndarray:
    """Analogue weighting prototype (n_zeros zeros at 0 Hz) -> bilinear transform, normalised to 0 dB at 1 kHz."""
    z, p, k = signal.bilinear_zpk([0] * n_zeros, [-2 * np.pi * f for f in poles_hz], (2 * np.pi * 12194.217) ** 2, fs)
    sos = signal.zpk2sos(z, p, k)
    _, h = signal.sosfreqz(sos, worN=[1000.0], fs=fs)
    sos[0, :3] /= abs(h[0])
    return sos


def c_weight_sos(fs: int) -> np.ndarray:
    """IEC 61672 C-weighting, 0 dB at 1 kHz."""
    return _weight_sos(fs, 2, [20.598997] * 2 + [12194.217] * 2)


def a_weight_sos(fs: int) -> np.ndarray:
    """IEC 61672 A-weighting, 0 dB at 1 kHz."""
    return _weight_sos(fs, 4, [20.598997] * 2 + [107.65265, 737.86223] + [12194.217] * 2)


def rms_weighted_dbfs(x: np.ndarray, fs: int, sos: np.ndarray) -> float:
    y = signal.sosfilt(sos, np.asarray(x, dtype=float))
    y = y[int(0.2 * fs):] if len(y) > 0.8 * fs else y
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12))


def rms_c_dbfs(x: np.ndarray, fs: int) -> float:
    """C-weighted rms in dBFS (a full-scale sine reads -3.01)."""
    return rms_weighted_dbfs(x, fs, c_weight_sos(fs))


def rms_a_dbfs(x: np.ndarray, fs: int) -> float:
    """A-weighted rms in dBFS."""
    return rms_weighted_dbfs(x, fs, a_weight_sos(fs))


# ---------------------------------------------------------------------------------------------------------------
# signal generation
# ---------------------------------------------------------------------------------------------------------------
def make_sweep(fs: int, dur: float, f1: float, f2: float) -> np.ndarray:
    n = int(dur * fs)
    t = np.arange(n) / fs
    r = np.log(f2 / f1)
    x = np.sin(2 * np.pi * f1 * dur / r * (np.exp(t * r / dur) - 1))
    nf = int(0.05 * fs)
    fade = 0.5 * (1 - np.cos(np.pi * np.arange(nf) / nf))
    x[:nf] *= fade
    x[-nf:] *= fade[::-1]
    return x


def build_tracks(sweep: np.ndarray, items: list[list[int]], amps_db: list[float], gap: float, fs: int, ref_last: bool = False):
    """One mono timeline per output channel. `items` = list of output-channel groups played together, in sequence.

    With ref_last the first item is played once more at the very end (the clock reference, no gap after it).
    Returns tracks, starts (one per item, plus one for the reference copy), total_samples.
    """
    g = int(gap * fs)
    pos = int(PRE * fs)
    seq = [(list(o), a) for o, a in zip(items, amps_db)]
    starts = []
    for k in range(len(seq)):
        starts.append(pos)
        pos += len(sweep) + g
    if ref_last:
        seq.append(seq[0])
        starts.append(pos)
        pos += len(sweep)
    total = pos + int(TAIL * fs)
    tracks = {}
    for (outs, adb), s0 in zip(seq, starts):
        for o in outs:
            t = tracks.setdefault(o, np.zeros(total, dtype=np.float32))
            t[s0:s0 + len(sweep)] = (sweep * 10 ** (adb / 20)).astype(np.float32)
    return tracks, starts, total


# ---------------------------------------------------------------------------------------------------------------
# windows and band levels
# ---------------------------------------------------------------------------------------------------------------
def _window(fs: int, W: float) -> np.ndarray:
    """3 ms before the peak (2 ms rise), then W seconds ending in a half-Hann fall over the last half."""
    a, b = int(0.003 * fs), int(W * fs)
    w = np.ones(a + b)
    rise = int(0.002 * fs)
    w[:rise] = 0.5 * (1 - np.cos(np.pi * np.arange(rise) / rise))
    h = b // 2
    w[a + b - h:] = 0.5 * (1 + np.cos(np.pi * np.arange(h) / h))
    return w


def _bands(P: np.ndarray, df: float, idx: np.ndarray | None = None) -> np.ndarray:
    """Mean power in each 1/12-octave band (dB). With `idx`, only those bands are computed (others NaN)."""
    f = np.arange(len(P)) * df
    lo, hi = dsp.CENTERS * 2 ** (-1 / 24), dsp.CENTERS * 2 ** (1 / 24)
    ilo, ihi = np.searchsorted(f, lo), np.searchsorted(f, hi)
    out = np.full(len(dsp.CENTERS), np.nan)
    for i in (range(len(out)) if idx is None else idx):
        out[i] = P[ilo[i]:ihi[i]].mean() if ihi[i] > ilo[i] else np.interp(dsp.CENTERS[i], f, P)
    return 10 * np.log10(np.maximum(out, 1e-30))


def _spec(x: np.ndarray, fs: int, nfft: int = 2 ** 17) -> np.ndarray:
    return _bands(np.abs(np.fft.rfft(x, nfft)) ** 2, fs / nfft)


def fdw_levels(ir: np.ndarray, start: int, fs: int, cycles: float, wmax: float, span: int = 0) -> np.ndarray:
    """Band levels (dB, 1/12 octave) of ir[start:] with a frequency-dependent window.

    Each band is analysed with a window of `cycles` periods of its own centre frequency (limited to FDW_TMIN..wmax).
    Low bands keep the long window they need for resolution; high bands see only the direct sound and the early
    reflections, so diffuse reverberation (which no EQ can fix) does not scramble them. cycles <= 0: one fixed
    window of wmax everywhere.

    span > 0 is for noise: the power of 50 %-overlapping windows that cover ir[start:start + span] is averaged (Welch).
    A single 15-cycle window holds about one independent frequency bin per band, which for noise means +-5.6 dB of
    scatter and a log-average 2.5 dB too low; averaging the many windows that fit into the span removes both.
    """
    c = dsp.CENTERS
    T = np.full(len(c), wmax) if cycles <= 0 else np.clip(cycles / c, FDW_TMIN, wmax)
    Tq = np.clip(2 ** (np.round(np.log2(T) * 4) / 4), FDW_TMIN, wmax)          # quarter-octave steps: few distinct windows
    out = np.empty(len(c))
    for t in np.unique(Tq):
        idx = np.flatnonzero(Tq == t)
        w = _window(fs, float(t))
        nfft = max(1 << 14, 1 << int(np.ceil(np.log2(len(w) * 8))))
        hop = max(len(w) // 2, 1)
        n = 1 if span <= len(w) else int((span - len(w)) // hop) + 1
        power = np.zeros(nfft // 2 + 1)
        for j in range(n):
            seg = np.zeros(len(w))
            part = ir[start + j * hop:start + j * hop + len(w)]
            seg[:len(part)] = part
            power += np.abs(np.fft.rfft(seg * w, nfft)) ** 2
        out[idx] = _bands(power / n, fs / nfft, idx)[idx]
    return out


def noise_windows(pk: int, gap: float, fs: int, wlen: int) -> list[int]:
    """Start indices of the late-IR windows used for the noise estimate (latest first, at most two).

    The deconvolved noise floor is only uniform up to about `gap` after the peak (beyond that fewer sweep samples
    contribute), so the windows sit at the end of the gap, as late as possible so the room has finished decaying.
    Two consecutive windows are used and the quieter one wins band by band: a burst (door, ventilation) can only hit one
    of them, and a decaying reverberant tail is always lower in the later one.
    """
    lo = pk + int(0.5 * fs)
    end = pk + int((gap - NOISE_GUARD) * fs)
    starts, s = [], end - wlen
    while s >= lo and len(starts) < 2:
        starts.append(s)
        s -= wlen
    return starts or [max(lo, end - wlen)]


def noise_levels(ir: np.ndarray, pk: int, gap: float, fs: int, wlen: int, cycles: float, wmax: float) -> tuple[np.ndarray, float]:
    """Band levels (dB) of the noise estimate, and its rms (for the 'is anything there' check)."""
    levels, rms = [], []
    for n0 in noise_windows(pk, gap, fs, wlen):
        if n0 + wlen <= len(ir):
            levels.append(fdw_levels(ir, n0, fs, cycles, wmax, span=wlen))
            rms.append(float(np.std(ir[n0:n0 + wlen])))
    if not levels:
        return np.full(len(dsp.CENTERS), -300.0), 0.0
    return np.min(levels, axis=0), float(min(rms))


# ---------------------------------------------------------------------------------------------------------------
# locating sweeps and measuring the clock ratio
# ---------------------------------------------------------------------------------------------------------------
def _locate(rec: np.ndarray, sweep: np.ndarray, lo: float, hi: float):
    """Matched-filter arrival (sub-sample start index) of `sweep` with its start in [lo, hi]. Returns (position, strength)."""
    lo = max(int(lo), 0)
    seg = rec[lo:int(hi) + len(sweep)]
    if len(seg) < len(sweep) + 600:
        return None, 0.0
    cc = signal.correlate(seg, sweep, mode="valid", method="fft")
    c = np.abs(cc)
    i = int(np.argmax(c))
    strength = float(c[i] / (np.median(c) + 1e-30))
    a, b = max(i - 256, 0), min(i + 257, len(cc))
    env = np.abs(signal.hilbert(cc[a:b]))                       # envelope of the matched-filter pulse: unbiased peak position
    j = int(np.argmax(env))
    frac = 0.0
    if 0 < j < len(env) - 1:
        y0, y1, y2 = env[j - 1], env[j], env[j + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return lo + a + j + float(np.clip(frac, -1, 1)), strength


def _resample(x: np.ndarray, num: int) -> np.ndarray:
    """Band-limited resampling of x to `num` samples (FFT method, padded to a fast length so it never crawls)."""
    n = len(x)
    if num == n:
        return x
    nf = next_fast_len(n)
    xp = np.zeros(nf)
    xp[:n] = x
    y = signal.resample(xp, int(round(nf * num / n)))
    out = np.zeros(num)
    out[:min(num, len(y))] = y[:num]
    return out


def _parabolic(y0: float, y1: float, y2: float) -> float:
    den = y0 - 2 * y1 + y2
    return float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0


def _deconv(seg: np.ndarray, sweep: np.ndarray, pre: int, amp: float) -> np.ndarray:
    nfft = 1 << int(np.ceil(np.log2(len(seg))))
    ref = np.zeros(nfft)
    ref[pre:pre + len(sweep)] = sweep
    X = np.fft.rfft(ref)
    P = np.abs(X) ** 2
    ir = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(X) / (P + 1e-6 * P.max()), nfft) / amp
    return np.roll(ir, pre)  # index `pre` == zero delay relative to channel 1


def _tail_decaying(ir: np.ndarray, pk: int, fs: int, gap: float) -> bool:
    """True if the room was still ringing at the end of the gap (energy still falling by > 3 dB per 0.12 s)."""
    w = int(0.12 * fs)
    end = pk + int((gap - NOISE_GUARD) * fs)
    if end > len(ir) or end - 3 * w < pk + int(0.3 * fs):
        return False
    sos = signal.butter(3, [500, 4000], "band", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, ir[pk:end + w // 2])
    e = [float(np.mean(x[end - pk - (j + 1) * w:end - pk - j * w] ** 2)) for j in (2, 1, 0)]
    return e[0] / max(e[2], 1e-30) > 2.0


# ---------------------------------------------------------------------------------------------------------------
# the analysis of one run
# ---------------------------------------------------------------------------------------------------------------
def analyze_run(rec: np.ndarray, fs: int, sweep: np.ndarray, starts: list[int], names: list[str],
                amps_db: list[float], gap: float, W: float, cal=None,
                f1: float = 30.0, f2: float = 20000.0, ref_last: bool = False, cycles: float = 15.0,
                search_s: float = 4.0) -> Run:
    """Deconvolve every channel of one run. See the module docstring for the clock correction (ref_last)."""
    rec = np.asarray(rec, dtype=np.float64)
    pre, g = int(SEG_PRE * fs), int(gap * fs)
    lseg = pre + len(sweep) + g
    run = Run()
    meta = run.meta = {"clock_ppm": None, "clock_ok": False, "noise_dbfs_a": None, "noise_dbfs_c": None,
                       "noise_dbfs_z": None, "tail_decaying": False}

    t1, strength = _locate(rec, sweep, 0, starts[0] + search_s * fs)
    if t1 is None or strength < DETECT:
        raise NotHeard("Did not hear the first speaker's sweep. Check the output routing, the DSP input and the mic position.")

    # --- clock ratio from the first sweep and its repeat at the end -------------------------------------------
    rho = 1.0                                            # output clock / input clock
    if ref_last and len(starts) == len(names) + 1:
        d_nom = starts[-1] - starts[0]
        t2, s2 = _locate(rec, sweep, t1 + d_nom - 0.08 * fs, t1 + d_nom + 0.08 * fs)
        if t2 is not None and s2 >= DETECT:
            r = d_nom / (t2 - t1)
            if abs(r - 1.0) <= CLOCK_MAX:
                rho = float(r)
                meta.update(clock_ppm=(r - 1.0) * 1e6, clock_ok=True)

    # --- room noise: the quiet lead-in before the first sweep -------------------------------------------------
    quiet = rec[int(0.15 * fs):int(t1) - int(0.1 * fs)]
    if len(quiet) >= int(0.4 * fs) and np.max(np.abs(quiet)) < 0.5:
        meta.update(noise_dbfs_a=rms_a_dbfs(quiet, fs), noise_dbfs_c=rms_c_dbfs(quiet, fs),
                    noise_dbfs_z=float(20 * np.log10(np.sqrt(np.mean(quiet ** 2)) + 1e-12)))

    win = _window(fs, W)
    a = int(0.003 * fs)
    cal_db = cal.at(dsp.CENTERS) if cal is not None else None
    for k, name in enumerate(names):
        anchor = t1 + (starts[k] - starts[0]) / rho          # expected arrival of this sweep, input-clock samples
        start_adc = anchor - pre / rho
        s_int = int(round(start_adc))
        e_adc = s_int - start_adc
        n_adc = int(round(lseg / rho))
        raw = np.zeros(n_adc)
        lo, hi = max(s_int, 0), min(s_int + n_adc, len(rec))
        if hi > lo:
            raw[lo - s_int:hi - s_int] = rec[lo:hi]
        seg = raw if n_adc == lseg else _resample(raw, lseg)

        ir = _deconv(seg, sweep, pre, 10 ** (amps_db[k] / 20))
        pk = int(np.argmax(np.abs(ir[:pre + int(0.1 * fs)])))
        nz, noise_rms = noise_levels(ir, pk, gap, fs, len(win), cycles, W)
        peak_raw = float(np.max(np.abs(raw))) if len(raw) else 0.0
        if pk < a or pk + len(win) - a > len(ir) or abs(ir[pk]) < 8 * noise_rms:   # the "peak" is only noise: not heard
            run[name] = _unheard()
            continue
        lv = fdw_levels(ir, pk - a, fs, cycles, W)
        if cal_db is not None:
            lv, nz = lv - cal_db, nz - cal_db
        pk_sub = pk + _parabolic(abs(ir[pk - 1]), abs(ir[pk]), abs(ir[pk + 1]))
        decaying = _tail_decaying(ir, pk, fs, gap)
        meta["tail_decaying"] = meta["tail_decaying"] or decaying
        run[name] = {
            "levels": lv, "noise": nz,
            "arrival_ms": (pk_sub - pre + e_adc * rho) / fs * 1000.0,
            "rms_c_dbfs": rms_c_dbfs(seg[pre:pre + len(sweep)], fs),
            "heard": True, "dist": distortion(ir, pk, fs, len(sweep) / fs, f1, f2, gap),
            "decay": decay_metrics(ir, pk, fs, gap), "sign": 1 if ir[pk] >= 0 else -1,
            "ir": ir[:pre + int(KEEP * fs)].astype(np.float32), "peak": pk,
            "clipped": peak_raw > 0.98, "rec_peak_dbfs": float(20 * np.log10(peak_raw + 1e-12)) if peak_raw else -120.0,
        }
    return run


def _unheard() -> dict:
    floor = np.full(len(dsp.CENTERS), -300.0)
    return {"levels": floor, "noise": floor.copy(), "arrival_ms": 0.0, "sign": 1, "ir": np.zeros(1, dtype=np.float32), "peak": 0,
            "clipped": False, "rec_peak_dbfs": -120.0, "rms_c_dbfs": -120.0, "dist": None,
            "decay": {"rt60": {}, "c50": None}, "heard": False}


def distortion(ir: np.ndarray, pk: int, fs: int, dur: float, f1: float, f2: float, gap: float = 2.5) -> dict:
    """Harmonic distortion vs frequency from an exponential sweep (Farina): the k-th harmonic's impulse response sits
    dur*ln(k)/ln(f2/f1) seconds BEFORE the main peak (wrapped to the end of the circular IR). Returns % at the mic,
    for 2nd-5th harmonics, ignoring harmonics that are not clearly above the noise.
    """
    n, r = len(ir), np.log(f2 / f1)
    a = int(0.003 * fs)
    w = _window(fs, 0.1)
    w4 = _window(fs, 0.4)
    h1 = smooth_levels(_spec(ir[pk - a:pk - a + len(w4)] * w4, fs))
    n0 = noise_windows(pk, gap, fs, len(w))[0]
    noise = smooth_levels(_spec(ir[n0:n0 + len(w)] * w, fs)) if n0 + len(w) <= n else np.full(len(dsp.CENTERS), -300.0)
    logf = np.log(dsp.CENTERS)
    power = np.zeros(len(dsp.CENTERS))
    for k in (2, 3, 4, 5):
        start = pk - int(dur * np.log(k) / r * fs) - a
        hk = smooth_levels(_spec(ir[(start + np.arange(len(w))) % n] * w, fs))
        fk = dsp.CENTERS * k
        ok = (fk < 18000) & (dsp.CENTERS > f1 * 1.5)
        hk_at, nz_at = np.interp(np.log(fk), logf, hk), np.interp(np.log(fk), logf, noise)
        ratio = 10 ** ((hk_at - h1) / 10)
        power += np.where(ok & (hk_at > nz_at + 6), ratio, 0.0)
    pct = 100 * np.sqrt(power)
    mid = (dsp.CENTERS >= 100) & (dsp.CENTERS <= 4000)
    i = int(np.argmax(np.where(mid, pct, 0)))
    return {"pct": pct, "mid_pct": float(np.median(pct[mid])), "max_pct": float(pct[i]), "max_hz": float(dsp.CENTERS[i]),
            "frac_high": float(np.mean(pct[mid] > 3.0))}   # share of 100 Hz-4 kHz bands above 3 %: isolated spikes are noise


def smooth_levels(db: np.ndarray) -> np.ndarray:
    return dsp.smooth_db(db, 1 / 3)


def align_sub(runs: list[dict], mains: list[str], sub: str, fc: float, fs: int) -> dict:
    """Best extra delay / polarity for the sub around the crossover, from the complex IRs of all positions.

    delay > 0: delay the SUB by that much; delay < 0: delay the MAINS by |delay|.
    """
    nfft = 1 << 16
    f = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (f >= 0.6 * fc) & (f <= 1.6 * fc)
    win = _window(fs, 0.4)
    a = int(0.003 * fs)

    def spec(r):
        x = np.zeros(len(r["ir"]))
        seg = r["ir"][r["peak"] - a:r["peak"] - a + len(win)]
        x[r["peak"] - a:r["peak"] - a + len(seg)] = seg * win[:len(seg)]
        return np.fft.rfft(x, nfft)[sel]

    delays = np.arange(-10e-3, 10e-3 + 1e-9, 0.1e-3)
    fsel = f[sel]
    ph = np.exp(-2j * np.pi * np.outer(delays, fsel))  # (nd, nf)
    score = np.zeros((2, len(delays)))
    for run in runs:
        hs = spec(run[sub])
        for m in mains:
            hm = spec(run[m])
            for pi, pol in enumerate((1, -1)):
                tot = hm[None, :] + pol * hs[None, :] * ph
                score[pi] += np.mean(10 * np.log10(np.abs(tot) ** 2 + 1e-30), axis=1)
    score /= len(runs) * len(mains)
    base = score[0, int(np.argmin(np.abs(delays)))]
    pi, di = np.unravel_index(int(np.argmax(score)), score.shape)
    return {"delay_ms": float(delays[di] * 1000), "invert": bool(pi == 1), "gain_db": float(score[pi, di] - base),
            "region_hz": [float(0.6 * fc), float(1.6 * fc)]}


def polarity_corr(a: dict, b: dict, fs: int) -> float:
    """Normalised correlation of two channels' direct sound (200 Hz-4 kHz, aligned at their own peaks).

    ~+1: same polarity, ~-1: one is inverted, near 0: unclear (very different speakers/paths).
    """
    sos = signal.butter(2, [200, 4000], "band", fs=fs, output="sos")

    def seg(r):
        x = signal.sosfilt(sos, r["ir"].astype(float))
        return x[r["peak"] - int(0.001 * fs): r["peak"] + int(0.008 * fs)]

    xa, xb = seg(a), seg(b)
    return float(np.dot(xa, xb) / (np.linalg.norm(xa) * np.linalg.norm(xb) + 1e-20))


OCT_BANDS = [125, 250, 500, 1000, 2000, 4000]


def _edc_db(e: np.ndarray, nf: float, fs: int):
    """Noise-corrected Schroeder decay curve in dB (truncated where the decay meets the noise), or None."""
    e = np.maximum(e - nf, 1e-30)
    k = max(int(0.01 * fs), 1)
    sm = np.convolve(e, np.ones(k) / k, mode="same")
    below = np.nonzero(sm < 3 * nf)[0]
    cut = int(below[0]) if len(below) else len(e)
    if cut < int(0.05 * fs):
        return None
    edc = np.cumsum(e[:cut][::-1])[::-1]
    return 10 * np.log10(edc / edc[0] + 1e-30), cut


def decay_metrics(ir: np.ndarray, pk: int, fs: int, gap: float) -> dict:
    """RT60 (from T20) per octave band and C50 (500 Hz-4 kHz) from a full-length impulse response.

    The noise level comes from the last 0.3 s of the gap; if the room is still ringing there the RT60 is
    reported as None (gap too short for this room).
    """
    end = pk + int((gap - 0.1) * fs)
    t0, t1 = pk + int((gap - 0.4) * fs), pk + int((gap - 0.1) * fs)
    out = {"rt60": {}, "c50": None}
    for fc in OCT_BANDS:
        sos = signal.butter(3, [fc / 1.414, fc * 1.414], "band", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, ir[:end + int(0.1 * fs)])
        res = _edc_db((x[pk:end]) ** 2, float(np.mean(x[t0:t1] ** 2)), fs)
        rt = None
        if res:
            db, _ = res
            i5, i25 = np.argmax(db <= -5), np.argmax(db <= -25)
            if db.min() <= -25 and i25 > i5:
                rt = float(3 * (i25 - i5) / fs)
        out["rt60"][fc] = rt
    sos = signal.butter(3, [500, 4000], "band", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, ir[:end + int(0.1 * fs)])
    res = _edc_db(x[pk - int(0.001 * fs):end] ** 2, float(np.mean(x[t0:t1] ** 2)), fs)
    if res:
        _, cut = res
        e = x[pk - int(0.001 * fs):end] ** 2
        n50 = int(0.051 * fs)
        out["c50"] = float(10 * np.log10(np.sum(e[:n50]) / max(np.sum(e[n50:cut]), 1e-30)))
    return out
