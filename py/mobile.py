"""Room EQ for the phone: the glue between the page (JavaScript) and the same dsp/sweep code the PC tool uses.

Runs inside Pyodide. Layouts: mono (M) and stereo (L, R). The phone plays the sweeps over AirPlay/Bluetooth/its speaker
and records the UMIK-1 at the same time; playback and recording are not locked together, so the first sweep is found
by matched filtering and all later timing is relative to it (L vs R delay is meaningful, absolute latency is not).
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from scipy import signal

from roomeq import calfile, dsp, sweep

FS = 48000
F_START = 30.0
SWEEP_S, GAP_S = 4.0, 2.0
IR_WINDOW, FDW_CYCLES = 0.4, 15.0
MIN_SNR_DB = 10.0
SEARCH_S = 9.0          # the first sweep may arrive this long after the start of the recording (AirPlay buffers ~2-3 s)
MAX_LEVEL_DB = -12.0

S: dict = {"cal": None, "layout": "2", "positions": [], "verify": [], "fit": None}


def _names():
    return ["M"] if S["layout"] == "1" else ["L", "R"]


def set_cal(text: str, name: str = "cal.txt"):
    path = "/tmp/" + (name if name.endswith(".txt") else "cal.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    S["cal"] = calfile.load(path)
    return {"label": S["cal"].label, "serial": S["cal"].serial, "sens": S["cal"].sens_factor, "deg": S["cal"].orientation_deg}


def reset(layout: str = "2"):
    S.update(layout=layout, positions=[], verify=[], fit=None)


def new_session(layout: str, level_db: float):
    reset(layout)
    S["level_db"] = min(float(level_db), MAX_LEVEL_DB)


def set_search(seconds: float):
    S["search_s"] = float(seconds)
    return {"ok": True}


def play_plan(level_db: float, short: bool = False):
    """The audio to play: one array per output channel, plus where the sweeps start (for the analysis)."""
    names = _names()
    n_sw, gap = (2.0, 1.0) if short else (SWEEP_S, GAP_S)
    lvl = min(float(level_db), MAX_LEVEL_DB)
    sw = sweep.make_sweep(FS, n_sw, F_START, 20000.0)
    items = [[i] for i in range(len(names))]
    ref = len(names) >= 2 and not short
    tracks, starts, total = sweep.build_tracks(sw, items, [lvl] * len(names), gap, FS, ref_last=ref)
    S["plan"] = dict(sw=sw, starts=starts, gap=gap, amps=[lvl] * len(names), ref=ref, n_sw=n_sw)
    out = np.stack([tracks[i] for i in range(len(names))])
    return {"channels": out.astype(np.float32), "fs": FS, "seconds": total / FS}


def _to_fs(rec: np.ndarray, fs_in: float) -> np.ndarray:
    rec = np.asarray(rec, dtype=np.float64)
    if abs(fs_in - FS) < 1:
        return rec
    from math import gcd
    a, b = int(round(FS)), int(round(fs_in))
    g = gcd(a, b)
    return signal.resample_poly(rec, a // g, b // g)


def _analyze(rec, fs_in):
    p = S["plan"]
    rec = _to_fs(rec, fs_in)
    return sweep.analyze_run(rec, FS, p["sw"], p["starts"], _names(), p["amps"], p["gap"], IR_WINDOW, S["cal"], F_START, 20000.0,
                             ref_last=p["ref"], cycles=FDW_CYCLES, search_s=S.get("search_s", SEARCH_S))


def _snr(r):
    band = (dsp.CENTERS >= 250) & (dsp.CENTERS <= 4000)
    return float(np.median((r["levels"] - r["noise"])[band]))


def check_level(rec, fs_in: float):
    rec = rec_array(rec)
    """Short sweep per speaker: is it heard, how far over the noise, clipped or distorted."""
    try:
        run = _analyze(rec, fs_in)
    except sweep.NotHeard as e:
        return {"ok": False, "error": str(e), "channels": []}
    chans, ok = [], True
    for name in _names():
        r = run[name]
        if not r["heard"]:
            chans.append({"name": name, "status": "bad", "text": "not heard - check the route to this speaker, or raise the volume"})
            ok = False
            continue
        snr = _snr(r)
        if r["clipped"]:
            chans.append({"name": name, "status": "bad", "snr": snr, "text": "microphone input clipped - lower the volume"})
            ok = False
        elif snr < 25:
            chans.append({"name": name, "status": "warn" if snr >= 15 else "bad", "snr": snr,
                          "text": f"{snr:.0f} dB over the noise - raise the volume (aim for 40 dB or more)"})
            ok = ok and snr >= 15
        else:
            d = r.get("dist") or {}
            chans.append({"name": name, "status": "ok", "snr": snr, "text": f"{snr:.0f} dB over the noise - good"})
    return {"ok": ok, "channels": chans, "noise_dba": run.meta.get("noise_dbfs_a")}


def _store(run):
    out = {}
    for n in _names():
        r = run[n]
        out[n] = {k: r[k] for k in ("levels", "noise", "arrival_ms", "heard", "clipped", "sign")}
    return {"ch": out, "meta": run.meta}


def measure(rec, fs_in: float, verify: bool = False):
    rec = rec_array(rec)
    try:
        run = _analyze(rec, fs_in)
    except sweep.NotHeard as e:
        return {"ok": False, "error": str(e)}
    if not all(run[n]["heard"] for n in _names()):
        miss = [n for n in _names() if not run[n]["heard"]]
        return {"ok": False, "error": f"Did not hear {', '.join(miss)}. Check the route and the volume, then repeat."}
    warn = [f"{n}: the microphone input clipped - lower the volume and repeat." for n in _names() if run[n]["clipped"]]
    if len(_names()) > 1 and not run.meta["clock_ok"]:
        warn.append("The clock check failed, so the L/R delay may be off by up to about 1 ms. Repeat if the delay matters.")
    if run.meta["tail_decaying"]:
        warn.append("The room is still ringing at the end of each gap (long reverberation); results are less certain.")
    (S["verify"] if verify else S["positions"]).append(_store(run))
    return {"ok": True, "n": len(S["verify"] if verify else S["positions"]), "warnings": warn,
            "snr": {n: _snr(run[n]) for n in _names()}}


def undo(verify: bool = False):
    lst = S["verify"] if verify else S["positions"]
    if lst:
        lst.pop()
    return len(lst)


def _avg(positions, ch):
    avg = dsp.power_mean_db([p["ch"][ch]["levels"] for p in positions])
    noise = dsp.power_mean_db([p["ch"][ch]["noise"] for p in positions])
    return avg, (avg - noise) >= MIN_SNR_DB


def _fl(x):
    return [round(float(v), 2) for v in np.asarray(x)]


def analyze():
    pos = S["positions"]
    if not pos:
        return {"ok": False, "error": "No positions measured yet."}
    names = _names()
    fr = dsp.CENTERS
    s = dsp.FitSettings(tilt=0.0)
    avgs = {c: _avg(pos, c) for c in names}
    weights = {c: dsp.consistency([p["ch"][c]["levels"] for p in pos], fr, s, avgs[c][1])[1] for c in names}
    sigma = {c: dsp.consistency([p["ch"][c]["levels"] for p in pos], fr, s, avgs[c][1])[0] for c in names}
    fit = {}
    # the channels share their filters above 300 Hz (stereo pair); the room modes below are fitted per channel
    joint = dsp.power_mean_db([avgs[c][0] for c in names])
    jvalid = np.logical_and.reduce([avgs[c][1] for c in names])
    jw = np.mean([weights[c] for c in names], axis=0)
    nhi = max(1, s.max_filters - s.low_filters)
    shi = replace(s, fmin=max(s.fmin, s.q_split), max_filters=nhi)
    hi = dsp.fit_peq(fr, joint, jvalid, shi, jw)
    for c in names:
        avg, valid = avgs[c]
        slo = replace(s, fmax=s.q_split, max_filters=s.low_filters)
        lo = dsp.fit_peq(fr, avg + hi["correction"], valid, slo, weights[c])
        flt = sorted(lo["filters"] + [dict(f) for f in hi["filters"]], key=lambda f: f["freq"])
        fit[c] = dsp.evaluate(fr, avg, valid, s, flt)
    notes = []
    out_ch = {}
    for c in names:
        avg, valid = avgs[c]
        bad = (~valid) & (fr >= s.fmin) & (fr <= s.fmax)
        if bad.any():
            notes.append(f"{c}: too little signal over the noise between {fr[bad].min():.0f}-{fr[bad].max():.0f} Hz; those bands were ignored. "
                         "Turn the volume up for a better result.")
        per = dsp.evaluate_positions(fr, [p["ch"][c]["levels"] for p in pos], valid, s, fit[c]["filters"])
        for i, q in enumerate(per):
            if q["rms_after"] > q["rms_before"] + 0.3:
                notes.append(f"{c}: position {i + 1} gets slightly worse ({q['rms_before']:.1f} to {q['rms_after']:.1f} dB rms); "
                             "the correction is a compromise for the whole listening area.")
        f = fit[c]
        out_ch[c] = {"filters": [{"freq": round(x["freq"], 1), "q": round(x["q"], 2), "gain": round(x["gain"], 1)} for x in f["filters"]],
                     "rms_before": f["rms_before"], "rms_after": f["rms_after"], "max_before": f["max_before"], "max_after": f["max_after"],
                     "trim_db": f["trim_db"], "positions": per,
                     "plot": {"f": _fl(fr), "meas": _fl(f["smoothed"]), "pred": _fl(f["predicted"]), "target": _fl(f["target"])}}
    # level and delay between the channels, from the position average
    extra = {}
    if len(names) == 2:
        a, b = names
        ref = (fr >= 250) & (fr <= 4000) & avgs[a][1] & avgs[b][1]
        lvl = float(np.median(avgs[a][0][ref] - avgs[b][0][ref])) if ref.sum() >= 4 else 0.0
        arr = [p["ch"][b]["arrival_ms"] - p["ch"][a]["arrival_ms"] for p in pos]
        d = float(np.median(arr))
        spread = float(max(arr) - min(arr))
        # delay is added to the earlier speaker; level trim lowers the louder one
        extra = {"delay_ms": abs(d), "delay_on": a if d > 0 else b, "level_db": abs(lvl), "level_on": a if lvl > 0 else b,
                 "delay_spread_ms": spread, "ok": all(p["meta"]["clock_ok"] for p in pos)}
        if not extra["ok"]:
            notes.append("The clock check failed on at least one position, so the L/R delay is uncertain. Repeat the measurement.")
    S["fit"] = {c: out_ch[c]["filters"] for c in names}
    S["fit_full"] = {c: fit[c] for c in names}
    S["settings"] = s
    S["avgs"] = avgs
    return {"ok": True, "channels": out_ch, "extra": extra, "notes": notes, "order": names}


def verify_score():
    ver = S["verify"]
    if not ver or not S.get("fit"):
        return {"ok": False, "error": "Measure again after entering the settings, then score."}
    fr = dsp.CENTERS
    s = S["settings"]
    res, allok = {}, True
    for c in _names():
        avg, valid = _avg(ver, c)
        v = dsp.verify(fr, avg, valid, s)
        before = S["fit_full"][c]["rms_before"]
        ok = v["rms"] <= 2.0
        allok = allok and ok
        res[c] = {"rms": v["rms"], "max": v["max"], "rms_before": before, "ok": ok,
                  "plot": {"f": _fl(fr), "meas": _fl(v["smoothed"]), "target": _fl(dsp.target_curve(fr, s.tilt))}}
    return {"ok": True, "pass": allok, "channels": res}


# ---------------------------------------------------------------------------------------------------------------
# bridge to JavaScript: every call returns a JSON string (numpy numbers and NaN made safe); audio goes as typed arrays
# ---------------------------------------------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def call(fn: str, *args):
    try:
        r = globals()[fn](*args)
    except Exception as e:                      # shown on the page
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return json.dumps(_clean(r))


def plan_audio(level_db, short=False):
    """Interleaved-by-channel float32 audio (all of channel 0, then channel 1...) and its layout, for Web Audio."""
    from pyodide.ffi import to_js
    p = play_plan(level_db, bool(short))
    return to_js(np.ascontiguousarray(p["channels"], dtype=np.float32).ravel()), len(_names()), p["seconds"]


def rec_array(x):
    return np.asarray(x.to_py() if hasattr(x, "to_py") else x, dtype=np.float32)
