import io
import json
import random
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Pitch Glide / Pitch Change Detection Threshold Test (Single-interval)
# - Single stimulus per trial: FLAT vs GLIDE
# - Respond: "変化あり（GLIDE）" / "変化なし（FLAT）"
# - Mix in FLAT trials to avoid expectancy and to estimate false alarms (FA)
# - Staircase on GLIDE duration D (ms), updated on GLIDE trials only
# - 2-down 1-up (signal-only): 2 consecutive HITs -> harder (D↓), MISS -> easier (D↑)
# - Big step until N reversals, then small step
# - Threshold = median of last 6 reversals in small-step phase (mean optional)
#
# Spec alignment (Click Fusion / FM と統一):
# - Sampling rate fixed: 48,000 Hz
# - Test order fixed: Series 1 / Series 2 (default: Series 1), 100 trials (40 FLAT / 60 GLIDE)
# - Early stop:
#     * Ceiling stop: D_max で 2回連続MISS（GLIDE試行）
#     * Floor stop:   D_min で 4回連続HIT（GLIDE試行）
#     * Reversal stop: small-step reversals が 6個集まれば終了
# - Practice:
#     * 50/50 random FLAT/GLIDE
#     * Easy GLIDE duration = D_max
#     * GLIDE試行のみの連続HITをカウントし、5連続HITで終了
# - Progress display: small reversals “x/6”
# - CSV export
#
# Final micro-fixes:
# - steady_ms=0 のとき、steady部が 1サンプル勝手に付く問題を修正（n_steady を 0 許容）
# - 開始時に設定の整合性チェック（明らかな不整合は開始できない）
# ============================================================

# -------------------------
# Fixed constants
# -------------------------
SR_FIXED = 48_000
N_TEST_TRIALS = 100
N_SMALL_REV_TARGET = 6  # threshold needs last 6 small-phase reversals

DEMO_RAMP_MS = 300  # demo duration (ms) for the instruction buttons

FINISH_REASON_LABELS = {
    "small_reversals": f"small reversals {N_SMALL_REV_TARGET}個（閾値算出条件）",
    "ceiling_miss": "D_maxで2回連続MISS",
    "floor_hit": "D_minで4回連続HIT",
    "n_trials": "n_trials到達",
    "manual": "手動終了",
    "internal_error": "内部エラー",
}

# -------------------------
# Fixed test series (1=FLAT, 2=GLIDE) — length 100
# -------------------------
SERIES_1 = [
    2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1,
    2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2,
    1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2,
    2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2,
    1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2,
]

SERIES_2 = [
    2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1,
    2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2,
    1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2,
    2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2,
    2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2,
]

FIXED_TRIAL_SERIES = {
    "系列1": SERIES_1,
    "系列2": SERIES_2,
}

# -------------------------
# Fixed glide-direction schedules for test (GLIDE trials only; length 60)
# 1 = up, 2 = down  (as requested)
# -------------------------
GLIDE_DIR_SERIES_1_CODES = [
    2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
    1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2,
    1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1,
]

GLIDE_DIR_SERIES_2_CODES = [
    1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2,
    1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1,
    2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1,
]

FIXED_GLIDE_DIR_CODES = {
    "系列1": GLIDE_DIR_SERIES_1_CODES,
    "系列2": GLIDE_DIR_SERIES_2_CODES,
}


def _trial_codes_to_types(seq: List[int]) -> List[str]:
    return ["flat" if int(v) == 1 else "glide" for v in seq]


def _dir_codes_to_labels(seq: List[int]) -> List[str]:
    # 1=up, 2=down
    return ["up" if int(v) == 1 else "down" for v in seq]


def _build_constrained_sequence(
    *,
    n1: int,
    n2: int,
    max_consecutive: int,
    rng: random.Random,
    max_restarts: int = 10_000,
) -> List[int]:
    """Build a length-(n1+n2) sequence with exact counts and a max-run constraint."""
    total = int(n1) + int(n2)
    if total <= 0:
        return []

    for _ in range(int(max_restarts)):
        rem = {1: int(n1), 2: int(n2)}
        out: List[int] = []
        while len(out) < total:
            # last run length
            last = out[-1] if out else None
            run = 1
            if last is not None:
                run = 1
                for j in range(len(out) - 2, -1, -1):
                    if out[j] == last:
                        run += 1
                    else:
                        break

            candidates: List[int] = []
            weights: List[float] = []
            for v in (1, 2):
                if rem[v] <= 0:
                    continue
                if last == v and run >= int(max_consecutive):
                    continue
                candidates.append(v)
                weights.append(float(rem[v]))  # bias toward remaining count

            if not candidates:
                break  # restart

            if len(candidates) == 1:
                choice = candidates[0]
            else:
                choice = rng.choices(candidates, weights=weights, k=1)[0]
            out.append(choice)
            rem[choice] -= 1

        if len(out) == total:
            return out

    raise RuntimeError("擬似ランダム系列の生成に失敗しました（制約が厳しすぎる可能性があります）。")


def build_test_plan(
    *,
    series_name: str,
    pseudo_seed: Optional[int] = None,
    max_consecutive: int = 3,
) -> Dict[str, Any]:
    """Build a frozen test plan: trial schedule (100) + glide directions (60).

    Returns
    -------
    dict with keys:
      - series_name
      - schedule_codes: List[int] (len=100; 1=FLAT, 2=GLIDE)
      - schedule_types: List[str] (len=100; "flat"/"glide")
      - glide_dir_labels: List[str] (len=60; "up"/"down")
      - glide_dir_codes: List[int] (len=60; 1=up, 2=down)
      - pseudo_seed: Optional[int]
      - max_consecutive: int
    """
    series_name = str(series_name)

    if series_name in FIXED_TRIAL_SERIES:
        schedule_codes = list(FIXED_TRIAL_SERIES[series_name])
        glide_dir_codes = list(FIXED_GLIDE_DIR_CODES[series_name])
        seed_used: Optional[int] = None
    else:
        # pseudo-random (exact counts, max-run constraint)
        seed_used = int(pseudo_seed) if pseudo_seed is not None else int(time.time())
        rng = random.Random(seed_used)
        schedule_codes = _build_constrained_sequence(
            n1=40,
            n2=60,
            max_consecutive=int(max_consecutive),
            rng=rng,
        )

        # Also fix glide direction for reproducibility (30 up / 30 down; max-run constraint)
        rng_dir = random.Random(seed_used + 1)
        glide_dir_codes = _build_constrained_sequence(
            n1=30,
            n2=30,
            max_consecutive=int(max_consecutive),
            rng=rng_dir,
        )

    # Sanity checks
    if len(schedule_codes) != N_TEST_TRIALS:
        raise RuntimeError(f"trial schedule length mismatch: {len(schedule_codes)}")
    if schedule_codes.count(1) != 40 or schedule_codes.count(2) != 60:
        raise RuntimeError("trial schedule counts mismatch (need 40 FLAT / 60 GLIDE).")
    if len(glide_dir_codes) != 60:
        raise RuntimeError(f"glide direction schedule length mismatch: {len(glide_dir_codes)}")
    if glide_dir_codes.count(1) != 30 or glide_dir_codes.count(2) != 30:
        raise RuntimeError("glide direction counts mismatch (need 30 up / 30 down).")

    return {
        "series_name": series_name,
        "schedule_codes": schedule_codes,
        "schedule_types": _trial_codes_to_types(schedule_codes),
        "glide_dir_codes": glide_dir_codes,
        "glide_dir_labels": _dir_codes_to_labels(glide_dir_codes),
        "pseudo_seed": seed_used,
        "max_consecutive": int(max_consecutive),
    }


# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Pitch Glide Detection Threshold Test",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 Pitch Glide Detection Threshold Test")

st.markdown(
    """
**目的**  
単発刺激で「**高さが平坦（FLAT）**」か「**高さが変化（GLIDE）**」かを答えてもらい、  
**ピッチ変化を検出できる最小のグライド長（duration, ms）**を推定します。

**設計の意図（患者運用を想定）**  
- 2区間比較（ABの2AFC）を避け、**単発**で回答できる形式  
- 「変化なし（FLAT）」を混ぜて、**“常に変化あり”戦略**を防止  
- 閾値推定（staircase）は **GLIDE試行のみ**で更新し、FLATは **false alarm** 推定に使います

**注意**  
- なるべく **有線ヘッドホン**（Bluetoothは遅延や途切れの原因になり得ます）  
- 音量は事前に快適レベルに調整  
- 原則 **replayしない**運用（提示は1回を想定）
"""
)

# ============================================================
# Presets (f_center, default delta)
# ============================================================
PRESETS = {
    "1240 Hz版（F2帯寄り：900–1580 Hz）": {"f_center": 1240.0, "delta_default": 340.0},
    "500 Hz版（低周波：350–650 Hz）": {"f_center": 500.0, "delta_default": 150.0},
}

# ============================================================
# Audio helpers
# ============================================================
def _cosine_ramp_env(n: int, sr: int, edge_ms: int) -> np.ndarray:
    ramp_n = int(round(sr * edge_ms / 1000))
    ramp_n = max(0, min(ramp_n, n // 2))
    env = np.ones(n, dtype=np.float32)
    if ramp_n > 0:
        t = np.arange(ramp_n, dtype=np.float32) / float(ramp_n)
        ramp = 0.5 - 0.5 * np.cos(np.pi * t)  # 0->1
        env[:ramp_n] = ramp
        env[-ramp_n:] = ramp[::-1]
    return env


def rms_normalize(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """RMS normalization on the actual sound interval (no silence included)."""
    x = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    return x * (float(target_rms) / rms)


def glide_stimulus_linear_ramp_to_center(
    *,
    sr: int,
    f_center: float,
    delta: float,
    ramp_ms: int,
    steady_ms: int,
    direction: str,  # "up" or "down"（本番は系列で固定／練習はランダム）
    edge_ramp_ms: int,
    target_rms: float,
) -> np.ndarray:
    """
    One-interval GLIDE stimulus (monotonic linear ramp + optional steady):

      - Linear frequency ramp lasting ramp_ms:
          up:   (f_center - delta) -> f_center
          down: (f_center + delta) -> f_center

      - Followed by steady tone at f_center lasting steady_ms

    Notes
    -----
    - Phase continuity is preserved by integrating instantaneous frequency for the ramp,
      then continuing the steady segment from the ramp's end phase.

    Micro-fix
    ---------
    - steady_ms=0 のとき、steady が 1サンプル付与されないよう、n_steady を 0 許容にしています。
    """
    ramp_ms = int(ramp_ms)
    steady_ms = int(steady_ms)

    # --- Frequency ramp (linear) ---
    n_ramp = max(2, int(round(sr * ramp_ms / 1000)))
    if direction == "down":
        f_start = float(f_center + delta)
    else:
        f_start = float(f_center - delta)
    f_end = float(f_center)

    f_inst = np.linspace(f_start, f_end, n_ramp, endpoint=True, dtype=np.float32)

    dphi = (2.0 * np.pi * f_inst / float(sr)).astype(np.float32)
    phase = np.concatenate(([0.0], np.cumsum(dphi)[:-1])).astype(np.float32)
    x_ramp = np.sin(phase).astype(np.float32)

    # --- Steady at f_center (continue from ramp end phase) ---
    n_steady = max(0, int(round(sr * steady_ms / 1000)))  # allow 0
    if n_steady > 0:
        t2 = np.arange(n_steady, dtype=np.float32) / float(sr)
        phase0 = float(phase[-1] + dphi[-1])  # phase at the *next* sample after the ramp
        x_steady = np.sin(phase0 + 2.0 * np.pi * f_end * t2).astype(np.float32)
        x = np.concatenate([x_ramp, x_steady]).astype(np.float32)
    else:
        x = x_ramp

    # Apply onset/offset cosine ramp (amplitude envelope) to the whole interval
    x *= _cosine_ramp_env(len(x), sr, edge_ramp_ms)

    # RMS normalize after applying the envelope
    x = rms_normalize(x, target_rms=target_rms)

    # Avoid clipping
    peak = float(np.max(np.abs(x)))
    if peak > 0.99:
        x *= 0.99 / peak

    return x


def flat_stimulus(
    *,
    sr: int,
    f_center: float,
    total_ms: int,
    edge_ramp_ms: int,
    target_rms: float,
) -> np.ndarray:
    """One-interval FLAT stimulus: steady tone only (same total duration as GLIDE interval)."""
    total_ms = int(total_ms)
    n = max(2, int(round(sr * total_ms / 1000)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    x = np.sin(2.0 * np.pi * f_center * t).astype(np.float32)
    x *= _cosine_ramp_env(len(x), sr, edge_ramp_ms)
    x = rms_normalize(x, target_rms=target_rms)
    peak = float(np.max(np.abs(x)))
    if peak > 0.99:
        x *= 0.99 / peak
    return x


def mono_to_stereo_bytes(x_mono: np.ndarray, sr: int, ear: str) -> bytes:
    x = np.clip(x_mono.astype(np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)

    if ear == "左耳のみ":
        left = pcm
        right = np.zeros_like(pcm)
    elif ear == "右耳のみ":
        left = np.zeros_like(pcm)
        right = pcm
    else:
        left = pcm
        right = pcm

    stereo = np.empty(2 * len(pcm), dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


def generate_trial_wav_single(
    *,
    sr: int,
    f_center: float,
    delta: float,
    ramp_ms: int,
    steady_ms: int,
    ear: str,
    edge_ramp_ms: int,
    target_rms: float,
    trial_type: str,  # "glide" or "flat"
    direction: str,   # "up" or "down" (used only if trial_type="glide")
) -> Tuple[bytes, int]:
    """Returns wav_bytes, total_ms (interval duration)."""
    total_ms = int(ramp_ms) + int(steady_ms)
    if trial_type == "glide":
        x = glide_stimulus_linear_ramp_to_center(
            sr=sr,
            f_center=f_center,
            delta=delta,
            ramp_ms=ramp_ms,
            steady_ms=steady_ms,
            direction=direction,
            edge_ramp_ms=edge_ramp_ms,
            target_rms=target_rms,
        )
    else:
        x = flat_stimulus(
            sr=sr,
            f_center=f_center,
            total_ms=total_ms,
            edge_ramp_ms=edge_ramp_ms,
            target_rms=target_rms,
        )
    return mono_to_stereo_bytes(x, sr, ear), total_ms


# ============================================================
# Staircase (duration ms) — updates on GLIDE trials only
# ============================================================
@dataclass
class DurationStaircase:
    start_ms: float
    floor_ms: float
    ceil_ms: float
    step_big_ms: float
    step_small_ms: float
    switch_after_reversals: int = 4  # after 4th reversal -> small step

    # internal
    x_ms: float = field(init=False)
    trial_index_updates: int = 0  # counts only GLIDE updates
    n_correct_streak: int = 0
    last_direction: Optional[str] = None  # "up" / "down"
    reversals: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.x_ms = float(self.start_ms)

    def current_step(self) -> float:
        return float(self.step_small_ms) if len(self.reversals) >= self.switch_after_reversals else float(self.step_big_ms)

    def phase(self) -> str:
        return "small" if len(self.reversals) >= self.switch_after_reversals else "big"

    def update_on_glide(self, hit: bool) -> Dict[str, Any]:
        """
        2-down 1-up (signal-only):
          - 2 consecutive HITs -> decrease duration (harder)
          - 1 MISS -> increase duration (easier)
        NOTE: called only on GLIDE trials.
        """
        self.trial_index_updates += 1
        prev_x = float(self.x_ms)
        phase_before = self.phase()
        step = float(self.current_step())

        direction: Optional[str] = None
        if hit:
            self.n_correct_streak += 1
            if self.n_correct_streak >= 2:
                direction = "down"  # duration decreases
                self.x_ms = max(float(self.floor_ms), self.x_ms - step)
                self.n_correct_streak = 0
        else:
            self.n_correct_streak = 0
            direction = "up"  # duration increases
            self.x_ms = min(float(self.ceil_ms), self.x_ms + step)

        reversal = False
        reversal_level = None
        if direction is not None and self.last_direction is not None and direction != self.last_direction:
            reversal = True
            reversal_level = prev_x
            self.reversals.append(
                {
                    "update_index": int(self.trial_index_updates),
                    "level_ms": float(reversal_level),
                    "phase": phase_before,
                    "step_ms": float(step),
                }
            )

        if direction is not None:
            self.last_direction = direction

        return {
            "prev_x_ms": prev_x,
            "new_x_ms": float(self.x_ms),
            "direction": direction,
            "step_used_ms": step,
            "phase": self.phase(),
            "phase_before": phase_before,
            "reversal": reversal,
            "reversal_phase": phase_before if reversal else None,
            "reversal_level_ms": reversal_level,
            "n_reversals": len(self.reversals),
            "n_updates": int(self.trial_index_updates),
        }

    def n_small_reversals(self) -> int:
        # Reversals in small-step phase = reversals after the first switch_after_reversals
        return max(0, len(self.reversals) - int(self.switch_after_reversals))

    def usable_reversal_levels(self) -> List[float]:
        if len(self.reversals) <= self.switch_after_reversals:
            return []
        return [float(r["level_ms"]) for r in self.reversals[self.switch_after_reversals :]]

    def threshold_last6_mean(self) -> Optional[float]:
        usable = self.usable_reversal_levels()
        if len(usable) < N_SMALL_REV_TARGET:
            return None
        return float(np.mean(usable[-N_SMALL_REV_TARGET:]))

    def threshold_last6_median(self) -> Optional[float]:
        usable = self.usable_reversal_levels()
        if len(usable) < N_SMALL_REV_TARGET:
            return None
        return float(np.median(usable[-N_SMALL_REV_TARGET:]))


def finish_reason_label(reason: str) -> str:
    return FINISH_REASON_LABELS.get(str(reason), str(reason))


def format_ms(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f} ms"


def format_ms_range(vmin: Optional[float], vmax: Optional[float]) -> str:
    if vmin is None or vmax is None or pd.isna(vmin) or pd.isna(vmax):
        return "—"
    return f"{float(vmin):.1f}–{float(vmax):.1f} ms"


def format_ms_list(values: List[float]) -> str:
    vals = [float(x) for x in values]
    if len(vals) == 0:
        return "—"
    return ", ".join([f"{x:.1f}" for x in vals])


def format_session_time(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))


def safe_filename_part(value: str) -> str:
    text_value = str(value).strip()
    if not text_value:
        return "no_subject"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text_value)
    safe = safe.strip("_")
    return safe or "no_subject"


def build_threshold_summary(sc: Optional[DurationStaircase]) -> Dict[str, Any]:
    usable = sc.usable_reversal_levels() if sc is not None else []
    usable = [float(x) for x in usable]

    summary: Dict[str, Any] = {
        "status": "none",
        "status_label": "参考値なし",
        "usable_count": int(len(usable)),
        "usable_levels": usable,
        "official_mean": None,
        "official_median": None,
        "official_levels": [],
        "reference_median": None,
        "reference_min": None,
        "reference_max": None,
    }

    if len(usable) >= int(N_SMALL_REV_TARGET):
        official_levels = usable[-int(N_SMALL_REV_TARGET):]
        summary.update(
            {
                "status": "official",
                "status_label": "正式閾値",
                "official_mean": float(np.mean(official_levels)),
                "official_median": float(np.median(official_levels)),
                "official_levels": official_levels,
            }
        )
    elif len(usable) >= 3:
        summary.update(
            {
                "status": "provisional",
                "status_label": "参考値（暫定中央値）",
                "reference_median": float(np.median(usable)),
            }
        )
    elif len(usable) == 2:
        summary.update(
            {
                "status": "reference_range",
                "status_label": "参考値（中央値＋範囲）",
                "reference_median": float(np.median(usable)),
                "reference_min": float(min(usable)),
                "reference_max": float(max(usable)),
            }
        )

    return summary


def build_glide_convergence_chart_df(dft: pd.DataFrame) -> pd.DataFrame:
    if dft is None or dft.empty or "trial_type" not in dft.columns:
        return pd.DataFrame()

    glide_df = dft[dft["trial_type"] == "glide"].copy()
    if glide_df.empty:
        return pd.DataFrame()

    if "n_updates_glide" in glide_df.columns and glide_df["n_updates_glide"].notna().all():
        glide_df["glide_count"] = glide_df["n_updates_glide"].astype(int)
    elif "glide_no_planned" in glide_df.columns and glide_df["glide_no_planned"].notna().all():
        glide_df["glide_count"] = glide_df["glide_no_planned"].astype(int)
    else:
        glide_df["glide_count"] = np.arange(1, len(glide_df) + 1)

    glide_df = glide_df.sort_values("glide_count").reset_index(drop=True)
    glide_df["D_ms_presented"] = pd.to_numeric(glide_df.get("D_ms_presented"), errors="coerce")
    glide_df["reversal_flag"] = glide_df["reversal"].fillna(0).astype(int) if "reversal" in glide_df.columns else 0
    glide_df["reversal_level_ms"] = (
        pd.to_numeric(glide_df["reversal_level_ms"], errors="coerce")
        if "reversal_level_ms" in glide_df.columns
        else np.nan
    )
    glide_df["phase_label"] = glide_df["phase"].fillna("—") if "phase" in glide_df.columns else "—"
    if "reversal_phase" in glide_df.columns:
        glide_df["reversal_phase_label"] = glide_df["reversal_phase"].fillna(glide_df["phase_label"])
    else:
        glide_df["reversal_phase_label"] = glide_df["phase_label"]
    glide_df["correct_label"] = np.where(glide_df["correct"].fillna(0).astype(int) == 1, "HIT", "MISS")

    return glide_df[
        [
            "glide_count",
            "D_ms_presented",
            "reversal_flag",
            "reversal_level_ms",
            "phase_label",
            "reversal_phase_label",
            "correct_label",
        ]
    ].copy()


def chart_records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    safe_df = df.copy().replace([np.inf, -np.inf], np.nan)
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def render_glide_convergence_chart(dft: pd.DataFrame, threshold_summary: Dict[str, Any]):
    chart_df = build_glide_convergence_chart_df(dft)
    if chart_df.empty:
        st.caption("GLIDE試行がないため、収束グラフは表示できません。")
        return

    line_df = chart_df[chart_df["D_ms_presented"].notna()].copy()
    if line_df.empty:
        st.caption("GLIDE試行の D 値が取得できないため、収束グラフは表示できません。")
        return

    layers: List[Dict[str, Any]] = [
        {
            "data": {"values": chart_records_json_safe(line_df)},
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": "glide_count", "type": "quantitative", "title": "変化あり（GLIDE）提示数"},
                "y": {"field": "D_ms_presented", "type": "quantitative", "title": "提示 D (ms)"},
                "tooltip": [
                    {"field": "glide_count", "type": "quantitative", "title": "GLIDE提示数"},
                    {"field": "D_ms_presented", "type": "quantitative", "title": "提示 D (ms)", "format": ".1f"},
                    {"field": "correct_label", "type": "nominal", "title": "反応"},
                    {"field": "phase_label", "type": "nominal", "title": "phase"},
                ],
            },
        }
    ]

    reversal_df = chart_df[(chart_df["reversal_flag"] == 1) & chart_df["reversal_level_ms"].notna()].copy()
    if not reversal_df.empty:
        big_reversal_df = reversal_df[reversal_df["reversal_phase_label"] != "small"].copy()
        small_reversal_df = reversal_df[reversal_df["reversal_phase_label"] == "small"].copy()

        if not big_reversal_df.empty:
            layers.append(
                {
                    "data": {"values": chart_records_json_safe(big_reversal_df)},
                    "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 120, "color": "#1f77b4"},
                    "encoding": {
                        "x": {"field": "glide_count", "type": "quantitative"},
                        "y": {"field": "reversal_level_ms", "type": "quantitative"},
                        "tooltip": [
                            {"field": "glide_count", "type": "quantitative", "title": "reversal時のGLIDE提示数"},
                            {"field": "reversal_level_ms", "type": "quantitative", "title": "reversal level (ms)", "format": ".1f"},
                            {"field": "reversal_phase_label", "type": "nominal", "title": "reversal phase"},
                        ],
                    },
                }
            )

        if not small_reversal_df.empty:
            layers.append(
                {
                    "data": {"values": chart_records_json_safe(small_reversal_df)},
                    "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 150, "color": "#d62728"},
                    "encoding": {
                        "x": {"field": "glide_count", "type": "quantitative"},
                        "y": {"field": "reversal_level_ms", "type": "quantitative"},
                        "tooltip": [
                            {"field": "glide_count", "type": "quantitative", "title": "reversal時のGLIDE提示数"},
                            {"field": "reversal_level_ms", "type": "quantitative", "title": "reversal level (ms)", "format": ".1f"},
                            {"field": "reversal_phase_label", "type": "nominal", "title": "reversal phase"},
                        ],
                    },
                }
            )

    ref_value: Optional[float] = None
    ref_label: Optional[str] = None
    if threshold_summary.get("status") == "official":
        ref_value = threshold_summary.get("official_median")
        ref_label = "正式閾値（中央値）"
    elif threshold_summary.get("reference_median") is not None:
        ref_value = threshold_summary.get("reference_median")
        ref_label = threshold_summary.get("status_label")

    if ref_value is not None and ref_label is not None:
        layers.append(
            {
                "data": {"values": [{"value_ms": float(ref_value), "label": str(ref_label)}]},
                "mark": {"type": "rule", "strokeDash": [6, 4]},
                "encoding": {
                    "y": {"field": "value_ms", "type": "quantitative"},
                    "tooltip": [
                        {"field": "label", "type": "nominal", "title": "指標"},
                        {"field": "value_ms", "type": "quantitative", "title": "値 (ms)", "format": ".1f"},
                    ],
                },
            }
        )

    chart_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 320,
        "layer": layers,
    }
    st.vega_lite_chart(spec=chart_spec, use_container_width=True)
    st.caption("横軸は総試行数ではなく、変化あり（GLIDE）刺激の提示数です。青ダイヤ = big-step reversal、赤ダイヤ = small-step reversal。")


def build_summary_text(
    *,
    dft: pd.DataFrame,
    sc: Optional[DurationStaircase],
    threshold_summary: Dict[str, Any],
    reason: str,
    test_settings: Dict[str, Any],
    plan: Dict[str, Any],
    started_at: Optional[float],
    finished_at: Optional[float],
) -> str:
    if dft is None or dft.empty:
        return "結果サマリーはまだありません。"

    n_total = len(dft)
    n_signal = int(dft["is_signal"].sum()) if "is_signal" in dft.columns else 0
    n_noise = max(0, n_total - n_signal)

    hits = int(((dft["trial_type"] == "glide") & (dft["response"] == "change")).sum())
    misses = int(((dft["trial_type"] == "glide") & (dft["response"] == "flat")).sum())
    fas = int(((dft["trial_type"] == "flat") & (dft["response"] == "change")).sum())
    crs = int(((dft["trial_type"] == "flat") & (dft["response"] == "flat")).sum())
    acc = float(dft["correct"].mean()) * 100.0 if n_total else float("nan")

    series_used = plan.get("series_name", test_settings.get("order_mode", "—"))
    seed_used = test_settings.get("pseudo_seed_used")

    lines: List[str] = [
        "Pitch Glide / Pitch Change Detection Threshold Test - Summary",
        f"出力日時: {format_session_time(time.time())}",
        f"本番開始: {format_session_time(started_at)}",
        f"本番終了: {format_session_time(finished_at)}",
        "",
        "[結果サマリー]",
        f"trial数: {n_total}",
        f"正答率: {acc:.1f}%" if n_total else "正答率: —",
        f"HIT率（GLIDE）: {_rate(hits, n_signal)}",
        f"FA率（FLAT）: {_rate(fas, n_noise)}",
        "",
        "[閾値（GLIDE duration）]",
        f"small-step reversals: {threshold_summary.get('usable_count', 0)} / {N_SMALL_REV_TARGET}",
    ]

    status = threshold_summary.get("status")
    if status == "official":
        lines.extend(
            [
                f"正式閾値（中央値）: {format_ms(threshold_summary.get('official_median'))}",
                f"正式閾値（平均）: {format_ms(threshold_summary.get('official_mean'))}",
                f"小ステップ期・最後{N_SMALL_REV_TARGET} reversals (ms): {format_ms_list(threshold_summary.get('official_levels', []))}",
            ]
        )
    elif status == "provisional":
        lines.extend(
            [
                "正式閾値: — （small-step reversals不足）",
                f"参考値（暫定中央値）: {format_ms(threshold_summary.get('reference_median'))}",
                f"小ステップ期 reversals (ms): {format_ms_list(threshold_summary.get('usable_levels', []))}",
            ]
        )
    elif status == "reference_range":
        lines.extend(
            [
                "正式閾値: — （small-step reversals不足）",
                f"参考値（中央値）: {format_ms(threshold_summary.get('reference_median'))}",
                f"参考範囲: {format_ms_range(threshold_summary.get('reference_min'), threshold_summary.get('reference_max'))}",
                f"小ステップ期 reversals (ms): {format_ms_list(threshold_summary.get('usable_levels', []))}",
            ]
        )
    else:
        lines.extend(
            [
                "正式閾値: — （small-step reversals不足）",
                "参考値: なし",
                f"小ステップ期 reversals (ms): {format_ms_list(threshold_summary.get('usable_levels', []))}",
            ]
        )

    lines.extend(
        [
            "",
            "[反応内訳]",
            f"HIT: {hits}",
            f"MISS: {misses}",
            f"FA: {fas}",
            f"CR: {crs}",
            "",
            "[終了条件]",
            finish_reason_label(reason),
            "",
            "[実施条件（スナップショット）]",
            json.dumps(test_settings, ensure_ascii=False, indent=2),
            "",
            "[使用系列（再現用）]",
            f"系列: {series_used}",
        ]
    )

    if series_used == "擬似ランダム":
        lines.append(f"seed: {seed_used}")
        lines.append("制約: 1×40, 2×60 / 同一値の連続は最大3回")

    lines.extend(
        [
            "GLIDE方向: up=30 / down=30（1=up, 2=down）",
            f"trial系列（1=FLAT,2=GLIDE）: {test_settings.get('trial_schedule_codes', [])}",
            f"GLIDE方向系列（1=up,2=down）: {test_settings.get('glide_direction_codes', [])}",
        ]
    )

    if sc is not None and len(sc.reversals) > 0:
        lines.extend(
            [
                "",
                "[reversals（GLIDE更新に基づく）]",
                pd.DataFrame(sc.reversals).to_string(index=False),
            ]
        )

    return "\n".join(lines) + "\n"


# ============================================================
# Session state
# ============================================================
def init_state():
    defaults = {
        "mode": "idle",  # idle | practice | test | finished
        "practice_streak": 0,  # counts GLIDE-HIT streak only
        "practice_log": [],
        "test_log": [],
        "trial": None,
        "awaiting_answer": False,
        "staircase": None,
        "test_trial_n": 0,
        "threshold_live_mean": None,
        "threshold_live_median": None,
        "threshold_final_mean": None,
        "threshold_final_median": None,
        "started_at": None,
        "finished_at": None,
        "finished_reason": None,
        "test_settings": None,
        "practice_settings": None,
        "test_plan": None,
        "schedule": None,
        "schedule_codes": None,
        "schedule_types": None,
        "glide_dir_codes": None,
        "glide_dir_labels": None,
        "order_mode_test": "系列1",
        "results_view": "本番ログ",
        "last_feedback": None,
        # demo buttons (instruction only; not logged)
        "demo_wav": None,
        "demo_label": None,
        "demo_total_ms": None,
        "runtime_error": None,
        # early stop streaks (GLIDE trials only; FLAT does not reset)
        "ceil_miss_streak": 0,
        "floor_hit_streak": 0,
        # config validation (for start gating)
        "config_errors": [],
        "config_warnings": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all():
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    init_state()


def clear_practice_state(*, clear_log: bool):
    st.session_state["practice_streak"] = 0
    st.session_state["practice_settings"] = None
    st.session_state["last_feedback"] = None
    if clear_log:
        st.session_state["practice_log"] = []


def clear_test_state(*, clear_log: bool):
    st.session_state["staircase"] = None
    st.session_state["test_trial_n"] = 0
    st.session_state["threshold_live_mean"] = None
    st.session_state["threshold_live_median"] = None
    st.session_state["threshold_final_mean"] = None
    st.session_state["threshold_final_median"] = None
    st.session_state["started_at"] = None
    st.session_state["finished_at"] = None
    st.session_state["finished_reason"] = None
    st.session_state["test_settings"] = None
    st.session_state["test_plan"] = None
    st.session_state["schedule"] = None
    st.session_state["schedule_codes"] = None
    st.session_state["schedule_types"] = None
    st.session_state["glide_dir_codes"] = None
    st.session_state["glide_dir_labels"] = None
    st.session_state["ceil_miss_streak"] = 0
    st.session_state["floor_hit_streak"] = 0
    st.session_state["order_mode_test"] = "系列1"
    if clear_log:
        st.session_state["test_log"] = []


def set_runtime_error(message: Optional[str]):
    st.session_state["runtime_error"] = None if message is None else str(message)


init_state()


# ============================================================
# Sidebar settings
# ============================================================
with st.sidebar:
    st.header("⚙️ 設定")

    preset_name = st.radio("周波数帯（プリセット）", list(PRESETS.keys()), index=0)
    f_center = float(PRESETS[preset_name]["f_center"])
    preset_delta_default = float(PRESETS[preset_name]["delta_default"])

    # Delta is user-adjustable; default = preset default
    if "pg_prev_preset_cd" not in st.session_state:
        st.session_state["pg_prev_preset_cd"] = preset_name
    if "delta_hz_cd" not in st.session_state:
        st.session_state["delta_hz_cd"] = preset_delta_default
    if st.session_state["pg_prev_preset_cd"] != preset_name:
        st.session_state["delta_hz_cd"] = preset_delta_default
        st.session_state["pg_prev_preset_cd"] = preset_name

    max_delta = max(10.0, float(f_center) - 10.0)
    delta = st.number_input(
        "偏移 Δf (Hz)",
        min_value=10.0,
        max_value=float(max_delta),
        value=float(st.session_state["delta_hz_cd"]),
        step=10.0,
        key="delta_hz_cd",
    )
    st.caption(
        f"中心周波数 f_center = **{f_center:.0f} Hz** / 偏移 Δf = **±{float(delta):.0f} Hz**（既定：±{preset_delta_default:.0f} Hz）"
    )

    ear = st.radio("出力", ["両耳", "左耳のみ", "右耳のみ"], index=0)

    st.divider()
    st.subheader("刺激")
    st.caption(f"サンプリング周波数は **{SR_FIXED} Hz 固定**です。")
    sr = SR_FIXED
    steady_ms = st.number_input("定常部 (ms)", min_value=0, max_value=1000, value=200, step=10)
    st.caption("※ GLIDEの周波数遷移（ramp_ms）は下のStaircaseの **D**（ms）で可変です。ここでは遷移後の定常部（steady_ms）と音のフェード（edge_ramp_ms）を設定します。")
    edge_ramp_ms = st.number_input("フェード（cosine, ms）", min_value=0, max_value=30, value=10, step=1)
    target_rms = st.number_input(
        "RMS正規化 target",
        min_value=0.01,
        max_value=0.3,
        value=0.10,
        step=0.01,
        format="%.2f",
    )

    st.divider()
    st.subheader("本番（系列）")
    order_mode_select = st.selectbox(
        "系列（本番開始時に固定）",
        options=["系列1", "系列2", "擬似ランダム"],
        index=0,
    )

    # --- pseudo-random controls / preview ---
    if "pseudo_seed" not in st.session_state:
        st.session_state["pseudo_seed"] = int(time.time()) % 1_000_000_000
    pseudo_seed = int(st.session_state["pseudo_seed"])
    max_consecutive = 3

    if order_mode_select == "擬似ランダム":
        pseudo_seed = int(
            st.number_input(
                "擬似ランダム seed（本番開始時に固定）",
                min_value=0,
                max_value=999_999_999,
                value=int(st.session_state["pseudo_seed"]),
                step=1,
            )
        )
        st.session_state["pseudo_seed"] = pseudo_seed

        if st.button("🎲 seed をランダムにする"):
            st.session_state["pseudo_seed"] = random.randint(0, 999_999_999)
            st.rerun()

        plan_preview = build_test_plan(series_name="擬似ランダム", pseudo_seed=pseudo_seed, max_consecutive=max_consecutive)
        seq_preview = plan_preview["schedule_codes"]
        st.caption(
            f"擬似ランダム：**{len(seq_preview)} trial**（FLAT={seq_preview.count(1)} / GLIDE={seq_preview.count(2)}）"
            f" / 連続制約：同一値は最大 {max_consecutive} 回まで"
        )
        with st.expander("擬似ランダム系列（1/2）のプレビュー"):
            st.code(str(seq_preview), language="text")

        st.caption("GLIDE方向も固定：**up=30 / down=30**（1=up, 2=down）。")
        with st.expander("GLIDE方向系列（1=up,2=down）プレビュー"):
            st.code(str(plan_preview["glide_dir_codes"]), language="text")
    else:
        seq_preview = FIXED_TRIAL_SERIES[order_mode_select]
        st.caption(f"この系列：**{len(seq_preview)} trial**（FLAT={seq_preview.count(1)} / GLIDE={seq_preview.count(2)}）")
        st.caption("GLIDE方向も固定：**up=30 / down=30**（1=up, 2=down）。")

    st.caption("※ 表記：**1=FLAT**, **2=GLIDE**。本番は **100 trial固定**です。")

    st.divider()
    st.subheader("Staircase（GLIDE duration ms）")
    start_ms = st.number_input("開始 D (ms)", min_value=20, max_value=800, value=300, step=10)
    floor_ms = st.number_input("D_min (ms)", min_value=5, max_value=200, value=20, step=5)
    ceil_ms = st.number_input("D_max (ms)", min_value=50, max_value=2000, value=600, step=50)

    step_big_ms = st.number_input("大ステップ (ms)", min_value=5, max_value=200, value=40, step=5)
    step_small_ms = st.number_input("小ステップ (ms)", min_value=1, max_value=100, value=20, step=1)
    switch_after = st.number_input("大→小 切替reversal数", min_value=1, max_value=10, value=4, step=1)

    st.divider()
    st.subheader("練習（任意）")
    practice_duration_ms = st.number_input(
        "練習用 D (ms)",
        min_value=20,
        max_value=2000,
        value=600,
        step=20,
        help="練習モードのGLIDE durationです。本番のstaircaseとは独立しています。",
    )
    practice_must = st.checkbox("練習で5連続HIT（GLIDE）を目標（推奨）", value=True)

    st.divider()
    if st.button("🧹 全リセット"):
        reset_all()
        st.rerun()


def snapshot_settings() -> Dict[str, Any]:
    """Freeze settings at block start (practice/test)."""
    return {
        "preset_name": preset_name,
        "f_center": float(f_center),
        "delta": float(delta),
        "ear": str(ear),
        "sr": int(sr),  # fixed 48k
        "steady_ms": int(steady_ms),
        "edge_ramp_ms": int(edge_ramp_ms),
        "target_rms": float(target_rms),
        "n_trials": int(N_TEST_TRIALS),  # fixed 100
        "order_mode": str(order_mode_select),  # selected series (frozen at test start as well)
        "pseudo_seed": None if str(order_mode_select) != "擬似ランダム" else int(st.session_state.get("pseudo_seed", 0)),
        "max_consecutive": int(3),
        "start_ms": float(start_ms),
        "floor_ms": float(floor_ms),
        "ceil_ms": float(ceil_ms),
        "step_big_ms": float(step_big_ms),
        "step_small_ms": float(step_small_ms),
        "switch_after": int(switch_after),
        "practice_duration_ms": float(practice_duration_ms),
        "practice_must": bool(practice_must),
    }


def validate_settings(s: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Parameter consistency check.

    Returns
    -------
    errors: List[str]
        Blocking issues (start_practice / start_test will not start)
    warnings: List[str]
        Non-blocking cautions
    """
    errors: List[str] = []
    warnings: List[str] = []

    # --- Fixed SR / trials ---
    if int(s.get("sr", -1)) != SR_FIXED:
        errors.append(f"SR は {SR_FIXED} Hz 固定です（現在: {s.get('sr')}）。")
    if int(s.get("n_trials", -1)) != N_TEST_TRIALS:
        errors.append(f"本番 trial 数は {N_TEST_TRIALS} 固定です（現在: {s.get('n_trials')}）。")

    # --- Series sanity ---
    if len(SERIES_1) != N_TEST_TRIALS or len(SERIES_2) != N_TEST_TRIALS:
        errors.append("固定系列（Series 1/2）の長さが 100 trial ではありません（実装の整合性エラー）。")

    order = str(s.get("order_mode"))
    series_options = ["系列1", "系列2", "擬似ランダム"]
    if order not in series_options:
        errors.append(f"系列が不明です: {order}")

    # Fixed series: direction schedule must be aligned with GLIDE count (60)
    if order in FIXED_TRIAL_SERIES:
        if FIXED_TRIAL_SERIES[order].count(2) != 60:
            errors.append("固定系列のGLIDE数が60ではありません（実装の整合性エラー）。")
        if len(FIXED_GLIDE_DIR_CODES.get(order, [])) != 60:
            errors.append("固定系列のGLIDE方向系列の長さが60ではありません（実装の整合性エラー）。")
    else:
        # pseudo-random
        if s.get("pseudo_seed") is None:
            errors.append("擬似ランダム seed が未設定です。")

    if int(s.get("max_consecutive", 3)) < 1:
        errors.append("連続制約（max_consecutive）は1以上にしてください。")

    # --- Stimulus params ---
    f0 = float(s.get("f_center", 0.0))
    df = float(s.get("delta", 0.0))
    if f0 <= 0:
        errors.append("f_center は正の値にしてください。")
    if df <= 0:
        errors.append("Δf は正の値にしてください。")
    if df >= f0:
        errors.append("Δf が f_center 以上です（開始周波数が 0 以下になり得ます）。")

    steady = int(s.get("steady_ms", 0))
    edge = int(s.get("edge_ramp_ms", 0))
    if steady < 0:
        errors.append("steady_ms は 0 以上にしてください。")
    if edge < 0:
        errors.append("edge_ramp_ms は 0 以上にしてください。")

    trms = float(s.get("target_rms", 0.0))
    if trms <= 0:
        errors.append("target_rms は 0 より大きい値にしてください。")

    # --- Staircase params ---
    floor = float(s.get("floor_ms", 0.0))
    ceil = float(s.get("ceil_ms", 0.0))
    start = float(s.get("start_ms", 0.0))
    step_big = float(s.get("step_big_ms", 0.0))
    step_small = float(s.get("step_small_ms", 0.0))
    sw = int(s.get("switch_after", 0))

    if floor <= 0:
        errors.append("D_min は 0 より大きい値にしてください。")
    if ceil <= floor:
        errors.append("D_max は D_min より大きい値にしてください。")
    if not (floor <= start <= ceil):
        errors.append("開始 D は D_min〜D_max の範囲にしてください。")

    if step_big <= 0 or step_small <= 0:
        errors.append("ステップ幅は正の値にしてください。")
    if sw <= 0:
        errors.append("大→小 切替reversal数は 1 以上にしてください。")

    if step_small > step_big:
        warnings.append("小ステップが大ステップより大きいです（意図した設定か確認してください）。")

    practice_duration = float(s.get("practice_duration_ms", 0.0))
    if practice_duration <= 0:
        errors.append("練習用 D は 0 より大きい値にしてください。")

    # --- Fade vs minimum total duration ---
    total_min = int(round(floor)) + int(steady)
    if total_min <= 0:
        errors.append("(D_min + steady_ms) が 0 以下です。刺激長が 0 になります。")
    if total_min > 0 and (2 * edge) > total_min:
        warnings.append(
            "edge_ramp_ms が刺激長に対して長すぎます。フェードは内部でクランプされ、指定どおりの長さにならない可能性があります。"
        )

    return errors, warnings


# ============================================================
# Trial creation and response handling
# ============================================================
def start_practice():
    # Clear any demo audio so it cannot collide with trial playback widgets.
    st.session_state["demo_wav"] = None
    st.session_state["demo_label"] = None
    st.session_state["demo_total_ms"] = None
    set_runtime_error(None)

    s = snapshot_settings()
    errors, warnings = validate_settings(s)
    st.session_state["config_errors"] = errors
    st.session_state["config_warnings"] = warnings
    if errors:
        st.session_state["mode"] = "idle"
        return

    st.session_state["mode"] = "practice"
    clear_test_state(clear_log=False)
    clear_practice_state(clear_log=True)
    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["practice_settings"] = s
    st.session_state["results_view"] = "練習ログ"


def start_test():
    # Clear any demo audio so it cannot collide with trial playback widgets.
    st.session_state["demo_wav"] = None
    st.session_state["demo_label"] = None
    st.session_state["demo_total_ms"] = None
    set_runtime_error(None)

    s = snapshot_settings()
    errors, warnings = validate_settings(s)
    st.session_state["config_errors"] = errors
    st.session_state["config_warnings"] = warnings
    if errors:
        st.session_state["mode"] = "idle"
        return

    st.session_state["mode"] = "test"
    clear_practice_state(clear_log=False)
    clear_test_state(clear_log=True)
    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["started_at"] = time.time()
    st.session_state["test_settings"] = s

    # Freeze series at test start
    st.session_state["order_mode_test"] = str(order_mode_select)

    # Build & freeze test plan (schedule + glide directions)
    try:
        plan = build_test_plan(
            series_name=str(st.session_state["order_mode_test"]),
            pseudo_seed=s.get("pseudo_seed"),
            max_consecutive=int(s.get("max_consecutive", 3)),
        )
    except Exception as exc:
        clear_test_state(clear_log=True)
        st.session_state["mode"] = "idle"
        set_runtime_error(f"本番系列の生成に失敗しました: {exc}")
        return

    st.session_state["test_plan"] = plan
    st.session_state["schedule_codes"] = plan["schedule_codes"]
    st.session_state["schedule_types"] = plan["schedule_types"]
    st.session_state["glide_dir_codes"] = plan["glide_dir_codes"]
    st.session_state["glide_dir_labels"] = plan["glide_dir_labels"]

    # Back-compat convenience
    st.session_state["schedule"] = plan["schedule_types"]

    # Keep reproducibility info in the frozen settings snapshot (shown in summary)
    st.session_state["test_settings"]["pseudo_seed_used"] = plan.get("pseudo_seed")
    st.session_state["test_settings"]["trial_schedule_codes"] = plan.get("schedule_codes")
    st.session_state["test_settings"]["glide_direction_codes"] = plan.get("glide_dir_codes")

    st.session_state["staircase"] = DurationStaircase(
        start_ms=float(s["start_ms"]),
        floor_ms=float(s["floor_ms"]),
        ceil_ms=float(s["ceil_ms"]),
        step_big_ms=float(s["step_big_ms"]),
        step_small_ms=float(s["step_small_ms"]),
        switch_after_reversals=int(s["switch_after"]),
    )
    st.session_state["threshold_live_mean"] = None
    st.session_state["threshold_live_median"] = None
    st.session_state["threshold_final_mean"] = None
    st.session_state["threshold_final_median"] = None

    st.session_state["results_view"] = "本番ログ"


def stop_now():
    """Stop current block.
    - practice -> back to idle
    - test -> finish (show summary)
    """
    if st.session_state.get("mode") == "practice":
        st.session_state["mode"] = "idle"
        st.session_state["trial"] = None
        st.session_state["awaiting_answer"] = False
        clear_practice_state(clear_log=False)
        st.session_state["results_view"] = "練習ログ"
        return
    if st.session_state.get("mode") == "test":
        finish_test(reason="manual")


def finish_test(reason: str = "n_trials"):
    st.session_state["mode"] = "finished"
    st.session_state["finished_at"] = time.time()
    st.session_state["finished_reason"] = str(reason)

    sc: DurationStaircase = st.session_state.get("staircase")
    if sc is not None:
        st.session_state["threshold_final_mean"] = sc.threshold_last6_mean()
        st.session_state["threshold_final_median"] = sc.threshold_last6_median()

    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["results_view"] = "結果サマリー"


def make_new_trial(mode: str):
    """
    Create trial and store into session_state['trial'].
    - practice: random trial type (50/50), easy duration (ceil_ms)
    - test: follows frozen schedule, duration from staircase (GLIDE trials)
    """
    settings = st.session_state["practice_settings"] if mode == "practice" else st.session_state["test_settings"]
    if not settings:
        settings = snapshot_settings()

    try:
        glide_no_planned = None
        glide_dir_code_planned = None
        glide_dir_label_planned = None

        if mode == "practice":
            trial_type = random.choice(["flat", "glide"])
            D_ms = int(round(float(settings["practice_duration_ms"])))
            planned_no = len(st.session_state["practice_log"]) + 1
            planned_code = None
            direction = random.choice(["up", "down"]) if trial_type == "glide" else "up"
        else:
            idx0 = int(st.session_state["test_trial_n"])  # 0-index for schedule

            # Frozen schedule (created in start_test)
            schedule_types = st.session_state.get("schedule_types")
            schedule_codes = st.session_state.get("schedule_codes")
            glide_dir_labels = st.session_state.get("glide_dir_labels")
            glide_dir_codes = st.session_state.get("glide_dir_codes")

            if not (schedule_types and schedule_codes and glide_dir_labels and glide_dir_codes):
                # Safety rebuild (should rarely happen)
                frozen = st.session_state.get("test_settings") or snapshot_settings()
                plan = build_test_plan(
                    series_name=str(st.session_state.get("order_mode_test", "系列1")),
                    pseudo_seed=frozen.get("pseudo_seed"),
                    max_consecutive=int(frozen.get("max_consecutive", 3)),
                )
                st.session_state["test_plan"] = plan
                st.session_state["schedule_codes"] = plan["schedule_codes"]
                st.session_state["schedule_types"] = plan["schedule_types"]
                st.session_state["glide_dir_codes"] = plan["glide_dir_codes"]
                st.session_state["glide_dir_labels"] = plan["glide_dir_labels"]
                st.session_state["schedule"] = plan["schedule_types"]
                schedule_types = plan["schedule_types"]
                schedule_codes = plan["schedule_codes"]
                glide_dir_labels = plan["glide_dir_labels"]
                glide_dir_codes = plan["glide_dir_codes"]

            if idx0 >= len(schedule_types):
                finish_test(reason="n_trials")
                return

            trial_type = schedule_types[idx0]
            sc: DurationStaircase = st.session_state["staircase"]
            D_ms = int(round(float(sc.x_ms)))
            planned_no = idx0 + 1
            planned_code = 1 if trial_type == "flat" else 2

            # Fixed / pseudo-random glide direction (GLIDE trials only)
            if trial_type == "glide":
                glide_idx0 = int(list(schedule_codes[:idx0]).count(2))  # 0-based index within glide-only list
                glide_no_planned = glide_idx0 + 1
                if 0 <= glide_idx0 < len(glide_dir_labels):
                    glide_dir_label_planned = str(glide_dir_labels[glide_idx0])
                    glide_dir_code_planned = int(glide_dir_codes[glide_idx0])
                else:
                    glide_dir_label_planned = random.choice(["up", "down"])
                    glide_dir_code_planned = 1 if glide_dir_label_planned == "up" else 2

            direction = str(glide_dir_label_planned) if trial_type == "glide" else "up"

        wav, total_ms = generate_trial_wav_single(
            sr=int(settings["sr"]),
            f_center=float(settings["f_center"]),
            delta=float(settings["delta"]),
            ramp_ms=int(D_ms),
            steady_ms=int(settings["steady_ms"]),
            ear=str(settings["ear"]),
            edge_ramp_ms=int(settings["edge_ramp_ms"]),
            target_rms=float(settings["target_rms"]),
            trial_type=trial_type,
            direction=direction,
        )
    except Exception as exc:
        st.session_state["trial"] = None
        st.session_state["awaiting_answer"] = False
        set_runtime_error(f"試行の生成に失敗しました: {exc}")
        if mode == "test":
            finish_test(reason="internal_error")
        else:
            clear_practice_state(clear_log=False)
            st.session_state["mode"] = "idle"
        return

    st.session_state["trial"] = {
        "wav": wav,
        "trial_type": trial_type,
        "direction": direction if trial_type == "glide" else None,
        "glide_no_planned": glide_no_planned,
        "glide_direction_planned": glide_dir_label_planned,
        "glide_direction_code_planned": glide_dir_code_planned,
        "D_ms": int(D_ms),
        "total_ms": int(total_ms),
        "trial_no_planned": int(planned_no),
        "trial_code_planned": planned_code,
        "series_name": st.session_state.get("order_mode_test") if mode == "test" else None,
        **settings,
        "created_at": time.time(),
    }
    st.session_state["awaiting_answer"] = True


def record_response(response: str):
    """
    response: "change" or "flat"
    """
    mode = st.session_state["mode"]
    trial = st.session_state.get("trial") or {}
    if not trial:
        return

    trial_type = trial["trial_type"]
    is_signal = (trial_type == "glide")

    # correctness for the detection task
    if is_signal:
        correct = (response == "change")  # HIT
    else:
        correct = (response == "flat")    # correct rejection

    row: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "trial_no": None,
        "trial_no_planned": trial.get("trial_no_planned"),
        "series_name": trial.get("series_name"),
        "pseudo_seed_used": trial.get("pseudo_seed_used"),
        "max_consecutive": trial.get("max_consecutive"),
        "trial_code_planned": trial.get("trial_code_planned"),  # 1=FLAT, 2=GLIDE (test only)
        "trial_type": trial_type,
        "direction": trial.get("direction"),
        "glide_no_planned": trial.get("glide_no_planned"),
        "glide_direction_planned": trial.get("glide_direction_planned"),
        "glide_direction_code_planned": trial.get("glide_direction_code_planned"),
        "response": response,
        "correct": int(bool(correct)),
        "is_signal": int(bool(is_signal)),
        "D_ms_presented": int(trial["D_ms"]),
        "total_ms": int(trial["total_ms"]),
        "preset": trial.get("preset_name"),
        "f_center": float(trial.get("f_center")),
        "delta": float(trial.get("delta")),
        "sr": int(trial.get("sr")),
        "steady_ms": int(trial.get("steady_ms")),
        "edge_ramp_ms": int(trial.get("edge_ramp_ms")),
        "target_rms": float(trial.get("target_rms")),
    }

    # ------------------------
    # Practice
    # ------------------------
    if mode == "practice":
        row["trial_no"] = len(st.session_state["practice_log"]) + 1
        st.session_state["practice_log"].append(row)

        # Practice streak counts only GLIDE-HITs; FLAT trials do not affect streak.
        if is_signal:
            st.session_state["practice_streak"] = st.session_state["practice_streak"] + 1 if correct else 0

        st.session_state["last_feedback"] = {"correct": bool(correct), "trial_type": trial_type}

        st.session_state["trial"] = None
        st.session_state["awaiting_answer"] = False

        if st.session_state["practice_settings"].get("practice_must", True) and st.session_state["practice_streak"] >= 5:
            st.session_state["mode"] = "idle"
        return

    # ------------------------
    # Test
    # ------------------------
    if mode == "test":
        st.session_state["test_trial_n"] += 1
        row["trial_no"] = int(st.session_state["test_trial_n"])

        sc: DurationStaircase = st.session_state["staircase"]

        upd = None
        n_small_rev = sc.n_small_reversals()
        if is_signal:
            upd = sc.update_on_glide(hit=bool(correct))
            st.session_state["threshold_live_mean"] = sc.threshold_last6_mean()
            st.session_state["threshold_live_median"] = sc.threshold_last6_median()
            n_small_rev = sc.n_small_reversals()

            # ---- Early-stop streaks (GLIDE only; FLAT does not reset) ----
            D_presented = int(trial["D_ms"])
            # ceiling miss streak
            if D_presented == int(round(float(sc.ceil_ms))) and (not correct):
                st.session_state["ceil_miss_streak"] += 1
            else:
                st.session_state["ceil_miss_streak"] = 0

            # floor hit streak
            if D_presented == int(round(float(sc.floor_ms))) and bool(correct):
                st.session_state["floor_hit_streak"] += 1
            else:
                st.session_state["floor_hit_streak"] = 0

        # Fill row with staircase info (even on FLAT, record current x_ms)
        row.update(
            {
                "update_used": int(is_signal),
                "D_ms_next": float(sc.x_ms),
                "direction_update": None if upd is None else upd["direction"],
                "step_used_ms": None if upd is None else upd["step_used_ms"],
                "phase": None if upd is None else upd["phase"],
                "reversal_phase": None if upd is None else upd.get("reversal_phase"),
                "reversal": 0 if upd is None else int(bool(upd["reversal"])),
                "reversal_level_ms": None if upd is None else upd["reversal_level_ms"],
                "n_reversals": int(len(sc.reversals)),
                "n_small_reversals": int(n_small_rev),
                "n_updates_glide": int(sc.trial_index_updates),
                "threshold_live_mean": st.session_state.get("threshold_live_mean"),
                "threshold_live_median": st.session_state.get("threshold_live_median"),
                "ceil_miss_streak": int(st.session_state.get("ceil_miss_streak", 0)),
                "floor_hit_streak": int(st.session_state.get("floor_hit_streak", 0)),
            }
        )

        st.session_state["test_log"].append(row)

        st.session_state["trial"] = None
        st.session_state["awaiting_answer"] = False

        # ---- Stop rules (priority: small reversals -> ceiling/floor -> n_trials) ----
        if is_signal:
            if int(n_small_rev) >= int(N_SMALL_REV_TARGET):
                finish_test(reason="small_reversals")
                return
            if int(st.session_state.get("ceil_miss_streak", 0)) >= 2:
                finish_test(reason="ceiling_miss")
                return
            if int(st.session_state.get("floor_hit_streak", 0)) >= 4:
                finish_test(reason="floor_hit")
                return

        if st.session_state["test_trial_n"] >= int(st.session_state["test_settings"]["n_trials"]):
            finish_test(reason="n_trials")
            return

        return


# ============================================================
# Top controls
# ============================================================
mode = st.session_state["mode"]

# While idle, keep showing preflight validation results for the current sidebar settings.
if mode == "idle":
    errs, warns = validate_settings(snapshot_settings())
    st.session_state["config_errors"] = errs
    st.session_state["config_warnings"] = warns

if mode == "idle":
    errs = st.session_state.get("config_errors", [])
    warns = st.session_state.get("config_warnings", [])
    runtime_error = st.session_state.get("runtime_error")
    if runtime_error:
        st.error(runtime_error)
    if errs:
        st.error("設定に不整合があるため開始できません：\n- " + "\n- ".join(errs))
    if warns:
        st.warning("注意（開始はできますが推奨しません）：\n- " + "\n- ".join(warns))



# ============================================================
# Instruction demo sounds (does NOT affect logs / staircase)
# ============================================================
st.markdown("### 🔊 ルール説明用：試聴（デモ）")
st.caption(
    f"この試聴はログに記録されません。GLIDEは **UP** と **DOWN** を別々に再生できます。"
    f"デモの周波数遷移（D, ramp_ms）は **{DEMO_RAMP_MS} ms 固定**です（FLATは total = D + steady_ms で長さ合わせ）。"
)

# Disable demo while a block is running (to discourage replay during test)
demo_disabled = mode in ["practice", "test"]

# Minimal safety check for demo generation
try:
    _s_demo = snapshot_settings()
    _f0_demo = float(_s_demo.get("f_center", 0.0))
    _df_demo = float(_s_demo.get("delta", 0.0))
    if _f0_demo <= 0 or _df_demo <= 0 or _df_demo >= _f0_demo:
        demo_disabled = True
        st.warning("Δf と f_center の設定が不適切なため、試聴できません（Δf < f_center を満たすように調整してください）。")
except Exception as _e:
    demo_disabled = True
    st.warning(f"設定の読み取りに失敗したため試聴できません: {_e}")

d1, d2, d3 = st.columns(3)
with d1:
    if st.button("🔊 変化あり（UP）", disabled=demo_disabled, key="demo_glide_up"):
        try:
            s = snapshot_settings()
            wav, total_ms = generate_trial_wav_single(
                sr=int(s["sr"]),
                f_center=float(s["f_center"]),
                delta=float(s["delta"]),
                ramp_ms=int(DEMO_RAMP_MS),
                steady_ms=int(s["steady_ms"]),
                ear=str(s["ear"]),
                edge_ramp_ms=int(s["edge_ramp_ms"]),
                target_rms=float(s["target_rms"]),
                trial_type="glide",
                direction="up",
            )
            st.session_state["demo_wav"] = wav
            st.session_state["demo_label"] = "GLIDE (UP)"
            st.session_state["demo_total_ms"] = int(total_ms)
            set_runtime_error(None)
        except Exception as exc:
            st.session_state["demo_wav"] = None
            st.session_state["demo_label"] = None
            st.session_state["demo_total_ms"] = None
            set_runtime_error(f"デモ音声の生成に失敗しました: {exc}")

with d2:
    if st.button("🔊 変化あり（DOWN）", disabled=demo_disabled, key="demo_glide_down"):
        try:
            s = snapshot_settings()
            wav, total_ms = generate_trial_wav_single(
                sr=int(s["sr"]),
                f_center=float(s["f_center"]),
                delta=float(s["delta"]),
                ramp_ms=int(DEMO_RAMP_MS),
                steady_ms=int(s["steady_ms"]),
                ear=str(s["ear"]),
                edge_ramp_ms=int(s["edge_ramp_ms"]),
                target_rms=float(s["target_rms"]),
                trial_type="glide",
                direction="down",
            )
            st.session_state["demo_wav"] = wav
            st.session_state["demo_label"] = "GLIDE (DOWN)"
            st.session_state["demo_total_ms"] = int(total_ms)
            set_runtime_error(None)
        except Exception as exc:
            st.session_state["demo_wav"] = None
            st.session_state["demo_label"] = None
            st.session_state["demo_total_ms"] = None
            set_runtime_error(f"デモ音声の生成に失敗しました: {exc}")

with d3:
    if st.button("🔊 変化なし（FLAT）", disabled=demo_disabled, key="demo_flat"):
        try:
            s = snapshot_settings()
            wav, total_ms = generate_trial_wav_single(
                sr=int(s["sr"]),
                f_center=float(s["f_center"]),
                delta=float(s["delta"]),
                ramp_ms=int(DEMO_RAMP_MS),
                steady_ms=int(s["steady_ms"]),
                ear=str(s["ear"]),
                edge_ramp_ms=int(s["edge_ramp_ms"]),
                target_rms=float(s["target_rms"]),
                trial_type="flat",
                direction="up",
            )
            st.session_state["demo_wav"] = wav
            st.session_state["demo_label"] = "FLAT"
            st.session_state["demo_total_ms"] = int(total_ms)
            set_runtime_error(None)
        except Exception as exc:
            st.session_state["demo_wav"] = None
            st.session_state["demo_label"] = None
            st.session_state["demo_total_ms"] = None
            set_runtime_error(f"デモ音声の生成に失敗しました: {exc}")

if mode == "idle" and st.session_state.get("demo_wav") is not None:
    st.audio(st.session_state["demo_wav"], format="audio/wav", autoplay=True)
    st.caption(
        f"再生：**{st.session_state.get('demo_label','')}**  |  "
        f"D={DEMO_RAMP_MS} ms  |  total={st.session_state.get('demo_total_ms','—')} ms"
    )
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.button("🧪 練習を開始", disabled=(mode in ["practice", "test"] or bool(st.session_state.get("config_errors"))), on_click=start_practice)
with c2:
    st.button("🎯 本番を開始（練習スキップ可）", disabled=(mode in ["practice", "test"] or bool(st.session_state.get("config_errors"))), on_click=start_test)
with c3:
    st.button("⏹️ 終了", disabled=(mode not in ["practice", "test"]), on_click=stop_now)

st.divider()

# ============================================================
# Status metrics (always shown)
# ============================================================
if mode == "practice":
    sc = None
    ts = st.session_state.get("practice_settings") or snapshot_settings()
elif mode in ["test", "finished"]:
    sc = st.session_state.get("staircase", None)
    ts = st.session_state.get("test_settings") or snapshot_settings()
else:
    sc = None
    ts = snapshot_settings()

series_now = st.session_state.get("order_mode_test", "系列1") if st.session_state.get("mode") in ["test", "finished"] else str(order_mode_select)

# Row 1
r1 = st.columns(4)
r1[0].metric("mode", st.session_state["mode"])
r1[1].metric("series", series_now)
r1[2].metric("trial", f"{st.session_state.get('test_trial_n', 0)}/{int(ts['n_trials'])}")
r1[3].metric("D (next)", "—" if sc is None else f"{sc.x_ms:.0f} ms")

# Row 2
r2 = st.columns(4)
r2[0].metric("updates", f"{sc.trial_index_updates if sc else 0}")
r2[1].metric("rev", f"{len(sc.reversals) if sc else 0}")
r2[2].metric("small", f"{(sc.n_small_reversals() if sc else 0)}/{N_SMALL_REV_TARGET}")

live_med = st.session_state.get("threshold_live_median", None)
r2[3].metric("thr (med)", "—" if live_med is None else f"{live_med:.1f} ms")

st.caption(
    f"本番：FLAT=40 / GLIDE=60（100 trial固定）  |  "
    f"Δf=±{float(ts['delta']):.0f} Hz / f_center={float(ts['f_center']):.0f} Hz  |  "
    f"SR={SR_FIXED} Hz"
)

# ============================================================
# Main interaction
# ============================================================
if mode == "idle":
    st.info("上のボタンから **練習** または **本番** を開始してください。設定は左のサイドバーで変更できます。")

elif mode in ["practice", "test"]:
    label = "🧪 練習" if mode == "practice" else "🎯 本番"
    st.subheader(label)

    if st.session_state["last_feedback"] is not None and mode == "practice":
        fb = st.session_state["last_feedback"]
        if fb["correct"]:
            st.success(f"✅ 正解（{fb['trial_type'].upper()}）")
        else:
            st.error(f"❌ 不正解（{fb['trial_type'].upper()}）")

    if mode == "practice":
        st.caption(f"練習：GLIDE-HIT 連続 {st.session_state.get('practice_streak', 0)} / 5（FLATはカウントに影響しません）")

    if not st.session_state["awaiting_answer"]:
        make_new_trial(mode)

    trial = st.session_state.get("trial")
    if st.session_state["awaiting_answer"] and trial:
        st.audio(trial["wav"], format="audio/wav")
        st.caption("上の再生ボタンで音を聞いてから、下の質問に回答してください。")
        st.markdown("**質問**：今の音は **高さが変化**しましたか？")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("変化あり（GLIDE）", key=f"resp_change_{mode}"):
                record_response("change")
                st.rerun()
        with a2:
            if st.button("変化なし（FLAT）", key=f"resp_flat_{mode}"):
                record_response("flat")
                st.rerun()

elif mode == "finished":
    st.subheader("✅ 本番終了（結果サマリーは下）")
    reason = st.session_state.get("finished_reason", "n_trials")
    st.caption(f"終了条件：**{finish_reason_label(reason)}**")

# ============================================================
# 📌 Logs / Results (always visible) — button switch
# ============================================================
st.divider()
st.subheader("📌 ログ・結果（常時表示）")

# Lock view during test/practice
if st.session_state["mode"] == "test":
    st.session_state["results_view"] = "本番ログ"
elif st.session_state["mode"] == "practice":
    st.session_state["results_view"] = "練習ログ"
elif st.session_state["mode"] == "finished":
    if st.session_state.get("results_view") not in ["練習ログ", "本番ログ", "結果サマリー"]:
        st.session_state["results_view"] = "結果サマリー"

bcols = st.columns(3)
with bcols[0]:
    if st.button("練習ログ", disabled=(st.session_state["mode"] == "test")):
        st.session_state["results_view"] = "練習ログ"
with bcols[1]:
    if st.button("本番ログ"):
        st.session_state["results_view"] = "本番ログ"
with bcols[2]:
    if st.button("結果サマリー", disabled=(st.session_state["mode"] != "finished")):
        st.session_state["results_view"] = "結果サマリー"

view = st.session_state["results_view"]
st.write(f"表示：**{view}**")


def _rate(x: int, n: int) -> str:
    if n <= 0:
        return "—"
    return f"{(x/n)*100:.1f}%"


if view == "練習ログ":
    if len(st.session_state["practice_log"]) == 0:
        st.caption("練習ログはまだありません。")
    else:
        dfp = pd.DataFrame(st.session_state["practice_log"])
        st.dataframe(dfp, use_container_width=True, height=360)
        st.download_button(
            "⬇️ 練習ログCSVをダウンロード",
            data=dfp.to_csv(index=False).encode("utf-8-sig"),
            file_name="pitch_glide_practice_log.csv",
            mime="text/csv",
        )

elif view == "本番ログ":
    if len(st.session_state["test_log"]) == 0:
        st.caption("本番ログはまだありません。")
    else:
        dft = pd.DataFrame(st.session_state["test_log"])
        st.dataframe(dft, use_container_width=True, height=360)
        st.download_button(
            "⬇️ 本番ログCSVをダウンロード",
            data=dft.to_csv(index=False).encode("utf-8-sig"),
            file_name="pitch_glide_test_log.csv",
            mime="text/csv",
        )

else:
    if st.session_state["mode"] != "finished" or len(st.session_state["test_log"]) == 0:
        st.caption("本番を実施して終了すると、ここに結果サマリーが表示されます。")
    else:
        dft = pd.DataFrame(st.session_state["test_log"])
        sc: DurationStaircase = st.session_state.get("staircase")
        threshold_summary = build_threshold_summary(sc)

        n_total = len(dft)
        n_signal = int(dft["is_signal"].sum())
        n_noise = n_total - n_signal

        hits = int(((dft["trial_type"] == "glide") & (dft["response"] == "change")).sum())
        misses = int(((dft["trial_type"] == "glide") & (dft["response"] == "flat")).sum())
        fas = int(((dft["trial_type"] == "flat") & (dft["response"] == "change")).sum())
        crs = int(((dft["trial_type"] == "flat") & (dft["response"] == "flat")).sum())

        acc = float(dft["correct"].mean()) * 100.0 if n_total else float("nan")

        reason = st.session_state.get("finished_reason", "n_trials")
        point_label = "100 trial時点" if str(reason) == "n_trials" else "終了時点"

        st.markdown("### ✅ 結果サマリー（本番）")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("trial数", f"{n_total}")
        m2.metric("正答率", f"{acc:.1f}%")
        m3.metric("HIT率（GLIDE）", _rate(hits, n_signal))
        m4.metric("FA率（FLAT）", _rate(fas, n_noise))

        st.markdown("#### 閾値（GLIDE duration）")
        st.write(f"- **small-step reversals**: {threshold_summary.get('usable_count', 0)} / {N_SMALL_REV_TARGET}")

        status = threshold_summary.get("status")
        if status == "official":
            st.write(f"- **正式閾値（中央値）**: **{threshold_summary['official_median']:.1f} ms**")
            st.write(f"- 正式閾値（平均）: {threshold_summary['official_mean']:.1f} ms")
            st.caption(
                f"小ステップ期・最後{N_SMALL_REV_TARGET} reversals: "
                f"{format_ms_list(threshold_summary.get('official_levels', []))}"
            )
        elif status == "provisional":
            st.info(f"{point_label}では正式閾値に未達のため、参考値（暫定中央値）を表示しています。")
            st.write(f"- **参考値（暫定中央値）**: **{threshold_summary['reference_median']:.1f} ms**")
            st.caption(
                f"小ステップ期 reversals: {format_ms_list(threshold_summary.get('usable_levels', []))}"
            )
        elif status == "reference_range":
            st.info(f"{point_label}では正式閾値に未達のため、参考値と参考範囲を表示しています。")
            st.write(f"- **参考値（中央値）**: **{threshold_summary['reference_median']:.1f} ms**")
            st.write(
                f"- **参考範囲**: {threshold_summary['reference_min']:.1f}–{threshold_summary['reference_max']:.1f} ms"
            )
            st.caption(
                f"小ステップ期 reversals: {format_ms_list(threshold_summary.get('usable_levels', []))}"
            )
        else:
            st.warning(f"{point_label}での small-step reversals が 0–1 個のため、参考値はありません。")
            st.caption(
                f"小ステップ期 reversals: {format_ms_list(threshold_summary.get('usable_levels', []))}"
            )

        st.markdown("#### 収束の折れ線グラフ")
        render_glide_convergence_chart(dft, threshold_summary)

        st.markdown("#### 反応内訳")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("HIT", str(hits))
        cB.metric("MISS", str(misses))
        cC.metric("FA", str(fas))
        cD.metric("CR", str(crs))

        st.markdown("#### 終了条件")
        st.write(f"- **{finish_reason_label(reason)}**")

        st.markdown("#### 実施条件（スナップショット）")
        test_settings = st.session_state.get("test_settings", {}) or {}
        st.json(test_settings)

        st.markdown("#### 使用系列（再現用）")
        plan = st.session_state.get("test_plan", {}) or {}
        series_used = plan.get("series_name", st.session_state.get("order_mode_test"))
        seed_used = test_settings.get("pseudo_seed_used")
        st.write(f"- 系列: **{series_used}**")
        if series_used == "擬似ランダム":
            st.write(f"- seed: **{seed_used}**")
            st.write("- 制約: 1×40, 2×60 / 同一値の連続は最大3回")
        st.write("- GLIDE方向: **up=30 / down=30**（1=up, 2=down）")

        with st.expander("trial系列（1=FLAT,2=GLIDE）／GLIDE方向系列（1=up,2=down）"):
            st.code(str(test_settings.get("trial_schedule_codes", [])), language="text")
            st.code(str(test_settings.get("glide_direction_codes", [])), language="text")

        st.markdown("#### reversals（GLIDE更新に基づく）")
        if sc is not None and len(sc.reversals) > 0:
            st.dataframe(pd.DataFrame(sc.reversals), use_container_width=True, height=260)
        else:
            st.write("reversalなし")

        summary_text = build_summary_text(
            dft=dft,
            sc=sc,
            threshold_summary=threshold_summary,
            reason=str(reason),
            test_settings=test_settings,
            plan=plan,
            started_at=st.session_state.get("started_at"),
            finished_at=st.session_state.get("finished_at"),
        )
        finished_at = st.session_state.get("finished_at")
        finished_tag = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(float(finished_at)))
            if finished_at is not None
            else time.strftime("%Y%m%d_%H%M%S")
        )

        st.download_button(
            "⬇️ Summary .txt をダウンロード",
            data=summary_text.encode("utf-8-sig"),
            file_name=f"pitch_glide_summary_{finished_tag}.txt",
            mime="text/plain",
            key="download_summary_txt",
        )
