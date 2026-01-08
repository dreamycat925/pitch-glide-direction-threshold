import io
import random
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Pitch Glide CHANGE Detection Threshold Test (Single-interval)
# - Single stimulus per trial: FLAT vs GLIDE
# - Respond: "変化あり" / "変化なし"
# - Mix in FLAT trials to avoid expectancy and to estimate false alarms
# - Staircase on GLIDE duration (ms), updated on GLIDE trials only
# - 2-down 1-up (signal-only): 2 consecutive HITs -> harder (duration↓), MISS -> easier (duration↑)
# - Big step until N reversals, then small step
# - Threshold = median (and mean) of last 6 reversals in small-step phase
# ============================================================

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Pitch Change Detection Threshold Test",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 Pitch Change Detection Threshold Test（ピッチ変化“検出”閾値）")

st.markdown(
    """
**目的**  
単発刺激で「**高さが平坦（FLAT）**」か「**高さが変化（GLIDE）**」かを答えてもらい、  
**ピッチ変化を検出できる最小のグライド長（duration, ms）**を推定します。

**設計の意図（患者運用を想定）**  
- 2区間比較（ABの2AFC）を避け、**単発**で回答できる形式  
- 「変化なし（FLAT）」を混ぜて、**“常に変化あり”と答える戦略**を防止  
- 閾値推定（staircase）は **GLIDE試行のみ**で更新し、FLATは **false alarm** の推定に使います

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
def _cosine_ramp_env(n: int, sr: int, ramp_ms: int) -> np.ndarray:
    ramp_n = int(round(sr * ramp_ms / 1000))
    ramp_n = max(0, min(ramp_n, n // 2))
    env = np.ones(n, dtype=np.float32)
    if ramp_n > 0:
        t = np.arange(ramp_n, dtype=np.float32) / float(ramp_n)
        ramp = 0.5 - 0.5 * np.cos(np.pi * t)  # 0->1
        env[:ramp_n] = ramp
        env[-ramp_n:] = ramp[::-1]
    return env


def rms_normalize(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    x = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    return x * (float(target_rms) / rms)


def glide_stimulus_triangular(
    *,
    sr: int,
    f_center: float,
    delta: float,
    glide_ms: int,
    steady_ms: int,
    direction: str,  # "up" or "down" (for variety; detection doesn't depend on direction)
    ramp_ms: int,
    target_rms: float,
) -> np.ndarray:
    """
    One-interval GLIDE stimulus:
      - triangular glide lasting glide_ms (center -> +/-delta at mid -> center)
      - followed by steady tone at f_center lasting steady_ms
    """
    glide_ms = int(glide_ms)
    steady_ms = int(steady_ms)

    n_glide = max(2, int(round(sr * glide_ms / 1000)))
    t = np.arange(n_glide, dtype=np.float32) / float(sr)
    T = glide_ms / 1000.0

    # triangular modulation 0 -> 1 -> 0
    m = 1.0 - 2.0 * np.abs((t / T) - 0.5)
    m = np.clip(m, 0.0, 1.0)

    sign = 1.0 if direction == "up" else -1.0
    f_inst = f_center + sign * delta * m  # instantaneous frequency

    phase = 2.0 * np.pi * np.cumsum(f_inst) / float(sr)
    x_glide = np.sin(phase).astype(np.float32)

    n_steady = max(1, int(round(sr * steady_ms / 1000)))
    t2 = np.arange(n_steady, dtype=np.float32) / float(sr)
    phase0 = float(phase[-1])
    x_steady = np.sin(phase0 + 2.0 * np.pi * f_center * t2).astype(np.float32)

    x = np.concatenate([x_glide, x_steady]).astype(np.float32)

    # Apply ramp to the whole interval
    x *= _cosine_ramp_env(len(x), sr, ramp_ms)
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
    ramp_ms: int,
    target_rms: float,
) -> np.ndarray:
    """One-interval FLAT stimulus: steady tone only (same total duration as GLIDE interval)."""
    total_ms = int(total_ms)
    n = max(2, int(round(sr * total_ms / 1000)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    x = np.sin(2.0 * np.pi * f_center * t).astype(np.float32)
    x *= _cosine_ramp_env(len(x), sr, ramp_ms)
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
    glide_ms: int,
    steady_ms: int,
    ear: str,
    ramp_ms: int,
    target_rms: float,
    trial_type: str,  # "glide" or "flat"
    direction: str,   # "up" or "down" (used only if trial_type="glide")
) -> Tuple[bytes, int]:
    """
    Returns wav_bytes, total_ms (interval duration).
    """
    total_ms = int(glide_ms) + int(steady_ms)
    if trial_type == "glide":
        x = glide_stimulus_triangular(
            sr=sr,
            f_center=f_center,
            delta=delta,
            glide_ms=glide_ms,
            steady_ms=steady_ms,
            direction=direction,
            ramp_ms=ramp_ms,
            target_rms=target_rms,
        )
    else:
        x = flat_stimulus(
            sr=sr,
            f_center=f_center,
            total_ms=total_ms,
            ramp_ms=ramp_ms,
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
                    "phase": self.phase(),
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
            "reversal": reversal,
            "reversal_level_ms": reversal_level,
            "n_reversals": len(self.reversals),
            "n_updates": int(self.trial_index_updates),
        }

    def usable_reversal_levels(self) -> List[float]:
        if len(self.reversals) <= self.switch_after_reversals:
            return []
        return [float(r["level_ms"]) for r in self.reversals[self.switch_after_reversals :]]

    def threshold_last6_mean(self) -> Optional[float]:
        usable = self.usable_reversal_levels()
        if len(usable) < 6:
            return None
        return float(np.mean(usable[-6:]))

    def threshold_last6_median(self) -> Optional[float]:
        usable = self.usable_reversal_levels()
        if len(usable) < 6:
            return None
        return float(np.median(usable[-6:]))


# ============================================================
# Session state
# ============================================================
def init_state():
    defaults = {
        "mode": "idle",  # idle | practice | test | finished
        "practice_streak": 0,
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
        "test_settings": None,
        "practice_settings": None,
        "schedule": None,
        "schedule_seed": None,
        "results_view": "本番ログ",
        "last_feedback": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all():
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    init_state()


init_state()


# ============================================================
# Sidebar settings
# ============================================================
with st.sidebar:
    st.header("⚙️ 設定")

    subject_id = st.text_input("被験者ID（任意）", value="")

    preset_name = st.radio("周波数帯（プリセット）", list(PRESETS.keys()), index=0)
    f_center = float(PRESETS[preset_name]["f_center"])
    preset_delta_default = float(PRESETS[preset_name]["delta_default"])

    # Delta is user-adjustable; default = preset default (340 for 1240 preset)
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
    st.caption(f"中心周波数 f_center = **{f_center:.0f} Hz** / 偏移 Δf = **±{float(delta):.0f} Hz**（既定：±{preset_delta_default:.0f} Hz）")

    ear = st.radio("出力", ["両耳", "左耳のみ", "右耳のみ"], index=0)

    st.divider()
    st.subheader("刺激")
    sr = st.selectbox("サンプリング周波数", options=[44100, 48000], index=0)
    steady_ms = st.number_input("定常部 (ms)", min_value=0, max_value=1000, value=200, step=10)
    ramp_ms = st.number_input("ramp (ms)", min_value=0, max_value=30, value=10, step=1)
    target_rms = st.number_input(
        "RMS正規化 target",
        min_value=0.01,
        max_value=0.3,
        value=0.10,
        step=0.01,
        format="%.2f",
    )

    st.divider()
    st.subheader("本番設計（単発検出）")
    n_trials = st.number_input("本番 trial数（最大100）", min_value=10, max_value=100, value=50, step=5)
    flat_ratio = st.slider("FLATの割合（%）", min_value=0, max_value=60, value=20, step=5)
    st.caption("FLATを混ぜることで“常に変化あり”戦略を防ぎ、false alarm率も推定できます。")

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
    practice_must = st.checkbox("練習で5連続正答を目標（推奨）", value=True)

    st.divider()
    if st.button("🧹 全リセット"):
        reset_all()
        st.rerun()


def snapshot_settings() -> Dict[str, Any]:
    return {
        "preset_name": preset_name,
        "f_center": float(f_center),
        "delta": float(delta),
        "ear": str(ear),
        "sr": int(sr),
        "steady_ms": int(steady_ms),
        "ramp_ms": int(ramp_ms),
        "target_rms": float(target_rms),
        "n_trials": int(n_trials),
        "flat_ratio": int(flat_ratio),
        "start_ms": float(start_ms),
        "floor_ms": float(floor_ms),
        "ceil_ms": float(ceil_ms),
        "step_big_ms": float(step_big_ms),
        "step_small_ms": float(step_small_ms),
        "switch_after": int(switch_after),
        "practice_must": bool(practice_must),
    }


def build_schedule(n_trials: int, flat_ratio_pct: int, seed: Optional[int] = None) -> Tuple[List[str], int]:
    """
    Build a fixed schedule at test start:
      - exact number of FLAT trials (rounded)
      - remaining are GLIDE trials
      - shuffled with seed and then frozen
    """
    n_flat = int(round(n_trials * flat_ratio_pct / 100.0))
    n_flat = max(0, min(n_flat, n_trials))
    n_glide = n_trials - n_flat
    schedule = (["flat"] * n_flat) + (["glide"] * n_glide)

    if seed is None:
        seed = int(time.time() * 1000) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(schedule)
    return schedule, int(seed)


# ============================================================
# Trial creation and response handling
# ============================================================
def start_practice():
    st.session_state["mode"] = "practice"
    st.session_state["practice_streak"] = 0
    st.session_state["practice_log"] = []
    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["last_feedback"] = None
    st.session_state["practice_settings"] = snapshot_settings()
    st.session_state["results_view"] = "練習ログ"


def start_test():
    st.session_state["mode"] = "test"
    st.session_state["test_log"] = []
    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["last_feedback"] = None
    st.session_state["started_at"] = time.time()
    st.session_state["finished_at"] = None
    st.session_state["test_settings"] = snapshot_settings()
    st.session_state["test_trial_n"] = 0

    s = st.session_state["test_settings"]
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

    schedule, seed = build_schedule(int(s["n_trials"]), int(s["flat_ratio"]), seed=None)
    st.session_state["schedule"] = schedule
    st.session_state["schedule_seed"] = seed

    st.session_state["results_view"] = "本番ログ"


def finish_test():
    st.session_state["mode"] = "finished"
    st.session_state["finished_at"] = time.time()
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
    - practice: random trial type (50/50), easier duration (ceil_ms)
    - test: follows frozen schedule, duration from staircase (glide trials)
    """
    settings = st.session_state["practice_settings"] if mode == "practice" else st.session_state["test_settings"]
    if not settings:
        settings = snapshot_settings()

    if mode == "practice":
        trial_type = random.choice(["flat", "glide"])
        D_ms = int(round(float(settings["ceil_ms"])))  # practice: easy-ish
    else:
        idx = st.session_state["test_trial_n"]
        trial_type = st.session_state["schedule"][idx]
        sc: DurationStaircase = st.session_state["staircase"]
        D_ms = int(round(float(sc.x_ms)))

    direction = random.choice(["up", "down"])  # variety only

    wav, total_ms = generate_trial_wav_single(
        sr=int(settings["sr"]),
        f_center=float(settings["f_center"]),
        delta=float(settings["delta"]),
        glide_ms=int(D_ms),
        steady_ms=int(settings["steady_ms"]),
        ear=str(settings["ear"]),
        ramp_ms=int(settings["ramp_ms"]),
        target_rms=float(settings["target_rms"]),
        trial_type=trial_type,
        direction=direction,
    )

    st.session_state["trial"] = {
        "wav": wav,
        "trial_type": trial_type,
        "direction": direction if trial_type == "glide" else None,
        "D_ms": int(D_ms),
        "total_ms": int(total_ms),
        **settings,
        "created_at": time.time(),
    }
    st.session_state["awaiting_answer"] = True


def record_response(subject_id: str, response: str):
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

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subject_id": subject_id,
        "mode": mode,
        "trial_no": None,
        "trial_type": trial_type,
        "direction": trial.get("direction"),
        "response": response,
        "correct": int(correct),
        "is_signal": int(is_signal),
        "D_ms_presented": int(trial["D_ms"]),
        "total_ms": int(trial["total_ms"]),
        "preset": trial.get("preset_name"),
        "f_center": float(trial.get("f_center")),
        "delta": float(trial.get("delta")),
        "sr": int(trial.get("sr")),
        "steady_ms": int(trial.get("steady_ms")),
        "ramp_ms": int(trial.get("ramp_ms")),
        "target_rms": float(trial.get("target_rms")),
    }

    if mode == "practice":
        row["trial_no"] = len(st.session_state["practice_log"]) + 1
        st.session_state["practice_log"].append(row)

        st.session_state["practice_streak"] = st.session_state["practice_streak"] + 1 if correct else 0
        st.session_state["last_feedback"] = {"correct": bool(correct), "trial_type": trial_type}

        st.session_state["trial"] = None
        st.session_state["awaiting_answer"] = False

        if st.session_state["practice_settings"].get("practice_must", True) and st.session_state["practice_streak"] >= 5:
            st.session_state["mode"] = "idle"
        return

    if mode == "test":
        st.session_state["test_trial_n"] += 1
        row["trial_no"] = int(st.session_state["test_trial_n"])

        sc: DurationStaircase = st.session_state["staircase"]

        upd = None
        if is_signal:
            upd = sc.update_on_glide(hit=bool(correct))
            st.session_state["threshold_live_mean"] = sc.threshold_last6_mean()
            st.session_state["threshold_live_median"] = sc.threshold_last6_median()

        row.update(
            {
                "update_used": int(is_signal),
                "D_ms_next": float(sc.x_ms),
                "direction_update": None if upd is None else upd["direction"],
                "step_used_ms": None if upd is None else upd["step_used_ms"],
                "phase": None if upd is None else upd["phase"],
                "reversal": 0 if upd is None else int(bool(upd["reversal"])),
                "reversal_level_ms": None if upd is None else upd["reversal_level_ms"],
                "n_reversals": int(len(sc.reversals)),
                "n_updates_glide": int(sc.trial_index_updates),
                "threshold_live_mean": st.session_state.get("threshold_live_mean"),
                "threshold_live_median": st.session_state.get("threshold_live_median"),
                "schedule_seed": st.session_state.get("schedule_seed"),
            }
        )

        st.session_state["test_log"].append(row)

        st.session_state["trial"] = None
        st.session_state["awaiting_answer"] = False

        if st.session_state["test_trial_n"] >= int(st.session_state["test_settings"]["n_trials"]):
            finish_test()
        return


# ============================================================
# Top controls
# ============================================================
mode = st.session_state["mode"]

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.button("🧪 練習を開始", disabled=(mode in ["practice", "test"]), on_click=start_practice)
with c2:
    st.button("🎯 本番を開始（練習スキップ可）", disabled=(mode in ["practice", "test"]), on_click=start_test)
with c3:
    st.button("⏹️ 終了（結果表示）", disabled=(mode not in ["practice", "test"]), on_click=finish_test)

st.divider()

# ============================================================
# Status metrics (always shown)
# ============================================================
sc: Optional[DurationStaircase] = st.session_state.get("staircase", None)
ts = st.session_state.get("test_settings") or snapshot_settings()

mcols = st.columns(6)
mcols[0].metric("mode", st.session_state["mode"])
mcols[1].metric("practice streak", f"{st.session_state.get('practice_streak', 0)}")
mcols[2].metric("trial", f"{st.session_state.get('test_trial_n', 0)}/{int(ts['n_trials'])}")
mcols[3].metric("updates(GLIDE)", f"{sc.trial_index_updates if sc else 0}")
mcols[4].metric("reversals", f"{len(sc.reversals) if sc else 0}")

live_med = st.session_state.get("threshold_live_median", None)
mcols[5].metric("暫定閾値 (median)", "—" if live_med is None else f"{live_med:.1f} ms")

st.caption(f"本番FLAT割合: **{int(ts['flat_ratio'])}%** / Δf=±{float(ts['delta']):.0f} Hz / f_center={float(ts['f_center']):.0f} Hz")

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

    if not st.session_state["awaiting_answer"]:
        if st.button("▶️ 提示", key=f"present_{mode}"):
            make_new_trial(mode)
            st.rerun()

    trial = st.session_state.get("trial")
    if st.session_state["awaiting_answer"] and trial:
        st.audio(trial["wav"], format="audio/wav", autoplay=True)
        st.markdown("**質問**：今の音は **高さが変化**しましたか？")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("変化あり", key=f"resp_change_{mode}"):
                record_response(subject_id, "change")
                st.rerun()
        with a2:
            if st.button("変化なし（平坦）", key=f"resp_flat_{mode}"):
                record_response(subject_id, "flat")
                st.rerun()

elif mode == "finished":
    st.subheader("✅ 本番終了（結果サマリーは下）")

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
            file_name="pitch_change_detection_practice_log.csv",
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
            file_name="pitch_change_detection_test_log.csv",
            mime="text/csv",
        )

else:
    if st.session_state["mode"] != "finished" or len(st.session_state["test_log"]) == 0:
        st.caption("本番を実施して終了すると、ここに結果サマリーが表示されます。")
    else:
        dft = pd.DataFrame(st.session_state["test_log"])
        sc: DurationStaircase = st.session_state.get("staircase")
        thr_med = st.session_state.get("threshold_final_median")
        thr_mean = st.session_state.get("threshold_final_mean")

        n_total = len(dft)
        n_signal = int(dft["is_signal"].sum())
        n_noise = n_total - n_signal

        hits = int(((dft["trial_type"] == "glide") & (dft["response"] == "change")).sum())
        misses = int(((dft["trial_type"] == "glide") & (dft["response"] == "flat")).sum())
        fas = int(((dft["trial_type"] == "flat") & (dft["response"] == "change")).sum())
        crs = int(((dft["trial_type"] == "flat") & (dft["response"] == "flat")).sum())

        acc = float(dft["correct"].mean()) * 100.0 if n_total else float("nan")

        st.markdown("### ✅ 結果サマリー（本番）")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("trial数", f"{n_total}")
        m2.metric("正答率", f"{acc:.1f}%")
        m3.metric("HIT率（GLIDE）", _rate(hits, n_signal))
        m4.metric("FA率（FLAT）", _rate(fas, n_noise))

        st.markdown("#### 閾値（GLIDE duration）")
        if thr_med is None and thr_mean is None:
            st.warning("reversal数が不足しているため、閾値を算出できません（小ステップ期で6 reversalsが必要）。")
        else:
            st.write(f"- **閾値（中央値）**: **{thr_med:.1f} ms**" if thr_med is not None else "- 閾値（中央値）: —")
            st.write(f"- 閾値（平均）: {thr_mean:.1f} ms" if thr_mean is not None else "- 閾値（平均）: —")

            usable = sc.usable_reversal_levels() if sc is not None else []
            if len(usable) >= 6:
                last6 = usable[-6:]
                st.caption(f"小ステップ期・最後6 reversals: {', '.join([f'{x:.1f}' for x in last6])}")

        st.markdown("#### 反応内訳")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("HIT", str(hits))
        cB.metric("MISS", str(misses))
        cC.metric("FA", str(fas))
        cD.metric("CR", str(crs))

        st.markdown("#### 実施条件（スナップショット）")
        st.json(st.session_state.get("test_settings", {}))

        st.markdown("#### reversals（GLIDE更新に基づく）")
        if sc is not None and len(sc.reversals) > 0:
            st.dataframe(pd.DataFrame(sc.reversals), use_container_width=True, height=260)
        else:
            st.write("reversalなし")
