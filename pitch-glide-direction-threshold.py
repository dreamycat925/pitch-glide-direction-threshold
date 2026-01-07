import io
import random
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Pitch Glide Direction Threshold Test (PGDT)
# - 2AFC: which interval was "DOWN" (initially descending)?
# - 2-down 1-up staircase on glide duration (ms)
# - Large step until N reversals, then small step
# - Threshold = mean of last 6 reversals in small-step phase
# ============================================================

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Pitch Glide Direction Threshold Test",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 Pitch Glide Direction Threshold Test（ピッチ・グライド方向弁別閾値）")

st.markdown(
    """
**目的**  
2AFCで「どちらが *下がる音*（DOWN; 最初に下降するグライド）か」を答えてもらい、  
**方向弁別が可能になる最小のグライド長（duration, ms）**を推定します（= 速い変化ほど難しい）。

**近道（開始ピッチ手がかり）を避ける設計**  
UP/DOWN の開始周波数・終了周波数を同一にするため、グライド部は **三角形（triangular）**の周波数変化にしています。  
- UP: 最初に上昇 → 中盤でピーク → 終盤で中心周波数に戻る  
- DOWN: 最初に下降 → 中盤でボトム → 終盤で中心周波数に戻る  
その後に定常部（steady tone）を付加します。

**注意**  
- なるべく **有線ヘッドホン**（Bluetoothは遅延や途切れの原因になり得ます）  
- 音量は事前に快適レベルに調整  
- 原則 **replayしない**運用（提示は1回を想定）
"""
)

# ============================================================
# Presets
# ============================================================
PRESETS = {
    "1240 Hz版（F2帯寄り：900–1580 Hz）": {"f_center": 1240.0, "delta": 340.0},
    "500 Hz版（低周波：350–650 Hz）": {"f_center": 500.0, "delta": 150.0},
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
    start_direction: str,  # "up" or "down"
    ramp_ms: int,
    target_rms: float,
) -> np.ndarray:
    """
    One interval stimulus:
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

    sign = 1.0 if start_direction == "up" else -1.0
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


def silence(sr: int, dur_ms: int) -> np.ndarray:
    n = max(1, int(round(sr * dur_ms / 1000)))
    return np.zeros(n, dtype=np.float32)


def generate_trial_wav(
    *,
    sr: int,
    f_center: float,
    delta: float,
    glide_ms: int,
    steady_ms: int,
    isi_ms: int,
    ear: str,
    ramp_ms: int,
    target_rms: float,
    down_interval: Optional[int] = None,  # 1/2 or None=random
) -> Tuple[bytes, int, str]:
    """
    Returns wav_bytes, correct_interval(=DOWN position), order_label ("DOWN-UP" or "UP-DOWN")
    """
    correct_interval = down_interval if down_interval in (1, 2) else random.choice([1, 2])

    stim_down = glide_stimulus_triangular(
        sr=sr,
        f_center=f_center,
        delta=delta,
        glide_ms=glide_ms,
        steady_ms=steady_ms,
        start_direction="down",
        ramp_ms=ramp_ms,
        target_rms=target_rms,
    )
    stim_up = glide_stimulus_triangular(
        sr=sr,
        f_center=f_center,
        delta=delta,
        glide_ms=glide_ms,
        steady_ms=steady_ms,
        start_direction="up",
        ramp_ms=ramp_ms,
        target_rms=target_rms,
    )
    gap = silence(sr, isi_ms)

    if correct_interval == 1:
        x = np.concatenate([stim_down, gap, stim_up])
        order = "DOWN-UP"
    else:
        x = np.concatenate([stim_up, gap, stim_down])
        order = "UP-DOWN"

    return mono_to_stereo_bytes(x, sr, ear), int(correct_interval), order


# ============================================================
# Staircase
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
    trial_index: int = 0
    n_correct_streak: int = 0
    last_direction: Optional[str] = None  # "up" / "down"
    reversals: list[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.x_ms = float(self.start_ms)

    def current_step(self) -> float:
        return float(self.step_small_ms) if len(self.reversals) >= self.switch_after_reversals else float(self.step_big_ms)

    def phase(self) -> str:
        return "small" if len(self.reversals) >= self.switch_after_reversals else "big"

    def update(self, correct: bool) -> Dict[str, Any]:
        """
        2-down 1-up:
          - 2 consecutive correct -> decrease duration (harder)
          - 1 incorrect -> increase duration (easier)
        """
        self.trial_index += 1
        prev_x = float(self.x_ms)
        step = float(self.current_step())

        direction: Optional[str] = None
        if correct:
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
                    "trial": int(self.trial_index),
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
        }

    def threshold_ms(self, last_n: int = 6) -> Optional[float]:
        """
        threshold = mean of last_n reversal levels in the small-step phase
        (i.e., excluding the first `switch_after_reversals` reversals).
        """
        if len(self.reversals) < self.switch_after_reversals + last_n:
            return None
        usable = self.reversals[self.switch_after_reversals :]
        last = usable[-last_n:]
        return float(np.mean([r["level_ms"] for r in last]))

    def n_small_reversals(self) -> int:
        return max(0, len(self.reversals) - self.switch_after_reversals)


# ============================================================
# Derived helpers
# ============================================================
def sweep_rate_hz_per_s(delta_hz: float, glide_ms: float) -> float:
    """
    For triangular glide:
      delta reached at mid-glide (glide_ms/2), so slope magnitude = delta / (glide_ms/2 sec)
      = 2000 * delta / glide_ms  [Hz/s]
    """
    glide_ms = float(glide_ms)
    if glide_ms <= 0:
        return float("nan")
    return 2000.0 * float(delta_hz) / glide_ms


def format_threshold(th_ms: Optional[float], floor: float, ceil: float) -> str:
    if th_ms is None:
        return "—"
    if th_ms <= floor + 1e-9:
        return f"≤ {floor:.0f} ms"
    if th_ms >= ceil - 1e-9:
        return f"≥ {ceil:.0f} ms"
    return f"{th_ms:.1f} ms"


# ============================================================
# Session state
# ============================================================
def init_state():
    defaults = {
        "mode": "idle",  # idle | practice | test | finished
        "practice_n_done": 0,
        "practice_log": [],
        "test_log": [],
        "trial": None,  # current trial dict
        "awaiting_answer": False,
        "staircase": None,
        "test_trial_n": 0,
        "max_trials_allowed": 100,
        "threshold_live_ms": None,
        "threshold_final_ms": None,
        "finished_reason": None,  # "threshold" | "max_trials" | "manual"
        "started_at": None,
        # lock settings at block start (to prevent mid-run sidebar edits)
        "practice_settings": None,
        "test_settings": None,
        "results_tab": "本番タブ",
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
    delta = float(PRESETS[preset_name]["delta"])
    st.caption(f"中心周波数 f_center = **{f_center:.0f} Hz** / 偏移 Δf = **±{delta:.0f} Hz**")

    ear = st.radio("出力", ["両耳", "左耳のみ", "右耳のみ"], index=0)

    st.divider()
    st.subheader("刺激")
    sr = st.selectbox("サンプリング周波数", options=[44100, 48000], index=0)
    steady_ms = st.number_input("定常部 (ms)", min_value=50, max_value=500, value=200, step=10)
    isi_ms = st.number_input("A-B間ISI (ms)", min_value=200, max_value=1500, value=800, step=50)
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
    st.subheader("Staircase（duration ms）")
    start_ms = st.number_input("開始 D (ms)", min_value=20, max_value=800, value=300, step=10)
    floor_ms = st.number_input("D_min (ms)", min_value=5, max_value=200, value=20, step=5)
    ceil_ms = st.number_input("D_max (ms)", min_value=100, max_value=2000, value=500, step=50)

    step_big_ms = st.number_input("大ステップ (ms)", min_value=5, max_value=200, value=40, step=5)
    step_small_ms = st.number_input("小ステップ (ms)", min_value=1, max_value=100, value=20, step=1)
    switch_after = st.number_input("大→小 切替reversal数", min_value=1, max_value=10, value=4, step=1)

    max_trials = st.number_input("最大trial（上限=100）", min_value=20, max_value=100, value=100, step=5)
    practice_n = st.number_input("練習trial数", min_value=0, max_value=30, value=10, step=1)

    st.divider()
    if st.button("🧹 全リセット"):
        reset_all()
        st.rerun()


def snapshot_settings() -> Dict[str, Any]:
    """Freeze current sidebar parameters for a block (practice/test)."""
    return {
        "preset_name": preset_name,
        "f_center": float(f_center),
        "delta": float(delta),
        "ear": str(ear),
        "sr": int(sr),
        "steady_ms": int(steady_ms),
        "isi_ms": int(isi_ms),
        "ramp_ms": int(ramp_ms),
        "target_rms": float(target_rms),
        "start_ms": float(start_ms),
        "floor_ms": float(floor_ms),
        "ceil_ms": float(ceil_ms),
        "step_big_ms": float(step_big_ms),
        "step_small_ms": float(step_small_ms),
        "switch_after": int(switch_after),
        "max_trials": int(max_trials),
        "practice_n": int(practice_n),
    }


def ensure_staircase():
    if st.session_state["staircase"] is None:
        s = st.session_state.get("test_settings") or snapshot_settings()
        st.session_state["staircase"] = DurationStaircase(
            start_ms=float(s["start_ms"]),
            floor_ms=float(s["floor_ms"]),
            ceil_ms=float(s["ceil_ms"]),
            step_big_ms=float(s["step_big_ms"]),
            step_small_ms=float(s["step_small_ms"]),
            switch_after_reversals=int(s["switch_after"]),
        )


def start_practice():
    st.session_state["mode"] = "practice"
    st.session_state["practice_n_done"] = 0
    st.session_state["practice_log"] = []
    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["finished_reason"] = None
    st.session_state["threshold_live_ms"] = None
    st.session_state["threshold_final_ms"] = None
    st.session_state["started_at"] = time.time()
    st.session_state["practice_settings"] = snapshot_settings()
    st.session_state["results_tab"] = "練習タブ"


def start_test():
    st.session_state["mode"] = "test"
    st.session_state["test_log"] = []
    st.session_state["trial"] = None
    st.session_state["awaiting_answer"] = False
    st.session_state["threshold_live_ms"] = None
    st.session_state["threshold_final_ms"] = None
    st.session_state["finished_reason"] = None
    st.session_state["started_at"] = time.time()
    st.session_state["test_settings"] = snapshot_settings()
    st.session_state["results_tab"] = "本番タブ"

    s = st.session_state["test_settings"]
    st.session_state["max_trials_allowed"] = int(s["max_trials"])
    st.session_state["staircase"] = DurationStaircase(
        start_ms=float(s["start_ms"]),
        floor_ms=float(s["floor_ms"]),
        ceil_ms=float(s["ceil_ms"]),
        step_big_ms=float(s["step_big_ms"]),
        step_small_ms=float(s["step_small_ms"]),
        switch_after_reversals=int(s["switch_after"]),
    )
    st.session_state["test_trial_n"] = 0


def finish_block(reason: str):
    """Finish current block (practice/test) and show results."""
    st.session_state["mode"] = "finished"
    st.session_state["finished_reason"] = reason
    st.session_state["awaiting_answer"] = False
    st.session_state["trial"] = None
    st.session_state["results_tab"] = "結果サマリー"


def make_new_trial(mode: str):
    """Create and store a new trial in session_state['trial'] using locked settings."""
    settings = st.session_state.get("practice_settings") if mode == "practice" else st.session_state.get("test_settings")
    if not settings:
        settings = snapshot_settings()

    if mode == "practice":
        D_ms = int(round(float(settings["start_ms"])))
    else:
        ensure_staircase()
        D_ms = int(round(float(st.session_state["staircase"].x_ms)))

    wav, correct_interval, order = generate_trial_wav(
        sr=int(settings["sr"]),
        f_center=float(settings["f_center"]),
        delta=float(settings["delta"]),
        glide_ms=int(D_ms),
        steady_ms=int(settings["steady_ms"]),
        isi_ms=int(settings["isi_ms"]),
        ear=str(settings["ear"]),
        ramp_ms=int(settings["ramp_ms"]),
        target_rms=float(settings["target_rms"]),
        down_interval=None,
    )

    st.session_state["trial"] = {
        "wav": wav,
        "D_ms": int(D_ms),
        "correct_interval": int(correct_interval),
        "order": order,
        **settings,
        "created_at": time.time(),
    }
    st.session_state["awaiting_answer"] = True


def record_response(subject_id: str, response_interval: int):
    mode = st.session_state["mode"]
    trial = st.session_state.get("trial") or {}
    if not trial:
        return

    correct_interval = int(trial["correct_interval"])
    is_correct = int(response_interval == correct_interval)

    base_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subject_id": subject_id,
        "mode": mode,
        "trial_in_block": st.session_state.get("practice_n_done", 0) + 1
        if mode == "practice"
        else st.session_state.get("test_trial_n", 0) + 1,
        "D_ms_presented": int(trial["D_ms"]),
        "down_interval": correct_interval,
        "response": int(response_interval),
        "is_correct": is_correct,
        "order": trial.get("order"),
        "preset": trial.get("preset_name"),
        "f_center": trial.get("f_center"),
        "delta": trial.get("delta"),
        "sr": trial.get("sr"),
        "steady_ms": trial.get("steady_ms"),
        "isi_ms": trial.get("isi_ms"),
        "ramp_ms": trial.get("ramp_ms"),
        "target_rms": trial.get("target_rms"),
    }

    if mode == "practice":
        st.session_state["practice_log"].append(base_row)
        st.session_state["practice_n_done"] += 1
        st.session_state["awaiting_answer"] = False
        st.session_state["trial"] = None  # hide audio to discourage replay

        # auto-finish practice when completed -> back to idle (logs remain)
        ps = st.session_state.get("practice_settings") or snapshot_settings()
        if st.session_state["practice_n_done"] >= int(ps["practice_n"]):
            st.session_state["mode"] = "idle"
        return

    # test mode: update staircase
    ensure_staircase()
    sc: DurationStaircase = st.session_state["staircase"]
    upd = sc.update(bool(is_correct))
    st.session_state["test_trial_n"] = int(sc.trial_index)

    live = sc.threshold_ms(last_n=6)
    st.session_state["threshold_live_ms"] = live

    row = {
        **base_row,
        "D_ms_next": float(upd["new_x_ms"]),
        "direction": upd["direction"],
        "step_used_ms": upd["step_used_ms"],
        "phase": upd["phase"],
        "reversal": int(bool(upd["reversal"])),
        "reversal_level_ms": upd["reversal_level_ms"],
        "n_reversals": int(upd["n_reversals"]),
        "n_small_reversals": int(sc.n_small_reversals()),
        "threshold_live_ms": live,
    }
    st.session_state["test_log"].append(row)

    # clear trial (hide audio to discourage replay)
    st.session_state["awaiting_answer"] = False
    st.session_state["trial"] = None

    # stopping rules
    if live is not None:
        st.session_state["threshold_final_ms"] = live
        finish_block("threshold")
        return

    if sc.trial_index >= int(st.session_state["max_trials_allowed"]):
        finish_block("max_trials")


# ============================================================
# Top controls
# ============================================================
def _practice_target_n() -> int:
    ps = st.session_state.get("practice_settings") or snapshot_settings()
    return int(ps["practice_n"])


mode = st.session_state["mode"]
practice_target_n = _practice_target_n()

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.button(
        "🧪 練習を開始",
        disabled=(mode in ["practice", "test"]),
        on_click=start_practice,
    )
with c2:
    st.button(
        "🎯 本番を開始",
        disabled=(mode in ["practice", "test"]),
        on_click=start_test,
    )
with c3:
    st.button(
        "⏹️ 終了（結果表示）",
        disabled=(mode not in ["practice", "test"]),
        on_click=lambda: finish_block("manual"),
    )

st.divider()

# ============================================================
# Status metrics (always shown)
# ============================================================
sc: Optional[DurationStaircase] = st.session_state.get("staircase", None)

mcols = st.columns(6)
mcols[0].metric("mode", st.session_state["mode"])
mcols[1].metric("practice", f"{st.session_state['practice_n_done']}/{practice_target_n}")
mcols[2].metric("trial", f"{st.session_state.get('test_trial_n', 0)}")
mcols[3].metric("reversals", f"{len(sc.reversals) if sc else 0}")
mcols[4].metric("small rev", f"{sc.n_small_reversals() if sc else 0}")

live_ms = st.session_state.get("threshold_live_ms", None)
ts = st.session_state.get("test_settings") or snapshot_settings()
mcols[5].metric("暫定閾値", format_threshold(live_ms, float(ts["floor_ms"]), float(ts["ceil_ms"])))

st.caption(f"本番の最大trial: **{int(ts['max_trials'])}**（上限=100） / 練習trial: **{practice_target_n}**")

# ============================================================
# Main interaction (practice/test)
# ============================================================
if mode == "idle":
    st.info("上のボタンから **練習** または **本番** を開始してください。設定は左のサイドバーで変更できます。")

elif mode == "practice":
    ps = st.session_state.get("practice_settings") or snapshot_settings()
    st.subheader("🧪 練習")
    st.caption("練習は **固定D（開始D）**で実施し、フィードバックを表示します。")

    if st.session_state["practice_n_done"] >= int(ps["practice_n"]):
        st.success("練習が終了しました。上のボタンから本番を開始できます。")

    if not st.session_state["awaiting_answer"] and st.session_state["practice_n_done"] < int(ps["practice_n"]):
        if st.button("▶️ 提示（練習）"):
            make_new_trial("practice")
            st.rerun()

    trial = st.session_state.get("trial")
    if st.session_state["awaiting_answer"] and trial:
        st.audio(trial["wav"], format="audio/wav", autoplay=True)
        st.markdown("**質問**：どちらが **下がる音（DOWN）** でしたか？（1回目=1 / 2回目=2）")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("1"):
                resp = 1
                correct = int(trial["correct_interval"])
                record_response(subject_id, resp)
                if resp == correct:
                    st.success("正解")
                else:
                    st.error(f"不正解（正解は {correct}）")
                st.rerun()
        with b2:
            if st.button("2"):
                resp = 2
                correct = int(trial["correct_interval"])
                record_response(subject_id, resp)
                if resp == correct:
                    st.success("正解")
                else:
                    st.error(f"不正解（正解は {correct}）")
                st.rerun()

elif mode == "test":
    ts = st.session_state.get("test_settings") or snapshot_settings()
    st.subheader("🎯 本番（2AFC + staircase）")
    st.caption("本番ではフィードバックなし（結果は下のタブで常に確認できます）")

    if not st.session_state["awaiting_answer"]:
        if st.button("▶️ 提示（A→無音→B）"):
            make_new_trial("test")
            st.rerun()

    trial = st.session_state.get("trial")
    if st.session_state["awaiting_answer"] and trial:
        st.audio(trial["wav"], format="audio/wav", autoplay=True)
        st.markdown("**質問**：どちらが **下がる音（DOWN）** でしたか？（1回目=1 / 2回目=2）")

        a1, a2 = st.columns(2)
        with a1:
            if st.button("1", disabled=not st.session_state["awaiting_answer"]):
                record_response(subject_id, 1)
                st.rerun()
        with a2:
            if st.button("2", disabled=not st.session_state["awaiting_answer"]):
                record_response(subject_id, 2)
                st.rerun()

# ============================================================
# Results (always visible)
# ============================================================
st.divider()
st.subheader("📌 ログ・結果（常時表示）")

# タブ表示はrerunのたびに初期化されやすいので、session_stateで保持します。
# 本番中は「本番タブ」を固定表示（ボタンを押しても戻らない）にします。
if st.session_state["mode"] == "test":
    st.session_state["results_tab"] = "本番タブ"
elif st.session_state["mode"] == "practice":
    st.session_state["results_tab"] = "練習タブ"

active_results_tab = st.radio(
    "表示",
    ["練習タブ", "本番タブ", "結果サマリー"],
    key="results_tab",
    horizontal=True,
    label_visibility="collapsed",
)

if active_results_tab == "練習タブ":
    if st.session_state.get("practice_log"):
        df_pr = pd.DataFrame(st.session_state["practice_log"])
        acc = float(df_pr["is_correct"].mean()) * 100.0 if len(df_pr) else float("nan")
        st.write(f"正答率（練習）: **{acc:.1f}%**  （n={len(df_pr)}）")
        st.dataframe(df_pr, use_container_width=True, height=360)
        st.download_button(
            "CSVダウンロード（練習）",
            data=df_pr.to_csv(index=False).encode("utf-8-sig"),
            file_name="pitch_glide_direction_threshold_log_practice.csv",
            mime="text/csv",
        )
    else:
        st.caption("練習ログはまだありません。")

elif active_results_tab == "本番タブ":
    if st.session_state.get("test_log"):
        df_test = pd.DataFrame(st.session_state["test_log"])
        st.dataframe(df_test, use_container_width=True, height=360)
        st.download_button(
            "CSVダウンロード（本番）",
            data=df_test.to_csv(index=False).encode("utf-8-sig"),
            file_name="pitch_glide_direction_threshold_log_test.csv",
            mime="text/csv",
        )
    else:
        st.caption("本番ログはまだありません。")

else:  # 結果サマリー
    ts = st.session_state.get("test_settings")
    reason = st.session_state.get("finished_reason")
    final_ms = st.session_state.get("threshold_final_ms", None)
    live_ms = st.session_state.get("threshold_live_ms", None)

    # 本番が「終了」しているときのみまとめを出す
    if not ts or reason is None or not st.session_state.get("test_log"):
        st.caption("本番を実施して終了すると、ここに結果サマリーが表示されます。")
    else:
        ts = ts or snapshot_settings()
        df_test = pd.DataFrame(st.session_state["test_log"])
        n_trials = len(df_test)

        sc = st.session_state.get("staircase")
        n_rev = len(sc.reversals) if sc is not None and getattr(sc, "reversals", None) else 0
        n_small_rev = sc.n_small_reversals() if sc is not None else 0

        # --- outcome ---
        if reason == "threshold" and final_ms is not None:
            st.success(f"収束：推定閾値（duration） = {format_threshold(final_ms, float(ts['floor_ms']), float(ts['ceil_ms']))}")
            rate = sweep_rate_hz_per_s(float(ts["delta"]), float(final_ms))
            st.write(f"- 等価sweep rate（三角形グライド片側）: **{rate:.0f} Hz/s**")
            st.caption("※三角形グライドでは、最大偏移Δfに到達するのが D/2 なので rate=2000×Δf/D で換算しています。")
        elif reason == "max_trials":
            st.warning(f"最大trial（{int(ts['max_trials'])}）まで到達（未収束の可能性）")
            if live_ms is not None:
                st.write(f"- 暫定閾値: {format_threshold(live_ms, float(ts['floor_ms']), float(ts['ceil_ms']))}")
        elif reason == "manual":
            st.info("手動終了（未収束の可能性）")
            if live_ms is not None:
                st.write(f"- 暫定閾値: {format_threshold(live_ms, float(ts['floor_ms']), float(ts['ceil_ms']))}")

        st.markdown("#### 条件（本番）")
        st.write(f"- プリセット: **{ts['preset_name']}**  / f_center={float(ts['f_center']):.0f} Hz / Δf=±{float(ts['delta']):.0f} Hz")
        st.write(f"- 出力: {ts['ear']}  / SR={int(ts['sr'])} Hz  / 定常={int(ts['steady_ms'])} ms  / ISI={int(ts['isi_ms'])} ms")
        st.write(f"- ramp={int(ts['ramp_ms'])} ms / target_rms={float(ts['target_rms']):.2f}")
        st.write(f"- Staircase: start={float(ts['start_ms']):.0f} ms, floor={float(ts['floor_ms']):.0f} ms, ceil={float(ts['ceil_ms']):.0f} ms")
        st.write(f"- step_big={float(ts['step_big_ms']):.0f} ms（rev {int(ts['switch_after'])}回まで）, step_small={float(ts['step_small_ms']):.0f} ms")

        st.markdown("#### 実施状況")
        st.write(f"- trial数: **{n_trials}** / reversals: **{n_rev}** / small-step reversals: **{n_small_rev}**")

        if sc is not None and getattr(sc, "reversals", None):
            with st.expander("reversals（本番）", expanded=False):
                st.dataframe(pd.DataFrame(sc.reversals), use_container_width=True, height=260)
