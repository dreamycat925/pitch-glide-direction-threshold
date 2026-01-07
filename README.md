# Pitch Glide Direction Threshold Test (PGDT) for Streamlit

**Purpose**  
A browser-based application for estimating the **direction-discrimination threshold** for a **pitch glide** (frequency glide / tone sweep).  
The test adaptively finds the **minimum glide duration (ms)** required to judge which interval contains a **DOWN glide** (initially descending), i.e., the ability to track **rapid pitch transitions** that may contribute to **environmental sound / melody perception**.

This app is intended as a **clinical / research prototype** (not a medical device).

---

## 🌐 Live Demo

**Try the PGDT app here:**  
`<PUT_YOUR_STREAMLIT_CLOUD_URL_HERE>`

(If the app does not load, Streamlit Community Cloud may be sleeping; open the URL once to wake it up.)

---

## Features

- **Two frequency presets** (selectable):
  - **1240 Hz preset (F2-band-like)**: `f_center=1240 Hz`, `Δf=±340 Hz` (900–1580 Hz)
  - **500 Hz preset (low-frequency)**: `f_center=500 Hz`, `Δf=±150 Hz` (350–650 Hz)
- **Output**: diotic (both ears), left-only, or right-only
- **Stimulus** (per interval):
  - **Triangular pitch glide** (starts at `f_center`, moves to `f_center±Δf`, returns to `f_center`)
  - + **steady-state** segment (default 200 ms)
  - **Cosine ramp** (default 10 ms), **RMS normalization** (default target 0.10)
- **Task** (2AFC): A → (ISI) → B  
  Prompt: **“Which interval was the DOWN glide? 1 or 2”**
- **Practice block**: default **10 trials**, with feedback
- **Adaptive threshold (staircase)**:
  - **2-down 1-up** on **glide duration D (ms)** (targets ~70.7%)
  - defaults: `start=300 ms`, `D_min=20 ms`, `D_max=500 ms`
  - steps: **40 ms** until the **4th reversal**, then **20 ms**
  - **Stop rule**: threshold defined when **6 small-step reversals** are collected
  - **Max trials**: up to **100**
- **Counterbalanced answer position**:
  - Selectable: **Series 1 / Series 2 / Series 3 / Fully random**
  - Series 1–3 are fixed 1–50 trial sequences (from the paper record sheet)
  - For >50 trials, the series repeats from the beginning (51=1, 52=2, ...)
- **Logging & export**:
  - Practice log CSV
  - Test log CSV
  - Result summary after the **test block is finished**
- Fully synthesized in Python/NumPy and served via `st.audio` (WAV); **no external audio server** required

---

## Quick Start Guide

### 1) Hardware / environment
- Use **wired, closed-back headphones** (recommended).
  - **Bluetooth is not recommended** (latency / compression / unpredictable level).
- Test in a **quiet room**.
- Keep device + headphones consistent for reference data.

> ⚠️ **Safety**: start at a comfortable listening level. This app does **not** calibrate absolute dB SPL.

### 2) Basic settings (sidebar)
- Choose **Preset**: 1240 Hz or 500 Hz
- Choose **Sequence**: Series 1/2/3 (recommended for validation) or Random
- Leave defaults unless you have a reason to change:
  - Sampling rate: 44,100 Hz
  - ISI: 800 ms
  - Steady segment: 200 ms
  - Ramp: 10 ms
  - Staircase: start 300 ms, min 20 ms, max 500 ms, step 40→20, switch after 4 reversals
  - Max trials: 100

> Note: Settings are **snapshotted at the start of each block** (practice/test).  
> If you change settings mid-block, they won’t affect the running block. Use reset if needed.

### 3) Practice block (examiner-operated)
- Click **Start Practice**
- On each trial:
  1. Play the stimulus (A→ISI→B)
  2. Ask the patient which interval was **DOWN**
  3. The examiner clicks **1** or **2**
- Practice includes feedback and is useful to ensure task understanding.

### 4) Test block
- Click **Start Test**
- Run until:
  - 6 small-step reversals are collected (threshold computed), or
  - max trials (100) reached (marked as not converged)

### 5) Result summary & CSV
- After the test ends, open **Result Summary**:
  - Final threshold **D (ms)**
  - Convergence / stop reason
  - Optional converted sweep-rate metric
- Download CSV logs for audit/tracking.

---

## Patient instructions (example, Japanese)

> 「これから2回、短い音を続けて流します。  
> どちらか一方は、最初に“下がる”感じの音が入っています。  
> “下がる音”が **1回目**なら『1』、**2回目**なら『2』と教えてください。」

(Examiner enters the response on the PC. The patient does not need to touch the device.)

---

## Scoring

### Primary outcome
- **Threshold (ms)** = mean of the **last 6 reversals during the small-step phase**

### Optional secondary metric: sweep rate (Hz/s)
Because the glide is triangular, the instantaneous slope during the first half is:

- **sweep_rate (Hz/s) = 2000 × Δf / D**

where:
- `Δf` is the preset frequency deviation (1240 preset: 340 Hz; 500 preset: 150 Hz)
- `D` is the glide duration in ms

---

## Local Installation

```bash
git clone https://github.com/<you>/<pitch-glide-threshold-test>.git
cd <pitch-glide-threshold-test>
pip install -r requirements.txt
streamlit run pitch_glide_direction_threshold_streamlit_app.py
