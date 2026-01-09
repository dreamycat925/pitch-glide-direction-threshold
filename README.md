# Pitch Glide (Pitch Change Detection) Threshold Test for Streamlit
## (Single‑interval FLAT/GLIDE, Duration Staircase, Series 1–2, CSV export)

**Purpose**  
A browser-based application for estimating the **pitch‑change detection threshold** for a **pitch glide** (frequency glide / tone sweep).
This version is designed for patient-friendly operation: it uses a **single‑interval** task (one stimulus per trial) and asks whether the pitch was **flat** or **changed (glide)**.

The test adaptively estimates the **minimum glide duration (ms)** required to reliably detect a pitch change, which may relate to
rapid pitch transition processing relevant to **environmental sound perception / melody perception**.

> ⚠️ **Not a medical device.**  
> Results depend on headphone calibration, listening environment, and device output. Use for research/prototyping only.

---

## 🌐 Live Demo

Try the app here (if deployed):  
`<PUT_YOUR_STREAMLIT_CLOUD_URL_HERE>`

(If the app does not load, Streamlit Community Cloud may be sleeping; open the URL once to wake it up.)

---

## What this version implements

### Stimuli
- **Two frequency presets** (selectable):
  - **1240 Hz preset (F2-band-like)**: `f_center=1240 Hz`, `Δf=±340 Hz` (900–1580 Hz)
  - **500 Hz preset (low-frequency)**: `f_center=500 Hz`, `Δf=±150 Hz` (350–650 Hz)
- **Output**: diotic (both ears), left-only, or right-only
- **Stimulus per trial** (single interval):
  - **GLIDE**: triangular pitch glide lasting `glide_ms`
    - `f_center → f_center±Δf → f_center` (up/down is randomized for variety)
    - followed by **steady** segment at `f_center` (default `steady_ms`)
  - **FLAT**: steady tone only with **matched total duration**
    - total duration is matched to GLIDE: `total_ms = glide_ms + steady_ms`
- **Cosine ramp** (default `ramp_ms`)
- **RMS normalization**: performed on the **audible waveform (actual sound segment)** for each stimulus
- **Sampling rate**: **48,000 Hz (fixed)**

> ⚠️ Safety: start at a comfortable level. This app does **not** calibrate absolute dB SPL.

---

## Task (single‑interval; detection)
Each trial plays **one stimulus** (FLAT or GLIDE).

Participant answers:
- **「変化あり（GLIDE）」** or **「変化なし（平坦＝FLAT）」**

This avoids the cognitive load of 2‑interval comparison (2AFC) and is intended to be more robust for patients with reduced attention.

### Example patient instructions (Japanese)
> 「いまの音は、**高さが変化**しましたか？  
> 変化したと思ったら『変化あり』、変化しなかったら『変化なし』と答えてください。」

---

## Trial schedule (Series 1 / Series 2)

To standardize sessions and support validation/retest comparisons, the main test uses **fixed schedules**:

- Total: **100 trials**
- **40 trials = FLAT**
- **60 trials = GLIDE**
- **Default**: **Series 1**
- Option: **Series 2**

Notation:
- `1 = FLAT`
- `2 = GLIDE`

### Series 1 (default)
```text
[2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2]
```

### Series 2
```text
[2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2]
```

---

## Adaptive threshold (staircase)

### Staircase rule (2‑down 1‑up; **GLIDE trials only**)
The adaptive parameter is the **glide duration D (ms)**.

- Staircase updates are applied **only on GLIDE trials** (signal trials).
- **Two consecutive HITs on GLIDE** → duration decreases (harder)
- **One MISS on GLIDE** → duration increases (easier)

FLAT trials are included to reduce expectancy and to estimate **false alarms**, but they do not update the staircase.

### Step sizes
- **Big step** until **4 reversals**
- **Small step** after that

### Threshold definition
- Use only **small-step phase reversals**
- **Threshold = median of the last 6 small-step reversals** (ms)

*(Optionally, the mean of the last 6 reversals can also be reported for reference.)*

---

## Early stopping rules (patient-friendly)
The test can stop before 100 trials if any criterion is met:

- **Ceiling stop**: at `D_max`, **2 consecutive MISS** on GLIDE
- **Floor stop**: at `D_min`, **4 consecutive HIT** on GLIDE
- **Reversal stop**: collect **6 small-step reversals**
- **Manual stop**: a dedicated **終了** button is available

---

## Practice (optional)
- GLIDE/FLAT are presented at **50/50**
- Uses an easy duration (typically `D_max`) and provides **feedback**
- Practice ends when the participant achieves **5 consecutive HITs on GLIDE trials**
  - (The streak counter is updated only on GLIDE trials)

---

## Reliability / interpretation notes
- **FA (false alarm)**: answering “変化あり” on a FLAT trial
  - Useful as an attention / response-bias quality metric
  - Cutoffs should be established using normative data
- Output is not SPL-calibrated; consider recording:
  - device model, OS/browser, headphone model, environment, volume setting

---

## Logging / Export
- Practice and test logs are shown in the app
- CSV download buttons:
  - `pitch_glide_practice_log.csv`
  - `pitch_glide_test_log.csv`
- Result summary includes:
  - threshold (ms) and last 6 reversal levels used
  - accuracy, HIT/MISS/FA/CR counts
  - reversal counts, stop reason
  - snapshot of test settings

---

## Local Installation

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run pitch-glide-direction-threshold.py
```

---

## Limitations
- Not SPL‑calibrated
- Browser audio behavior varies (especially on iOS)
- Bluetooth audio is not recommended for clinical/research threshold work due to latency/processing variability
