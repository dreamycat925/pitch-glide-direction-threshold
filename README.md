# Pitch Glide Detection Threshold Test

This is a **single-interval** **Pitch Glide / Pitch Change Detection Threshold Test** implemented in Streamlit.  
On each trial, only **one sound** is presented, and the participant answers whether it was **flat (FLAT)** or **changed in pitch (GLIDE)**.

This app estimates the **minimum glide duration D (ms)** at which a **GLIDE** can be detected, using a staircase procedure.

> **Intended use**: research, prototyping, and evaluation  
> **Note**: This is not a medical device. Results are affected by headphones, playback device, browser, ambient environment, and volume settings.

> **Operational assumption**: The examiner operates the screen, while the participant responds verbally without looking at it. The examiner enters the response using the buttons.
> If a response is entered incorrectly, the examiner can press **Back one trial** to restore the immediately previous trial and re-enter the answer.

---

## What This App Can Do

- Perform **FLAT / GLIDE judgments** using single-interval presentation
- Run a fixed **100-trial** test session (**FLAT = 40 / GLIDE = 60**)
- Select from **Series 1 / Series 2 / pseudo-random** trial schedules
- Update the staircase on **GLIDE trials only**
- Adjust **D (glide duration in ms)** using a **2-down 1-up** rule
- Show not only the **official threshold**, but also a **reference value** when 100 trials end before full convergence
- Display a **convergence line chart** in the Summary
  - **Blue diamonds** = big-step reversals
  - **Red diamonds** = small-step reversals
- Show the **audio player and response buttons from the beginning** in both practice and test modes
  - There is **no separate “Present” button**
  - Audio is played from the upper section, and the examiner answers **GLIDE / FLAT** in the lower section
- Show a **Back one trial** button below the response buttons
  - Restores the immediately previous trial so the examiner can correct a mistyped response
  - Can be used repeatedly with no preset limit
  - Remains available even after **PASS / FAIL / convergence / manual stop**
- Download **practice logs / test logs** as CSV
- Download a **Summary .txt** file containing the result summary and test conditions

---

## Test Concept

### 1. Stimulus format

Each trial presents **only one sound**. The participant gives one of two responses:

- **Change present (GLIDE)**
- **No change (FLAT)**

This design uses single-interval presentation rather than two-interval comparison (2AFC), which can reduce comparison load.

### 2. GLIDE and FLAT

- **GLIDE**
  - A linear frequency ramp with monotonic frequency change is presented
  - Direction is `up` or `down`
  - Ramp length is **D (ms)**
  - After the ramp, an optional steady portion of **steady_ms** can be added
- **FLAT**
  - A steady tone with no frequency change
  - Its duration is matched to GLIDE using **total_ms = D + steady_ms**

### 3. Threshold

This app estimates **GLIDE duration D (ms)**.  
In other words, it measures **how long a glide must be for the participant to detect the pitch change**.

---

## Stimulus Specifications

### Frequency presets

- **1240 Hz version (around the F2 region: 900–1580 Hz)**
  - `f_center = 1240 Hz`
  - Default `Δf = ±340 Hz`
- **500 Hz version (low-frequency range: 350–650 Hz)**
  - `f_center = 500 Hz`
  - Default `Δf = ±150 Hz`

`Δf` can be changed from the sidebar.  
However, it must satisfy **Δf < f_center** so that the starting frequency never falls to 0 Hz or below.

### Output

- Binaural
- Left ear only
- Right ear only

### Audio generation

- Sampling frequency: **fixed at 48,000 Hz**
- Fade: cosine ramp
- RMS normalization: uses `target_rms`
- GLIDE uses a **linear ramp toward the center frequency**
  - `up`: `(f_center - Δf) → f_center`
  - `down`: `(f_center + Δf) → f_center`
- `steady_ms = 0` is also allowed

---

## Demo Audio (for listening)

At the top of the screen, demo audio is provided for explaining the rule.

- **Change present (UP)**
- **Change present (DOWN)**
- **No change (FLAT)**

Specifications are as follows:

- Demo glide duration is fixed at **D = 300 ms**
- FLAT is presented with a duration matched to `D + steady_ms`
- **It is not recorded in the logs**
- It does not affect the staircase or the results
- It cannot be played during practice or test sessions

---

## Test Trial Schedule

### Fixed schedules

The test session is fixed at **100 trials**.

- **FLAT = 40 trials**
- **GLIDE = 60 trials**

You can select one of the following fixed schedules:

- **Series 1**
- **Series 2**

### Pseudo-random

A pseudo-random schedule is also available.

- Exact counts: `1×40`, `2×60`
- `1 = FLAT`, `2 = GLIDE`
- Maximum run length of the same value: **3**
- A seed can be specified
- The schedule is fixed when the test starts, and the reproducibility information is saved in the Summary

### GLIDE direction schedule

The GLIDE direction sequence is also fixed for the test session.

- `1 = up`
- `2 = down`
- Total of 60 items
- **up = 30 / down = 30**

For fixed schedules, the direction sequence is also fixed. For pseudo-random schedules, it is fixed based on the seed.

---

## Staircase Specifications

### What gets updated

The staircase is updated on **GLIDE trials only**.  
FLAT trials are used to reduce expectancy effects and to check false alarms, but they do not update D.

### Rule

The app uses **2-down 1-up (signal-only)**.

- **2 consecutive HITs** on GLIDE trials → decrease D (make it harder)
- **1 MISS** on a GLIDE trial → increase D (make it easier)

### Step sizes

- **Big step**: used until a certain number of reversals is reached
- **Small step**: used after that

Default values:

- Starting D: `300 ms`
- D_min: `20 ms`
- D_max: `600 ms`
- Big step: `40 ms`
- Small step: `20 ms`
- Reversal count for switching big → small: `4`

### Official threshold and reference values

The content shown in the Summary depends on the number of small-step reversals.

- **0–1**: no reference value
- **2**: reference value (median) + reference range (min–max)
- **3–5**: reference value (provisional median)
- **6 or more**: official threshold

The official threshold is defined as the **median of the last 6 reversals in the small-step phase**.  
The mean is also shown as a reference.

---

## Stop Criteria

The test ends when any of the following conditions is met:

- **6 small-step reversals** have been collected
- **2 consecutive MISS responses at D_max** (GLIDE trials)
- **4 consecutive HIT responses at D_min** (GLIDE trials)
- **100 trials reached**
- **Manual stop**

The Summary also shows the reason for stopping.

---

## Practice Mode

Practice is optional.

- FLAT / GLIDE are randomized at **50/50**
- GLIDE duration D is fixed using **Practice D (ms)** in the sidebar
- GLIDE direction is randomized only in practice mode
- Correct/incorrect feedback is shown
- Only **GLIDE HITs** count toward the consecutive-correct streak

The default is **600 ms**. If needed, it can be changed to a more difficult condition such as **200 ms**.  
This setting is **independent from the staircase used in the test** and is fixed when practice starts.

If `Target 5 consecutive HITs (GLIDE) in practice` is enabled, practice ends automatically after **5 consecutive HITs**.  
FLAT trials do not affect this streak count.

---

## Layout of the Test Screen

The practice and test trial screens are displayed in the following order:

1. **Top section**: audio player
2. **Middle section**: question text
3. **Bottom section**: response buttons for **Change present (GLIDE) / No change (FLAT)**

### Key points

- There is **no “Present” button**
- Each new trial is prepared automatically
- The participant listens to the sound using the **play button above**, then answers
- In practice mode, correct/incorrect feedback is shown after each response
- In test mode, no correctness feedback is shown

---

## Information Displayed in the Summary

After the test ends, the **result summary** shows the following:

- Number of trials
- Accuracy
- HIT rate (GLIDE)
- FA rate (FLAT)
- Official threshold or reference value
- Number of small-step reversals
- Response counts (HIT / MISS / FA / CR)
- Stop condition
- Snapshot of test settings
- Trial schedule / seed / GLIDE direction schedule used
- Reversals table

### Convergence line chart

The Summary includes a line chart for visually checking convergence.

- Vertical axis: presented **D (ms)**
- Horizontal axis: **number of GLIDE presentations, not total trial count**
- Reversal points are overlaid
  - **Blue diamonds** = big-step reversals
  - **Red diamonds** = small-step reversals
- If an official threshold or reference value exists, a horizontal line is also shown

---

## Downloadable Files

### 1. Practice log CSV

- File name: `pitch_glide_practice_log.csv`

### 2. Test log CSV

- File name: `pitch_glide_test_log.csv`

### 3. Summary .txt

You can download a single `.txt` file containing:

- Result summary
- Threshold / reference value
- Test conditions
- Trial schedule used
- Reversals

---

## Running Locally

### 1. Run directly with Python

```bash
pip install -r requirements.txt
streamlit run pitch-glide-direction-threshold.py
```

### 2. Run with Docker / Docker Compose

If you want to use Docker, see `README_DOCKER.md`.

---

## Notes

- Absolute sound pressure level (dB SPL) is not calibrated
- Audio behavior may differ slightly depending on the browser or operating system
- Bluetooth is not recommended for threshold measurement because of latency and internal processing
- If used in a clinical context, record the playback device, headphones, environment, and volume conditions
