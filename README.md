# Frequency Modulation (FM) Auditory Test for Streamlit

**Purpose**  
A browser-based application for testing **frequency modulation (FM) detection** at low and high modulation rates (e.g., **2 Hz** and **40 Hz**) using pure tones.  
The app is intended as a **simple clinical / research prototype** for assessing sensitivity to **slow and fast pitch fluctuations** (e.g., prosody-like vs. phoneme-like cues) and is designed to work on **iPhone/Safari** with **wired headphones**.

---

## 🌐 Live Demo

**Try the FM app here:**  
[https://dreamycat925-frequency-modul-fm-modulation-streamlit-app-7wrg4f.streamlit.app/](https://dreamycat925-frequency-modul-fm-modulation-streamlit-app-7wrg4f.streamlit.app/)

(If the app does not load, please check that Streamlit Community Cloud is running and your network allows external HTTPS connections.)

---

## Features

- Pure-tone carrier (default: **500 Hz**, adjustable)
- Adjustable **FM rate** (default: 2 Hz or 40 Hz via shortcut buttons, free range 0.5–100 Hz)
- Discrete **FM depth grid** matching typical psychophysical ranges:

  - `0.01, 0.02, 0.03, 0.04, 0.05,`
  - `0.06, 0.07, 0.08, 0.09, 0.10,`
  - `0.20, 0.30, 0.40, 0.50`

  Here, `depth ≈ Δf/f` (fraction of carrier frequency).  
  For a 500 Hz carrier:
  - depth 0.01 ≈ ±1% (495–505 Hz)
  - depth 0.02 ≈ ±2% (490–510 Hz)
  - …
  - depth 0.10 ≈ ±10% (450–550 Hz)
  - depth 0.50 ≈ ±50% (250–750 Hz; very large, for practice/heavy impairment)

- Single-tone playback:
  - **“FMなし（フラット）”**: unmodulated reference tone
  - **“FMあり（変調）”**: FM tone at the current depth/rate
  - **“ランダム”**: either FMあり or FMなし, randomized each time
- Fully synthesized in Python/NumPy and served via `st.audio` (WAV); no external audio server is required
- Works in modern desktop and mobile browsers, including **iPhone/Safari**, as long as wired headphones are used

> Note: Unlike the Click Fusion Test app, this FM app currently does **not** include built-in CSV logging.  
> Response logging and staircase control are intended to be done on paper or in a separate spreadsheet.

---

## Demo

> (You can insert screenshots or GIFs here, e.g., sidebar with FM rate/depth and the three playback buttons.)

---

## Quick Start Guide

1. **Hardware / environment**
   - Use **wired, closed-back headphones**.  
     Bluetooth and speakers are not recommended due to latency and potential distortion.
   - Test in a **quiet room**.

2. **Basic settings**
   - Open the live app URL in a modern browser.
   - In the sidebar, leave defaults or set:
     - Sampling rate: e.g., **44,100 Hz**
     - Carrier: **500 Hz**
     - Duration: **1,000 ms**
   - Use the shortcut buttons to set the FM rate:
     - `2 Hz` button → slow, prosody-like fluctuation
     - `40 Hz` button → faster, rougher modulation (more phoneme-like)

3. **FM depth selection**
   - Choose a depth from the discrete list:
     - For **practice / demonstration**: start with **0.30–0.50** (very obvious wobble)
     - For **clinical / research testing**: use **0.01–0.10**, especially **0.02–0.05**

4. **Playing stimuli**
   - Use the three buttons in the main area:
     - **FMなし（フラット）**: play reference tone
     - **FMあり（変調）**: play modulated tone
     - **ランダム（一発）**: play either FMあり or FMなし (random); the app displays which one was presented (“FMあり”/“FMなし”) for the examiner only

5. **Patient instructions (example, Japanese)**
   > 「これから『ピー』という音を聞いていただきます。  
   > まっすぐな音と、少し“揺れている音”が出ます。  
   > 今の音は揺れていましたか？ それとも、まっすぐでしたか？」

   The examiner can record the patient’s responses and compare them to the “last random stimulus” label shown on screen.

---

## Recommended Clinical Implementation (Prototype)

These are suggested settings for a **simple, clinically usable protocol**, inspired by FM detection literature and core auditory processing studies in PPA:

### 1. Practice

- **FM rate**: 2 Hz (slow)
- **Depth**: 0.30–0.50  
  - Alternate **FMなし** and **FMあり** to demonstrate the “wobbling” sensation.
  - Ensure the patient reliably understands what “揺れている音” means.

### 2. Screening at suprathreshold

- **FM rate**: 2 Hz and 40 Hz (both tested)
- **Depth**: 0.10 (±10%)  
- Procedure:
  - Use the **ランダム** button for ~20 trials at depth 0.10.
  - Ask on each trial: “今の音は揺れていましたか？”（yes/no or “揺れている/いない”）
  - A rough rule of thumb:
    - ≥ 80% correct → FM detection at this depth is likely intact.
    - < 60–70% correct → consider increasing depth (e.g., 0.20) and repeating, or suspect reduced FM sensitivity.

### 3. Rough threshold estimation (simple mini-staircase)

For more detailed assessment (e.g., in research or advanced clinical use):

- **FM rate**:  
  - 2 Hz → slow modulation (prosodic / TFS-like)  
  - 40 Hz → faster modulation (roughness / phoneme-like)
- **Depth levels** (example grid):  
  `0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10`
- Start at an easy depth (e.g., 0.10), then:

  - If the patient answers correctly (FMあり vs なし) → move **one step down** (smaller depth)
  - If the patient answers incorrectly → move **two steps up** (larger depth)

- Stop after **2 reversals** or ~10–12 trials per condition.
- The depth around the last reversal(s) can be taken as a **rough FM detection zone**:
  - e.g., “2 Hz FM ≈ depth 0.02–0.03”, “40 Hz FM ≈ depth 0.04–0.06”.

> For routine clinical work, this “rough zone” is often sufficient;  
> for formal psychophysics, a more rigorous staircase (e.g., 2-down 1-up with 6–8 reversals) would be needed.

---

## Local Installation

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run fm_modulation_streamlit_app.py
