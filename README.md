# NeuroSafetyGuard

> **Real-Time Cognitive Overload Detection in Construction Workers via Multimodal fNIRS–EEG Fusion and Adaptive Safety Interventions**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

---

## Research Statement

**"Can fused neurophysiological signals (fNIRS + EEG) encoded through a lightweight temporal attention network enable real-time, personalized cognitive overload detection in construction workers, and can that detection trigger closed-loop adaptive safety interventions that measurably reduce unsafe behavior in high-pressure VR-simulated work zones?"**

### Motivation

Construction fatalities remain stubbornly high (~1,000/year in the US alone). Prior work by Pooladvand & Hasanzadeh (2022, 2023, 2024) demonstrated that cognitive tunneling, time pressure, and peer influence each independently degrade workers' hazard-perception and decision-making. These studies used single-modal neuroimaging (fNIRS or EEG) in offline post-hoc analyses. A critical gap remains: **no system closes the loop** between real-time brain-state estimation and an on-site adaptive intervention that immediately modifies the worker's environment or task before an incident occurs.

### Core Hypothesis

Fusing the complementary strengths of fNIRS (hemodynamic, higher spatial resolution) and EEG (electrical, millisecond temporal resolution) through a cross-modal attention mechanism will outperform single-modality classifiers in detecting four safety-critical cognitive states — *normal, elevated-load, cognitive-tunneling, and fatigue* — with enough precision to trigger targeted micro-interventions (audio alerts, AR overlays, task-pause signals) that reduce unsafe behavior rates by ≥ 20 % compared to no-intervention baselines in a VR construction-site simulation.

---

## Literature Survey

### 1. Cognitive States & Construction Safety

| Study | Key Finding | Method |
|-------|------------|--------|
| Pooladvand & Hasanzadeh, *Autom. Constr.* (2022) | Time pressure + high mental demand worsen risk-compensation; neural arousal measurable via fNIRS | fNIRS + MR simulation |
| Pooladvand & Hasanzadeh, *JCEM* (2023) | Stress induces cognitive tunneling and impairs selective attention; 2023 Editorial Choice Paper | fNIRS + VR |
| Pooladvand, Chang & Hasanzadeh, *Autom. Constr.* (2024) | fNIRS-based mental load classification using ML identifies at-risk workers in mixed reality | fNIRS + MR + ML |
| Lee, Pooladvand et al., *J. Comput. Civil Eng.* (2024) | Neurophysiological signatures distinguish risk-perception *processes*, not just outcomes | EEG + fNIRS + VR |
| Pooladvand et al., Conf. Paper (2024) | fNIRS functional connectivity differentiates novice vs. experienced driver inhibitory control in work-zone intrusion scenarios | fNIRS functional connectivity |

### 2. Multimodal fNIRS–EEG Fusion

- **Complementarity**: EEG captures fast event-related potentials (ERPs, alpha/theta suppression) in <100 ms; fNIRS captures slow hemodynamic BOLD-like responses (peak ~6 s). Fusing both yields temporal + spatial richness unavailable from either alone (Li et al., *IEEE TBME*, 2020; Han et al., *Front. Public Health*, 2023).
- **Fusion architectures**: Early fusion (concatenated features), late fusion (ensemble decisions), and *hybrid cross-modal attention* are the three dominant paradigms (Multimodal MBC-ATT, *Front. Hum. Neurosci.*, 2025). Cross-modal attention consistently outperforms both early and late fusion on emotion/cognitive-state tasks.
- **Construction-domain gap**: Virtually all fNIRS–EEG fusion work is in clinical or lab emotion-recognition settings. Translating to construction (noisy, motion-artifact-prone, ecological validity needed) requires domain-specific signal cleaning and lightweight architectures deployable on edge hardware.

### 3. Hazard Recognition & Intervention

- Eye tracking + fNIRS reveal whether workers fixate hazards and which cognitive stage (attention vs. recognition vs. decision) breaks down (Hasanzadeh et al., 2017; Jeon & Cai, 2021).
- Smart helmets with EEG have shown promise for fatigue monitoring in driving and heavy equipment operation (Bitbrain, 2026 review; Ji, 2023).
- **No closed-loop system** in construction context exists that (a) classifies cognitive state in real time, then (b) triggers a graded intervention hierarchy — a clear open research frontier.

### 4. Adaptive Intervention Design

- Auditory alerts reduce missed hazards by ~30 % in simulated environments (NIOSH, 2021).
- AR overlays that highlight hazard zones improve safe-behavior compliance without increasing cognitive load when overlay density is kept low (Sacks et al., 2020).
- Task-pause signals ("micro-break prompts") reduce error rates in high-demand periods (ISO 9241-210 human-centered design standard).

### 5. Gaps Addressed by This Work

| Gap | How This Study Addresses It |
|-----|----------------------------|
| Single-modality classification | fNIRS + EEG cross-modal fusion |
| Offline / post-hoc analysis | Streaming real-time inference pipeline |
| No intervention feedback | Closed-loop VR intervention controller |
| Fixed (non-personalized) thresholds | Individual calibration with online threshold adaptation |
| Lab-only validation | VR construction scenario replicating Pooladvand's electrical-work paradigm |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION LAYER                       │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  fNIRS Cap   │    │   EEG Headset│    │  Eye Tracker /  │   │
│  │ (e.g. Kernel │    │ (e.g. Emotiv │    │  Wristband HR   │   │
│  │  Flow Pro)   │    │  EPOC Flex)  │    │  (peripheral)   │   │
│  └──────┬───────┘    └──────┬───────┘    └────────┬────────┘   │
└─────────┼──────────────────┼─────────────────────┼────────────┘
          │  LSL streams      │                     │
┌─────────▼──────────────────▼─────────────────────▼────────────┐
│                  SIGNAL PROCESSING LAYER                        │
│  fNIRS: motion artifact removal (MARA), MBLL, z-score norm     │
│  EEG:  bandpass (1–40 Hz), ICA artifact rejection, epoching    │
│  Sync: timestamp alignment via Lab Streaming Layer (LSL)       │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│              CROSS-MODAL ATTENTION CLASSIFIER                   │
│                                                                 │
│  ┌─────────────┐    ┌──────────────────────────────────────┐   │
│  │ fNIRS       │    │  Cross-Modal Temporal Attention Net   │   │
│  │ Temporal    ├───►│  (CMTAN)                             │   │
│  │ CNN encoder │    │  • fNIRS query, EEG key/value        │   │
│  └─────────────┘    │  • Sliding window: 5-s, 1-s stride   │   │
│  ┌─────────────┐    │  • Output: 4-class cognitive state    │   │
│  │ EEG Band-   ├───►│    [Normal | High-Load |             │   │
│  │ power LSTM  │    │     Tunneling | Fatigue]             │   │
│  └─────────────┘    └──────────────────┬─────────────────── ┘  │
└──────────────────────────────────────┬─┴──────────────────────┘
                                       │ state + confidence
┌──────────────────────────────────────▼──────────────────────────┐
│              ADAPTIVE INTERVENTION CONTROLLER                    │
│                                                                  │
│  Confidence ≥ 0.7 → Graded intervention:                        │
│    • High-Load   → subtle audio chime + AR hazard highlight      │
│    • Tunneling   → AR peripheral flash + voice "check surrounds" │
│    • Fatigue     → task pause prompt + supervisor alert          │
│                                                                  │
│  Personalization: rolling z-score baseline per worker           │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────┐
│                    VR SIMULATION LAYER                           │
│         (Unity + Varjo XR / Meta Quest Pro)                     │
│  Electrical-line construction task, varied time pressure         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
neuro_safety_guard/
├── configs/
│   ├── default.yaml          # Hyperparameters, thresholds
│   └── experiment.yaml       # VR scenario configs
├── src/
│   ├── data/
│   │   ├── lsl_streamer.py   # Real-time LSL data ingestion
│   │   ├── preprocessor.py   # fNIRS + EEG preprocessing pipeline
│   │   └── dataset.py        # PyTorch dataset / dataloader
│   ├── models/
│   │   ├── fnirs_encoder.py  # Temporal CNN for fNIRS
│   │   ├── eeg_encoder.py    # Band-power LSTM for EEG
│   │   ├── cmtan.py          # Cross-Modal Temporal Attention Network
│   │   └── classifier.py     # 4-class softmax head
│   ├── utils/
│   │   ├── metrics.py        # F1, confusion matrix, latency
│   │   └── personalizer.py   # Per-worker baseline calibration
│   └── viz/
│       ├── dashboard.py      # Real-time Streamlit dashboard
│       └── plots.py          # Offline analysis figures
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── tests/
│   ├── test_preprocessor.py
│   └── test_cmtan.py
├── docs/
│   ├── literature_survey.md  # Extended lit review
│   └── experiment_protocol.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/neuro_safety_guard.git
cd neuro_safety_guard
pip install -e .

# 2. Simulate fake data and run the full pipeline
python -m src.data.dataset --demo

# 3. Train the CMTAN classifier
python -m src.models.classifier --config configs/default.yaml --mode train

# 4. Launch the real-time dashboard (connects to LSL streams)
streamlit run src/viz/dashboard.py

# 5. Run tests
pytest tests/
```

---

## Expected Outcomes

| Metric | Target |
|--------|--------|
| 4-class cognitive state F1 | ≥ 0.80 |
| Inference latency | < 500 ms (5-s sliding window) |
| Unsafe behavior rate reduction (VR) | ≥ 20 % vs. no-intervention |
| False positive alert rate | < 10 % |

---

## Alignment with Dr. Pooladvand's Research Program

This project directly extends the **fNIRS-based mental load classification** work (Pooladvand et al., 2024) by:
1. **Adding EEG modality** via cross-modal attention fusion
2. **Closing the loop** with adaptive VR interventions
3. **Personalizing thresholds** per worker to handle inter-individual variability
4. **Open-sourcing the pipeline** for broader construction-safety community use

---

## Citation

If you use this work, please cite:
```bibtex
@misc{neuro_safety_guard_2025,
  title  = {NeuroSafetyGuard: Real-Time Cognitive Overload Detection via fNIRS-EEG Fusion for Construction Safety},
  author = {[Your Name]},
  year   = {2025},
  note   = {Research prototype, Arizona State University}
}
```

---
