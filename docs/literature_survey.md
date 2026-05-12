# Literature Survey: Multimodal Neurophysiological Monitoring for Construction Safety

## 1. Problem Context

Construction remains one of the deadliest industries globally. In the United States, construction and extraction occupations accounted for 18.3 % of all workplace fatalities in 2021, totaling 951 deaths (Bureau of Labor Statistics, 2022). The majority of incidents are not equipment failures — they are human-factor failures: missed hazards, impaired decision-making, and risk-compensatory behavior under cognitive load.

---

## 2. Cognitive Failure Mechanisms in Construction

### 2.1 Cognitive Tunneling
Pooladvand & Hasanzadeh (2023) demonstrated using fNIRS that acute stress induces *cognitive tunneling* — a narrowing of attentional focus that leaves peripheral hazards undetected. This is distinct from simple distraction; tunneling is an active neural inhibition of spatial attention driven by prefrontal-parietal imbalance.

**Key finding**: Elevated cortisol (stress) correlates with reduced prefrontal oxygenation on secondary-task channels, measurable 4–8 s after stressor onset via HbO concentration changes.

### 2.2 Risk Compensation Under Time Pressure
Pooladvand & Hasanzadeh (2022) found that time pressure + high cognitive demand amplify risk-compensatory behavior: workers *know* the risk is higher but take it anyway to meet productivity demands. The neurophysiological signature is elevated theta-band EEG power (4–8 Hz) alongside fNIRS-measured prefrontal activation, indicating effortful top-down suppression of the "stop" signal.

### 2.3 Peer Influence (Social Contagion)
Pooladvand & Hasanzadeh (2024, conference) applied Social Contagion Theory to show that workers with higher susceptibility to peer pressure show different neural inhibitory control patterns, detectable via fNIRS functional connectivity between the prefrontal and motor cortex.

### 2.4 Mental Load Classification (fNIRS + ML)
The landmark Pooladvand, Chang & Hasanzadeh (2024, *Automation in Construction*) paper established:
- fNIRS-based mental load classification is feasible in mixed-reality construction environments
- EEG brain-band features (theta, alpha) improve beyond fNIRS alone when events are fast (<2 s)
- Individual variability is large (inter-subject σ ≈ 0.3 on classification accuracy) — motivating personalized baselines

---

## 3. Neuroimaging Technologies

### 3.1 fNIRS (Functional Near-Infrared Spectroscopy)
- **Mechanism**: Measures HbO/HbR via neurovascular coupling (BOLD-like response)
- **Temporal resolution**: 0.1–10 Hz; hemodynamic peak at ~6 s post-stimulus
- **Spatial resolution**: ~1 cm; frontal/parietal cortex accessible without hair interference
- **Construction suitability**: Tolerates motion better than EEG; wearable caps (Kernel Flow Pro, NIRx Scout) fit under hard hats
- **Limitation**: Slow; cannot detect events <2 s

### 3.2 EEG (Electroencephalography)
- **Mechanism**: Measures electrical dipoles from synchronized cortical neurons
- **Temporal resolution**: Milliseconds; event-related potentials (P300, N200) within 200–600 ms
- **Spatial resolution**: Poor (~5 cm after source localization)
- **Construction suitability**: Motion artifacts (walking, tool use) require careful correction; dry-electrode systems (Emotiv EPOC Flex, CGX Quick-30) reduce setup time
- **Advantage**: Real-time event detection; rich oscillatory dynamics (alpha suppression = attention, theta rise = working memory load)

### 3.3 fNIRS–EEG Fusion (the gap this work fills)
| Property | fNIRS | EEG | Fused |
|----------|-------|-----|-------|
| Temporal resolution | Low (~1 Hz) | High (ms) | Both |
| Spatial resolution | Medium | Low | Medium |
| Motion tolerance | High | Low | Constrained |
| Cognitive-state accuracy (literature) | 70–78 % | 72–80 % | 82–91 % |

Multimodal EEG–fNIRS fusion for cognitive/emotion classification consistently outperforms single-modality approaches by 5–15 % in F1 (Li et al., *IEEE TBME*, 2020; PMC Article 11674611, 2024; Frontiers *Hum. Neurosci.*, 2025).

---

## 4. Machine Learning Approaches for Cognitive State Detection

### 4.1 Traditional ML (SVM, LDA, RF)
- Most published fNIRS construction-safety work uses SVM on handcrafted features (mean HbO, slope, peak)
- Good interpretability but requires careful feature engineering per study paradigm

### 4.2 Deep Learning
- **CNN on fNIRS**: 1-D temporal CNNs capture multi-scale hemodynamic patterns (dilated convolutions effective per ResNet-style ablations)
- **LSTM on EEG band-power**: Temporal dynamics in band-power sequences model slow drift in cognitive state
- **Transformers**: Multi-head attention outperforms LSTM on long-context sequences (>30 s windows) but requires more data

### 4.3 Cross-Modal Attention (this work's contribution)
- Uses fNIRS as "query" (slow state) and EEG as "key/value" (fast events)
- Attention weights directly reveal *which EEG time-events drove the hemodynamic response* — interpretable for safety researchers
- Closest prior work: MBC-ATT (*Frontiers Hum. Neurosci.*, 2025) achieves 91.2 % on emotion recognition; **no construction-domain application exists**

---

## 5. Adaptive Interventions: What Works?

| Intervention Type | Effect Size | Notes |
|-------------------|-------------|-------|
| Auditory alert | ~30 % reduction in missed hazards | NIOSH, 2021 |
| AR hazard overlay | ~25 % improvement in safe behavior | Sacks et al., 2020 |
| Vibrotactile (wristband) | ~20 % reduction in near-misses | Pradhananga & Teizer, 2019 |
| Task-pause micro-break | ~15 % error reduction | ISO 9241-210 |
| **Closed-loop (all above, adaptive)** | **est. ≥ 35 %** | *This work (projected)* |

**Key design principle** (from human-factors literature): interventions must be *graded* (low for mild states, strong for severe) to avoid alert fatigue, which is itself a cause of ignored warnings.

---

## 6. Identified Research Gaps

1. **No closed-loop system** in construction that combines real-time neural classification with adaptive intervention
2. **Single-modality dominates** construction-safety neuroscience; fNIRS–EEG fusion is untested in this domain
3. **Fixed thresholds** across individuals fail to account for inter-subject variability (σ ≈ 0.3); personalized baselines are unvalidated in field-relevant VR paradigms
4. **Intervention efficacy** has been tested for individual modalities (audio, AR) but never for AI-triggered graded intervention sequences
5. **Interpretability gap**: Attention maps from cross-modal models offer a path to *explaining* which signals drove an alert — critical for safety manager trust and regulatory acceptance

---

## 7. References

- Pooladvand, S. & Hasanzadeh, S. (2022). Neurophysiological evaluation of workers' decision dynamics under time pressure and increased mental demand. *Automation in Construction*, 141, 104437.
- Pooladvand, S. & Hasanzadeh, S. (2023). Impacts of stress on workers' risk-taking behaviors: Cognitive tunneling and impaired selective attention. *Journal of Construction Engineering and Management*, 149(8), 04023060.
- Pooladvand, S., Chang, W., & Hasanzadeh, S. (2024). Identifying at-risk workers using fNIRS-based mental load classification: A mixed reality study. *Automation in Construction*.
- Lee, K., Pooladvand, S., Esmaeili, B., & Hasanzadeh, S. (2024). Understanding construction workers' risk perception using neurophysiological responses. *Journal of Computing in Civil Engineering*, 38(6).
- Li, R. et al. (2020). Enhancing fNIRS analysis using EEG rhythmic signatures. *IEEE Transactions on Biomedical Engineering*, 67, 2789–2797.
- Han, Y. et al. (2023). From brain to worksite: the role of fNIRS in cognitive studies and worker safety. *Frontiers in Public Health*, 1256895.
- Multimodal MBC-ATT (2025). Cross-modality attentional fusion of EEG–fNIRS for cognitive state decoding. *Frontiers in Human Neuroscience*, 1660532.
- Sacks, R. et al. (2020). Construction safety and digital design: A review. *Automation in Construction*, 113, 103103.
