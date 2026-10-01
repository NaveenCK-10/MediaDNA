import os

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

ensure_dir('.github/ISSUE_TEMPLATE')
ensure_dir('docs/assets')

# 1. GitHub Issue Templates
issues = {
    'bug_report.md': """---
name: 🐛 Forensic Bug
about: Report an error in the MediaDNA forensic pipeline, UI, or backend
title: '[BUG] '
labels: 'bug'
assignees: ''
---

**CASE ID:** 
**MODULE:** [e.g., Visual Extraction, Audio Calibration, Playwright]

**DESCRIPTION**
A clear and concise description of what the bug is.

**SCIENTIFIC IMPACT**
Does this bug affect inference results, probability output, or purely UI display?

**REPRODUCTION STEPS**
1. 
2. 
3. 

**EVIDENCE**
Provide screenshots, stack traces, or corrupted PDF reports.
""",
    'research.md': """---
name: 🔬 Research Proposal
about: Propose a new ablation, architecture experiment, or dataset evaluation
title: '[RESEARCH] '
labels: 'research'
assignees: ''
---

**RESEARCH QUESTION**
What hypothesis is being tested?

**PROPOSED METHODOLOGY**
Describe the modification to the pipeline or evaluation metric.

**DATASET / SPLIT**
What data will this experiment use?
""",
}

for name, content in issues.items():
    with open(f'.github/ISSUE_TEMPLATE/{name}', 'w', encoding='utf-8') as f:
        f.write(content)

# 2. PR Template
pr_template = """CASE ID:
MODULE:
CHANGE TYPE:

- [ ] MODEL
- [ ] PIPELINE
- [ ] UI
- [ ] FORENSICS
- [ ] RESEARCH

**RATIONALE:**

**SCIENTIFIC IMPACT:**
Does this alter ROC-AUC, threshold policies, or AVFF fusion behavior?

**ENGINEERING IMPACT:**
Does this change FFmpeg extraction, dependencies, or Playwright reporting?

**TEST EVIDENCE:**

**REPRODUCIBILITY:**
"""
with open('.github/pull_request_template.md', 'w', encoding='utf-8') as f:
    f.write(pr_template)

# 3. CITATION.cff
citation = """cff-version: 1.2.0
message: "If you use MediaDNA for forensic or deepfake analysis research, please cite it as below."
title: "MediaDNA: Deepfake Provenance & Multimodal Media Authenticity Analysis"
authors:
  - name: "MediaDNA Contributors"
date-released: 2026-10-01
url: "https://github.com/NaveenCK-10/MediaDNA"
references:
  - type: article
    title: "AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection"
    authors:
      - family-names: "Oorloff"
        given-names: "Treven A."
      - family-names: "Masi"
        given-names: "Iacopo"
      - family-names: "Hosseini"
        given-names: "Ramin"
    journal: "CVPR"
    year: 2024
"""
with open('CITATION.cff', 'w', encoding='utf-8') as f:
    f.write(citation)

# 4. Docs: SCIENTIFIC_LIMITATIONS.md
limitations = """# Scientific Limitations & Forensic Constraints

MediaDNA is an experimental forensic instrument. It is NOT a definitive oracle.

## 1. Dataset Dependence
The system is anchored entirely to the `FakeAVCeleb_v1.2` dataset distribution.

## 2. Modality Imbalance
The current V22.3/V22.4 pipeline shows an extreme dependency on the audio channel. Silent videos or aggressively compressed audio may yield unexpected or unreliable visual-only outputs.

## 3. Threshold Calibration
Calibration was fit strictly against `V22_2` predictions. The 0.95 (AUTHENTIC) / 0.96 (SYNTHETIC) boundaries are not universal truths.

## 4. OOD (Out of Distribution)
Generalization to wild deepfakes (e.g., Sora, Midjourney V6 video, advanced 2026 audio cloning) is unverified.

## 5. Security Limitations
MediaDNA currently lacks hardened sandboxing for FFmpeg parsing. Malicious video streams could exploit underlying `libav` vulnerabilities.
"""
with open('docs/SCIENTIFIC_LIMITATIONS.md', 'w', encoding='utf-8') as f:
    f.write(limitations)

# 5. Docs: REPRODUCIBILITY.md
reproducibility = """# Reproducibility Guide

## Environment
Ensure PyTorch 2.0+ with CUDA 11.8.

## Checkpoint Identity
The baseline uses `best_audio_model.pth` (V14). SHA256 integrity checks must pass the 313 tensor audit script.

## Splitting
MediaDNA relies on strict, immutable dataset manifesting. All samples in `locked_test` are cryptographically verified to have zero overlap with `train` and `dev` folds.

## Preprocessing Parity
- **Video:** 16 frames uniformly extracted (ffprobe duration / 16).
- **Audio:** 16kHz PCM, 1024 frames, 128 Mel bands.
"""
with open('docs/REPRODUCIBILITY.md', 'w', encoding='utf-8') as f:
    f.write(reproducibility)

# 6. SVG Assets (Social Preview)
svg = """<svg width="1280" height="640" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#0a0a0c"/>
  <rect x="40" y="40" width="1200" height="560" fill="none" stroke="#1f4e94" stroke-width="2" stroke-dasharray="10, 10"/>
  
  <text x="640" y="260" font-family="monospace" font-size="72" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="10">MEDIADNA</text>
  <text x="640" y="320" font-family="monospace" font-size="24" fill="#99ceff" text-anchor="middle" letter-spacing="4">DEEPFAKE PROVENANCE &amp; MULTIMODAL FORENSICS</text>
  
  <text x="640" y="420" font-family="monospace" font-size="18" fill="#1f4e94" text-anchor="middle" letter-spacing="8">AUDIO × VIDEO × FUSION × EVIDENCE</text>
  
  <rect x="440" y="500" width="400" height="40" fill="#001426" stroke="#1a3e5f" rx="4"/>
  <text x="640" y="525" font-family="monospace" font-size="16" fill="#99ceff" text-anchor="middle" letter-spacing="2">[ FORENSIC COMMAND CENTER ]</text>
</svg>
"""
with open('docs/assets/mediadna-social-preview.svg', 'w', encoding='utf-8') as f:
    f.write(svg)

print("Repository templates and scientific documentation generated.")
