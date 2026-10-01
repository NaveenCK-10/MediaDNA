# MediaDNA PDF Report Design

## 1. Report Structure & Page Logic
The MediaDNA generated report utilizes a flexible, responsive 1-2 page A4 template. 
- **Page 1: Executive Summary**
  - Plain-language overview for a non-technical audience.
  - Dynamically responds to the overall `decision_state` with clear visual bounds (Authentic / Synthetic / Uncertain).
  - Explicit explanation of the decision, why the system made it (Visual + Audio = Fusion), and basic file details.
- **Page 2: Technical Forensic Details**
  - Designed for researchers, engineers, and reviewers.
  - Contains exact boundary logic, raw modality scores, fusion scores, and calibrated probabilities.
  - Transparent statements about limitations, model provenance, and methodology.

## 2. Audience Strategy
The PDF is designed to be readable by non-technical audiences via the first page, which abstracts complex numerical probabilities into a simple confidence statement. Technical audiences can continue to the second page to inspect the underlying raw mechanics, ablation-level scores, and dataset methodology.

## 3. Decision Wording
- `AUTHENTIC`: "MediaDNA estimates that this media is likely authentic."
- `SYNTHETIC`: "MediaDNA estimates that this media is likely synthetic."
- `UNCERTAIN`: "MediaDNA could not distinguish the media confidently. The result is therefore marked uncertain."
- Each decision is always accompanied by: "This is a statistical model estimate, not a guarantee."

## 4. Evidence Wording
Scores for the independent visual and acoustic pathways, as well as the late fusion product, are provided dynamically. The plain language translations deliberately use neutral, non-absolute terminology like: "Audio analysis contributed evidence toward the final assessment."

## 5. Technical Wording
The PDF avoids hyperbolic claims. It explicitly notes that "Model-sensitive Region" analysis does not equate to "ground-truth localization", and explicitly limits the capability to the evaluated data distribution. All references to exact generator attribution have been removed in accordance with the project's true scientific limitations.

## 6. Layout Rules
- Avoid empty pages or oversized fonts to span spaces. 
- Ensure a unified, modern, robust design language matching the web application frontend (Amber for Uncertain, Green for Authentic, Red for Synthetic).
- Never render mock values. The entire HTML/CSS render tree reads directly from the `profile_data` populated by the production pipeline in `backend/inference.py`.
