# MediaDNA Forensic Report Specification

## Overview
The MediaDNA Forensic Report is a premium, research-grade, downloadable PDF that aggregates all multimodal evidence (visual, audio, temporal) alongside model confidence and calibrated authenticity assessments. It operates autonomously via the NVIDIA NIM LLM provider (or fallback models/deterministic generation) to offer grounded, explainable AI narratives without fabricating conclusions.

## Objectives
1. **Explainable AI**: Visualize what the OpenAVFF and branches detect using heatmaps, spectrograms, and temporal sequences.
2. **Provenance & Authenticity**: Provide estimated generation sources and calibrated probability scores.
3. **Traceability**: Link every report section back to the core project abstract.
4. **Auditability**: Ensure all system metadata, versionings, and fingerprints are embedded in the report for reproducible digital forensic reviews.

## LLM Integration
- **Provider**: NVIDIA NIM (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`)
- **Role**: Natural language interpretation of structured JSON evidence.
- **Constraints**: The LLM *must not* independently determine real/fake probabilities, provenance categories, or generate timestamp data. It solely writes the interpretation narrative.

## PDF Generation Engine
- **Tool**: Playwright Chromium (Headless)
- **Methodology**: HTML/CSS templates via Jinja2 are populated and dynamically rendered. The report adheres to a dark, forensic-scientific aesthetic with cyan/blue accents and glassmorphism styling.

## Required Sections
1. **Executive Forensic Summary**: Core classification, fused probability, and key modality findings.
2. **Media Profile**: Extracted properties (codec, dimensions, duration, etc.).
3. **MediaDNA Fingerprint**: Cryptographic or deterministic embedding signature.
4. **Visual, Audio, Temporal Evidence**: Extracted signals with anomaly distributions and heatmaps/spectrograms.
5. **Manipulation & Provenance Analysis**: Predicted generation categories.
6. **Methodology & Ablation**: Scientific justification and research context.
7. **AI Forensic Explanation**: LLM-generated narrative interpretation.

## API Integration
- `POST /api/report/{case_id}`: Generates the PDF using history tracking JSON data and saves it.
- `GET /api/report/{case_id}`: Serves the generated PDF to the client.

*Generated autonomously by MediaDNA.*
