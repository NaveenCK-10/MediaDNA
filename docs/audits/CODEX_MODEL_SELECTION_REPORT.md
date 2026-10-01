# Model Selection Report

## Decision: do not select or replace the current model

The available record cannot compare AVFF, specialists, late/gated fusion, multitask, synchronization, current research systems, or multimodal LLM approaches: V22.1 is unexecuted/invalid and V14 provenance is incomplete. Large multimodal LLMs are not justified as a detector substitute; they add cost/latency and would require the same split, calibration, robustness, and reproducibility evidence.

Recommended research sequence, pending a valid protocol:

1. Freeze identity/source/derivative-disjoint train, dev, calibration, and locked-test manifests from FakeAVCeleb metadata; hash every input and publish overlap reports.
2. Establish the current AVFF baseline and a visual specialist separately. Measure RVRA/RVFA/FVRA/FVFA, including audio replacement and silence, before fusion work.
3. Add an audio specialist only if it improves dev-condition metrics without worsening FVRA. Compare simple late fusion before gated fusion; fit weights only on dev.
4. Add A/V synchronization only with genuine shifted-pair labels and a real learned representation—not zero proxy features. Calibrate and evaluate once on the locked test.
5. Run external validation after license/access/protocol review. AV-Deepfake1M supplies official loaders/evaluation but requires EULA access; LAV-DF is CC BY-NC 4.0 and is research-only, so neither is currently a production-data solution. Sources: [AV-Deepfake1M official repository](https://github.com/ControlNet/AV-Deepfake1M) and [LAV-DF official repository](https://github.com/ControlNet/LAV-DF).

**Recommended MediaDNA architecture:** a staged AVFF-baseline + independently validated visual/audio specialists with late fusion initially, calibration, quality gates, and abstention. This is a research plan, not an approved architecture. Promote to gated fusion or synchronization only on preregistered dev gains and locked-test replication.
