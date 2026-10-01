# Dataset Reality Audit

## Verdict: FULL LOCAL DATASET; V22.1 SELECTION/PROTOCOL FAILURE

The inspected directory `FakeAVCeleb_v1.2/FakeAVCeleb_v1.2` contains 21,560 MP4 files (6,617,323,527 bytes) and 24,126 files total (6,621,466,154 bytes). It is not a 4–6 sample dataset.

| Modality condition | MP4 files |
|---|---:|
| FakeVideo-FakeAudio | 11,366 |
| FakeVideo-RealAudio (FVRA) | 10,225 |
| RealVideo-RealAudio | 1,516 |
| RealVideo-FakeAudio (RVFA) | 1,016 |

`meta_data.csv` has 21,567 lines including its header. The locked V21.4 manifest instead contains exactly six paths, all present locally. `V22_1_EXECUTION.py` loads that manifest into `DummyDataset` and trains for one epoch on it. Therefore the historical “4 samples”/“limited dataset” assertion is contradicted by the filesystem; the immediate defect is loader/manifest selection, not data availability.

`src/dataloader.py` accepts any CSV and lazily decodes samples, so it does not itself impose a six-sample cap. The repository lacks a versioned, identity-disjoint V22 train/dev/calibration manifest. A full-data V22 run must not begin until those artifacts exist.

No files in the dataset, manifests, or loader were changed in this audit.
