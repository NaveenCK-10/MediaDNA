# MediaDNA Viva Q&A

**Q: Did you prove a causal relationship between audio phonemes and lip movements?**
A: No, we provided observational evidence consistent with audio contribution. The diagnostic triad showed that altering the audio stream changes the fusion score, confirming the model relies on audio features, though it does not strictly prove causal phonetic-level alignment.

**Q: Why does the model predict almost everything as fake on the locked test?**
A: Our Platt calibration, fitted on the CAL split, skewed probabilities high. When we established the operating-point reference using Youden's J statistic on the DEV set, the optimal cut-point was 0.9773. The underlying ROC-AUC (0.60) shows there is some signal, but evaluating threshold sensitivity under the locked-test data distribution revealed poor specificity.

**Q: Why did you split the architecture into two specialists?**
A: We identified a pooling collapse bug in the baseline V14 AVFF model where 1536-dimensional features were averaged, destroying the acoustic signal. By splitting the model and using late fusion, we preserved both modalities independently.
