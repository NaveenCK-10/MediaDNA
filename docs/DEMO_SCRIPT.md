# MediaDNA Demonstration Script (5 Minutes)

## 0:00 - Introduction
*(Start on the MediaDNA Dashboard hero section)*
"Welcome to MediaDNA. This is a multimodal digital forensics instrument designed to detect deepfakes. Rather than just looking at a picture, MediaDNA specifically analyzes the relationship between the video track and the audio track, searching for acoustic and spatial desynchronization using a deep learning architecture known as OpenAVFF."

## 0:30 - Show Architecture
*(Navigate to the /architecture page)*
"If we look under the hood, the system is entirely transparent. MediaDNA ingests a video, strips the audio into Kaldi filterbanks, and pulls 16 uniform frames. These modalities are fed into a Cross-Attention Video-Audio network, where the audio actively attends to the video and vice-versa. At the same time, we run classical computer vision heuristics—like Laplacian variance—to independently measure physical blur or face-swap glitching. These signals are fused together to produce our final forensic assessment."

## 1:15 - Upload Real Media
*(Navigate to /analyze. Upload a genuine .mp4 from the desktop)*
"Let's run a live analysis. I'm dragging in an authentic, unedited video clip. You can see the metadata extract immediately. We click 'BEGIN INFERENCE'."
*(Wait for cinematic sequence)*
"The system runs the PyTorch tensors, and as expected, it is classified as 'AUTHENTIC / PRISTINE'. If we drop down 'VIEW EVIDENCE', we can see the exact breakdown. The visual anomaly score is low, and the OpenAVFF signal heavily favors real."

## 2:30 - Show FakeVideo + RealAudio Blindspot
*(Click EXPORT FORENSIC REPORT to show the capability, then point out the Limitations section)*
"However, this system is not a magic bullet. Right here in the report, we explicitly document a severe architectural blindspot: FakeVideo combined with RealAudio. Because the model relies heavily on cross-attention, if an attacker dubs completely pristine, unaltered audio over a fake video, the model's audio encoder gets 'tricked' and suppresses the visual anomalies."

## 3:15 - Show Four-Case Demo Lab
*(Navigate back to Dashboard, scroll to Forensic Demo Lab)*
"To prove this, we built the Forensic Demo Lab. Here we have the four quadrants of multimodal media mapped to actual local files. Let's run the 'FAKE / FAKE' case."
*(Click FAKE/FAKE, show it correctly catches it as Fake)*
"The model perfectly catches a fully synthetic video. Now, let's run the 'FAKE / REAL' case—the blindspot."
*(Go back, click FAKE/REAL, show it fails and says Real)*
"As you can see, the neural network fails and predicts 'AUTHENTIC'. This is exactly why we require transparent visual heuristics and refuse to claim the model is infallible."

## 4:00 - Show Research Results
*(Navigate to /research page)*
"We've rigorously documented these findings. On our Research dashboard, you can see the hard data. The OpenAVFF baseline achieved a 64.69% accuracy on the FakeAVCeleb dataset, but plummeted to 27.20% on the FakeVideo-RealAudio subset. We attempted an experimental MediaDNA fusion to mathematically force visual anomaly detection, but while it improved the blindspot, it degraded the overall AUC."

## 4:30 - Show Limitations
*(Stay on Research page, point to the Robustness section)*
"Furthermore, we ran red-team robustness tests. We discovered that introducing a 10dB audio noise profile completely breaks the acoustic encoder, rendering the system unusable in noisy environments, though it remains surprisingly resilient to heavy video compression."

## 5:00 - Conclusion
*(Return to Dashboard)*
"In conclusion, MediaDNA is not a perfect 'state-of-the-art' black box. It is a transparent, scientifically accountable research instrument. It exposes exactly how it thinks, provides exportable evidence for analysts, and openly documents its own failure modes. Thank you."
