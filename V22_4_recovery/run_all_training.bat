call activate base
call conda activate base

echo "--- RUNNING VISUAL ---"
python V22_4_recovery\train_visual_specialist.py

echo "--- RUNNING AUDIO ---"
python V22_4_recovery\train_audio_specialist.py

echo "--- RUNNING MULTIMODAL ---"
python V22_4_recovery\train_multimodal.py
