def calculate_fusion(openavff_fake_prob: float, visual_anomaly_score: float) -> dict:
    """
    Combines the OpenAVFF deep learning model probability with the deterministic
    visual anomaly score.
    
    OpenAVFF probability is P(Fake).
    If visual anomaly is high, it pushes P(Fake) higher.
    If visual anomaly is low, it slightly reduces P(Fake).
    
    Weights: 80% OpenAVFF, 20% Visual Anomaly
    """
    
    fusion_prob = (openavff_fake_prob * 0.8) + (visual_anomaly_score * 0.2)
    
    # Clamp to 0.0 - 1.0
    fusion_prob = max(0.0, min(1.0, fusion_prob))
    
    return {
        "mediadna_fake_prob": float(fusion_prob),
        "mediadna_real_prob": float(1.0 - fusion_prob),
        "prediction": "fake" if fusion_prob >= 0.60 else "real"
    }
