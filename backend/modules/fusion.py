def calculate_fusion(openavff_fake_score: float, visual_anomaly_score: float) -> dict:
    """
    Combines the OpenAVFF deep learning model score with the deterministic
    visual anomaly score.
    
    OpenAVFF score is P(Fake).
    If visual anomaly is high, it pushes P(Fake) higher.
    If visual anomaly is low, it slightly reduces P(Fake).
    
    Weights: 80% OpenAVFF, 20% Visual Anomaly
    """
    
    fusion_score = (openavff_fake_score * 0.8) + (visual_anomaly_score * 0.2)
    
    # Clamp to 0.0 - 1.0
    fusion_score = max(0.0, min(1.0, fusion_score))
    
    return {
        "mediadna_fake_score": float(fusion_score),
        "mediadna_real_score": float(1.0 - fusion_score),
        "prediction": "fake" if fusion_score >= 0.60 else "real"
    }
