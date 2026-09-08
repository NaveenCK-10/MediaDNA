import torch
import torch.nn as nn
from backend.inference import OpenAVFFService

def test_inference_bias():
    service = OpenAVFFService()
    model = service.model
    model.eval()
    
    print("Testing zero-tensor bias...")
    
    # Correct shapes as expected by model (B, 1024, 128) for audio, (B, 3, 16, 224, 224) for video
    vframes = torch.zeros((1, 3, 16, 224, 224)).to(service.device)
    fbank = torch.zeros((1, 1024, 128)).to(service.device)
    
    with torch.no_grad():
        out = model(fbank, vframes)
        if isinstance(out, dict):
            logits = out['pred']
        else:
            logits = out
            
        probs = torch.sigmoid(logits)
        fake_prob = probs[0][0].item()
        
    print(f"Fake Probability on Zero Tensor: {fake_prob:.4f}")
    
    # Assert that it is roughly 0.5302
    assert 0.52 < fake_prob < 0.54, f"Expected Fake prob ~0.53, got {fake_prob}"
    print("Bias test passed! Zero tensors yield ~0.53 probability.")

if __name__ == '__main__':
    test_inference_bias()
