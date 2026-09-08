import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage-3.pth")
    parser.add_argument("--csv_file", type=str, default="./data/test_small.csv")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    audio_model = VideoCAVMAEFT()
    audio_model = torch.nn.DataParallel(audio_model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    audio_model.load_state_dict(ckpt, strict=False)
    audio_model.to(device)
    audio_model.eval()

    dataset_mean=-5.081
    dataset_std=4.4849
    target_length=1024
    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                      'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}
    dataset = dataloader.VideoAudioEvalDataset(csv_file=args.csv_file, audio_conf=val_audio_conf)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("\n--- DIAGNOSIS RESULTS ---")
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_names) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                audio_output = audio_model(a_input, v_input)
            
            raw_output = audio_output.cpu().numpy()[0]
            sigmoid_out = torch.sigmoid(audio_output).cpu().numpy()[0]
            softmax_out = torch.softmax(audio_output, dim=1).cpu().numpy()[0]
            
            # Label is [batch_size, 2] -> e.g. [[1, 0]]
            # But the dataset returns a single sample in dataloader since batch_size=1
            label_val = labels.cpu().numpy()[0]
            csv_label = 1 if label_val[0] == 1 else 0

            pred_class_sigmoid = 1 if sigmoid_out[0] > 0.5 else 0
            pred_class_softmax = np.argmax(softmax_out)
            
            # Since index 0 is Fake (1), class is reversed if argmax is used.
            # If softmax_out[0] > softmax_out[1], it predicts [1, 0], which is Fake (1).
            pred_class = 1 if pred_class_softmax == 0 else 0

            print(f"Video: {video_names[0].split('/')[-1]}")
            print(f"CSV label: {csv_label} (Target tensor: {label_val})")
            print(f"Raw output (logits): {raw_output}")
            print(f"Sigmoid: {sigmoid_out}")
            print(f"Softmax: {softmax_out}")
            print(f"Predicted class (based on index 0): {1 if softmax_out[0] > softmax_out[1] else 0}")
            print(f"Confidence (index 0): {sigmoid_out[0]:.4f}")
            print("-------------------------")

if __name__ == '__main__':
    main()
