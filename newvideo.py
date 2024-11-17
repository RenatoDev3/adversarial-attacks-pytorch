# video.py

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchattacks
import cv2
import numpy as np
from tqdm import tqdm

def process_frame(frame, model, atk, device, transform, label):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image = transform(pil_image).unsqueeze(0).to(device)
    
    # Enable gradients for this frame
    image.requires_grad = True
    
    # Generate adversarial example
    adv_image = atk(image, label)
    
    # Convert back to numpy for OpenCV
    adv_frame = adv_image.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_frame = np.clip(adv_frame, 0, 1)  # Ensure values are in [0,1]
    adv_frame = cv2.cvtColor((adv_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return adv_frame

def run_video_attack(video_path, class_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.eval().to(device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Attack setup with reduced epsilon
    atk1 = torchattacks.FGSM(model, eps=1/255)
    atk2 = torchattacks.PGD(model, eps=1/255, alpha=2/255, steps=4, random_start=True)
    atk = torchattacks.MultiAttack([atk1, atk2])

    #  atk = torchattacks.FGSM(model, eps=1/255)  # Reduced epsilon

    atk.set_normalization_used(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    label = torch.tensor([class_index]).to(device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    adv_writer = cv2.VideoWriter(
        'adversarial_output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    try:
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                adv_frame = process_frame(frame, model, atk, device, transform, label)
                adv_writer.write(adv_frame)
                pbar.update(1)
                
    finally:
        cap.release()
        adv_writer.release()
        print("Saved: adversarial_output.mp4")

if __name__ == "__main__":
    run_video_attack("input.mp4", 0)