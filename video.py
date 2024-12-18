# test.py

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchattacks
import os
import cv2
import numpy as np
from tqdm import tqdm

def process_frame(frame, model, atk, device, transform, label):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # Transform and process
    image = transform(pil_image).unsqueeze(0).to(device)
    # Generate adversarial example
    adv_image = atk(image, label)
    # Calculate perturbation
    perturbation = adv_image - image
    perturbation_normalized = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    
    # Convert back to numpy for OpenCV
    adv_frame = adv_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    pert_frame = perturbation_normalized.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Convert to BGR and uint8
    adv_frame = cv2.cvtColor((adv_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    pert_frame = cv2.cvtColor((pert_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return adv_frame, pert_frame

def run_video_attack(video_path, class_index, eps=2/255, alpha=1/255):
    # Set for reproducibility
    torch.backends.cudnn.deterministic = True
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.eval().to(device)
    
    # Transform setup
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Attack setup
    atk1 = torchattacks.FGSM(model, eps=eps)
    atk2 = torchattacks.PGD(model, eps=eps/2, alpha=alpha, steps=4, random_start=True)
    atk = torchattacks.MultiAttack([atk1, atk2])
    atk.set_normalization_used(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Create label tensor
    label = torch.tensor([class_index]).to(device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writers
    adv_writer = cv2.VideoWriter(
        'adversarial_output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    pert_writer = cv2.VideoWriter(
        'perturbation_output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    try:
        # Process each frame
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                adv_frame, pert_frame = process_frame(frame, model, atk, device, transform, label)
                
                adv_writer.write(adv_frame)
                pert_writer.write(pert_frame)
                
                pbar.update(1)
                
    finally:
        # Clean up
        cap.release()
        adv_writer.release()
        pert_writer.release()
        print("Finished processing video")
        print("Saved:\n- Adversarial video: adversarial_output.mp4\n- Perturbation video: perturbation_output.mp4")

if __name__ == "__main__":
    run_video_attack("input.mp4", 0)  # Replace with your video path