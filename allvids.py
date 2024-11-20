# video.py

import torch
import torch_directml  # Add this import
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchattacks
import cv2
import numpy as np
from tqdm import tqdm
import subprocess  # Add this import at the top
import glob
import os
import re

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
    device = torch_directml.device()  # Change the device setup
    print(f"Using device: {device}")
    print(f"Device name: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"DirectML available: {torch_directml.is_available()}")
    
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.eval().to(device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Attack setup with reduced epsilon
    atk1 = torchattacks.FGSM(model, eps=1/255)
    atk2 = torchattacks.PGD(model, eps=1/255, alpha=2/255, steps=10, random_start=True)
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
        
        # Add audio from original video using ffmpeg
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', 'adversarial_output.mp4',  # Video without audio
                '-i', video_path,                # Original video with audio
                '-c:v', 'copy',                  # Copy video stream
                '-map', '0:v:0',                 # Use video from first input
                '-map', '1:a:0',                 # Use audio from second input
                '-y',                            # Overwrite output if exists
                'output.mp4'               # Final output file
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            print("Final video with audio saved as: final_output.mp4")
        except subprocess.CalledProcessError as e:
            print(f"Error merging audio: {e}")
        except FileNotFoundError:
            print("ffmpeg not found. Please ensure ffmpeg is installed and in your PATH")

def get_number_from_filename(filename):
    # Extract number from filename like "input (1).mp4"
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else 0

def process_multiple_videos(class_index):
    # Find all input videos matching the pattern
    input_videos = glob.glob('input (*).mp4')
    
    if not input_videos:
        print("No input videos found matching pattern 'input (#).mp4'")
        return
        
    input_videos.sort(key=get_number_from_filename)
    
    for input_video in input_videos:
        # Get the number from input filename
        num = get_number_from_filename(input_video)
        
        # Set output filenames based on input number
        temp_output = f'adversarial_output ({num}).mp4'
        final_output = f'output ({num}).mp4'
        
        print(f"\nProcessing {input_video}...")
        
        # Modify run_video_attack to use dynamic output names
        device = torch_directml.device()
        print(f"Using device: {device}")
        
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model = model.eval().to(device)
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        atk1 = torchattacks.FGSM(model, eps=1/255)
        atk2 = torchattacks.PGD(model, eps=1/255, alpha=2/255, steps=10, random_start=True)
        atk = torchattacks.MultiAttack([atk1, atk2])
        
        atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        label = torch.tensor([class_index]).to(device)
        
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Could not open {input_video}, skipping...")
            continue
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        adv_writer = cv2.VideoWriter(
            temp_output,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        try:
            with tqdm(total=total_frames, desc=f"Processing frames for video {num}") as pbar:
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
            print(f"Saved: {temp_output}")
            
            # Add audio from original video using ffmpeg
            try:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', temp_output,
                    '-i', input_video,
                    '-c:v', 'copy',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-y',
                    final_output
                ]
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"Final video with audio saved as: {final_output}")
                
                # Clean up temporary file
                os.remove(temp_output)
                
            except subprocess.CalledProcessError as e:
                print(f"Error merging audio: {e}")
            except FileNotFoundError:
                print("ffmpeg not found. Please ensure ffmpeg is installed and in your PATH")

if __name__ == "__main__":
    process_multiple_videos(0)