# test.py

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchattacks
import os

def run_attack(image_path, class_index):  # Reduced from 8/255 to 4/255
    # Set for reproducibility
    torch.backends.cudnn.deterministic = True
    
    # Check if CUDA is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model using new weights parameter
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.eval().to(device)
    
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor()  # Just convert to tensor, maintain original dimensions
    ])
    
    try:
        # Load and transform image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path)
        print(f"Original image dimensions: {image.size}")
        
        image = transform(image).unsqueeze(0).to(device)
        print(f"Transformed image dimensions: {image.shape}")
        
        # Create label tensor
        label = torch.tensor([class_index]).to(device)
        
        # Initialize attack with weaker parameters
        # print("Initializing PGD attack...")
        # atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)

        # print("Initializing FGSM attack...")
        # atk = torchattacks.FGSM(model, eps=2/255)

        atk1 = torchattacks.CW(model, c=0.1, steps=1000, lr=0.01)
        atk2 = torchattacks.CW(model, c=1, steps=1000, lr=0.01)
        atk = torchattacks.MultiAttack([atk1, atk2])
        
        # Set normalization (ImageNet values)
        atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Generate adversarial example
        print("Generating adversarial example...")
        adv_image = atk(image, label)
        
        # Calculate perturbation (noise pattern)
        perturbation = adv_image - image
        
        # Normalize perturbation for visibility
        perturbation_normalized = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
        
        # Save results
        torchvision.utils.save_image(adv_image, "adversarial_output.png")
        torchvision.utils.save_image(perturbation_normalized, "perturbation.png")
        print(f"Saved:\n- Adversarial image: adversarial_output.png\n- Perturbation pattern: perturbation.png")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage with weaker attack parameters
    run_attack("kopf3.png", 0)  # Reduced strength