import torch
import argparse
from resnet18 import ResNet18
from smallvit import ViTSmallCIFAR10

def get_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in bytes (only trainable parameters)
    # Each parameter is a float32 by default → 4 bytes
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 ** 2)  # Convert to MB

    print(f"Trainable Parameters: {total_params:,}")
    print(f"Estimated Model Size: {model_size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "vit"])
    args = parser.parse_args()
    if args.model == "resnet":
        model = ResNet18() 
    elif args.model == "vit":
        model = ViTSmallCIFAR10(pretrained=True, num_classes=10)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    get_model_stats(model)
