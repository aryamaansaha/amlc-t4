import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=32, root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
