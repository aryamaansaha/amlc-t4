import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

import pandas as pd

from resnet18 import ResNet18
from data_loader import get_cifar10_loaders

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def profile_training(model_name="resnet", warmup_epochs=5, profile_epochs=3, batch_size=64, max_batches=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet18() if model_name == "resnet" else None
    model.to(device)

    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"[Warmup] Training {model.__class__.__name__} for {warmup_epochs} epochs...")
    for epoch in range(warmup_epochs):
        loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        print(f"Epoch {epoch+1}/{warmup_epochs} - Loss: {loss:.4f}")

    print(f"\n[Profiling] Training for {profile_epochs} epochs (only first {max_batches} batches each)...")

    os.makedirs("profiler_output", exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiler activities: {[a.name for a in activities]}")

    trace_file = f"profiler_output/{model_name}_trace.json"
    csv_file = f"profiler_output/{model_name}_summary.csv"

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=2, warmup=2, active=6, repeat=1  # Total = 10 batches
        ),
        on_trace_ready=lambda prof: prof.export_chrome_trace(trace_file),
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        model.train()
        batch_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with record_function("forward"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            prof.step()
            batch_count += 1
            if batch_count >= 10:  # total_batches = wait + warmup + active
                break


    print("\nProfiler Summary:")
    summary = prof.key_averages().table(
        sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total",
        row_limit=15
    )
    print(summary)

    total_flops = 0
    events = prof.key_averages()
    for evt in events:
        if hasattr(evt, 'flops'):
            total_flops += evt.flops

    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    summary_txt_file = f"profiler_output/{model_name}_summary.txt"
    with open(summary_txt_file, 'w') as f:
        f.write(summary)
    print(f"Saved profiler summary to: {summary_txt_file}")
    print(f"Saved Chrome trace to: {trace_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "mlp"])
    args = parser.parse_args()

    profile_training(model_name=args.model)