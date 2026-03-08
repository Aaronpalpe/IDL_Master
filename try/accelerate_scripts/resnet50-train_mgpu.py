import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from accelerate import Accelerator

class RandomImageDataset(Dataset):
    def __init__(self, length: int, num_classes: int = 10):
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # CPU tensors generated on the fly (small memory footprint)
        x = torch.rand(3, 224, 224, dtype=torch.float32)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y

def main():
    accelerator = Accelerator()

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    per_device_batch_size = 32
    num_samples = 4096      # safe now (no big allocation)
    num_epochs = 3

    dataset = RandomImageDataset(num_samples, num_classes=10)
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    # Warmup
    warmup_steps = 5
    it = iter(dataloader)
    for _ in range(warmup_steps):
        inputs, targets = next(it)
        inputs = inputs.to(accelerator.device, non_blocking=True)
        targets = targets.to(accelerator.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed training
    total_seen_local = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(accelerator.device, non_blocking=True)
            targets = targets.to(accelerator.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            total_seen_local += inputs.size(0)

        avg_loss = torch.tensor(running_loss / len(dataloader), device=accelerator.device)
        avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    world = accelerator.num_processes
    global_seen = total_seen_local * world
    if accelerator.is_main_process:
        print(f"Done. world_size={world} global_images={global_seen} time={t1-t0:.2f}s "
              f"throughput={global_seen/(t1-t0):.1f} img/s")

    accelerator.end_training()

if __name__ == "__main__":
    main()

