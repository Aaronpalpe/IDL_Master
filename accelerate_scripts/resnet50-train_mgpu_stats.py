import os, time, socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs


class RandomImageDataset(Dataset):
    def __init__(self, length: int, num_classes: int = 10):
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.rand(3, 224, 224, dtype=torch.float32)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y


def main():
    # Profile CUDA kernels
    profile_kwargs = ProfileKwargs(
        activities=["cuda"],
        record_shapes=True,
    )
    accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
    device = accelerator.device

    rank = accelerator.process_index
    world = accelerator.num_processes
    local_rank = accelerator.local_process_index
    host = socket.gethostname()

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    per_device_batch_size = 32
    num_samples = 4096
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

    # Warmup (optional)
    warmup_steps = 5
    it = iter(dataloader)
    for _ in range(warmup_steps):
        inputs, targets = next(it)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Training + profiling
    total_seen_local = 0
    t0 = time.time()

    with accelerator.profile() as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()

                running_loss += loss.item()
                total_seen_local += inputs.size(0)

            # Reduce loss across ranks for consistent logging (rank0 prints)
            avg_loss = torch.tensor(running_loss / len(dataloader), device=device)
            avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", flush=True)

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    # Global images processed (safe without gather_object)
    global_seen = int(
        accelerator.reduce(torch.tensor(total_seen_local, device=device), reduction="sum").item()
    )

    # Optional: per-rank throughput (local) and global throughput
    local_throughput = total_seen_local / (t1 - t0) if (t1 - t0) > 0 else float("nan")
    global_throughput = global_seen / (t1 - t0) if (t1 - t0) > 0 else float("nan")

    # Print summary from rank0
    if accelerator.is_main_process:
        print(
            f"\nDone. world_size={world} global_images={global_seen} time={t1-t0:.2f}s "
            f"global_throughput={global_throughput:.1f} img/s\n",
            flush=True
        )

    # ---- Print each rank's profiler table (serialized to avoid interleaving) ----
    accelerator.wait_for_everyone()
    for r in range(world):
        if rank == r:
            gpu_name = None
            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                except Exception:
                    gpu_name = None

            print("=" * 70, flush=True)
            print(
                f"[RANK {rank}/{world}] host={host} local_rank={local_rank} device={device} gpu={gpu_name} "
                f"local_images={total_seen_local} local_throughput={local_throughput:.1f} img/s",
                flush=True
            )
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
            print("=" * 70, flush=True)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()

