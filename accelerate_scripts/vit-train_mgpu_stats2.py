import os, time, socket, platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import ViTForImageClassification
from accelerate import Accelerator, ProfileKwargs


# -------- CPU RSS helpers (best effort) --------
def get_rss_bytes():
    """Current process RSS in bytes (best effort)."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system().lower() == "linux":
            return int(usage.ru_maxrss * 1024)
        return int(usage.ru_maxrss)
    except Exception:
        return None


def format_bytes(n):
    if n is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} B"


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

    # ---- track memory baseline ----
    cpu_rss_start = get_rss_bytes()
    cpu_rss_peak = cpu_rss_start

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)

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

    # Warmup
    warmup_steps = 5
    it = iter(dataloader)
    for _ in range(warmup_steps):
        inputs, targets = next(it)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()

        rss = get_rss_bytes()
        if rss is not None and (cpu_rss_peak is None or rss > cpu_rss_peak):
            cpu_rss_peak = rss

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
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()

                running_loss += loss.item()
                total_seen_local += inputs.size(0)

                rss = get_rss_bytes()
                if rss is not None and (cpu_rss_peak is None or rss > cpu_rss_peak):
                    cpu_rss_peak = rss

            avg_loss = torch.tensor(running_loss / len(dataloader), device=device)
            avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    global_seen = total_seen_local * world

    # ---- Memory stats ----
    gpu_alloc = gpu_peak = None
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated(device)
        gpu_peak = torch.cuda.max_memory_allocated(device)

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Done. world_size={world} global_images={global_seen} "
              f"time={t1-t0:.2f}s throughput={global_seen/(t1-t0):.1f} img/s")
        print(f"[rank={rank} host={host} local_rank={local_rank}]")
        print(f"CPU RSS start : {format_bytes(cpu_rss_start)}")
        print(f"CPU RSS peak  : {format_bytes(cpu_rss_peak)}")
        if gpu_alloc is not None:
            print(f"GPU mem alloc : {format_bytes(gpu_alloc)}")
            print(f"GPU mem peak  : {format_bytes(gpu_peak)}")
        print(f"{'='*60}\n")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    accelerator.end_training()

if __name__ == "__main__":
    main()
