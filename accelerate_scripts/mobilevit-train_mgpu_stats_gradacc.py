import os, time, socket, platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import MobileViTForImageClassification
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
        # MobileViT expects 256x256 images
        x = torch.rand(3, 256, 256, dtype=torch.float32)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y


def main():
    # Gradient accumulation steps
    gradient_accumulation_steps = 4

    # Profile CUDA kernels
    profile_kwargs = ProfileKwargs(
        activities=["cuda"],
        record_shapes=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[profile_kwargs],
    )
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

    model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-small", num_labels=10, ignore_mismatched_sizes=True
    )

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

        with accelerator.accumulate(model):
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # update cpu "peak" (best effort)
        rss = get_rss_bytes()
        if rss is not None and (cpu_rss_peak is None or rss > cpu_rss_peak):
            cpu_rss_peak = rss

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Training + profiling with gradient accumulation
    total_seen_local = 0
    t0 = time.time()

    with accelerator.profile() as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with accelerator.accumulate(model):
                    outputs = model(pixel_values=inputs).logits
                    loss = criterion(outputs, targets)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()
                total_seen_local += inputs.size(0)

                # update cpu "peak" (best effort)
                rss = get_rss_bytes()
                if rss is not None and (cpu_rss_peak is None or rss > cpu_rss_peak):
                    cpu_rss_peak = rss

            avg_loss = torch.tensor(running_loss / len(dataloader), device=device)
            avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", flush=True)

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    # Global images processed (no gather_object needed)
    global_seen = int(
        accelerator.reduce(torch.tensor(total_seen_local, device=device), reduction="sum").item()
    )

    local_throughput = total_seen_local / (t1 - t0) if (t1 - t0) > 0 else float("nan")
    global_throughput = global_seen / (t1 - t0) if (t1 - t0) > 0 else float("nan")

    # ---- final memory snapshot ----
    cpu_rss_end = get_rss_bytes()

    gpu_name = None
    gpu_alloc = gpu_reserved = gpu_peak_alloc = gpu_peak_reserved = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = None
        gpu_alloc = torch.cuda.memory_allocated()
        gpu_reserved = torch.cuda.memory_reserved()
        gpu_peak_alloc = torch.cuda.max_memory_allocated()
        gpu_peak_reserved = torch.cuda.max_memory_reserved()

    if accelerator.is_main_process:
        print(
            f"\nDone. world_size={world} global_images={global_seen} time={t1-t0:.2f}s "
            f"global_throughput={global_throughput:.1f} img/s "
            f"gradient_accumulation_steps={gradient_accumulation_steps}\n",
            flush=True
        )

    # ---- Print each rank's profiler + memory (serialized) ----
    accelerator.wait_for_everyone()
    for r in range(world):
        if rank == r:
            print("=" * 70, flush=True)
            print(
                f"[RANK {rank}/{world}] host={host} local_rank={local_rank} device={device} gpu={gpu_name}\n"
                f"  local_images={total_seen_local} local_throughput={local_throughput:.1f} img/s\n"
                f"  CPU RSS: start={format_bytes(cpu_rss_start)}  end={format_bytes(cpu_rss_end)}  peak~={format_bytes(cpu_rss_peak)}\n"
                f"  GPU mem: alloc={format_bytes(gpu_alloc)}  reserved={format_bytes(gpu_reserved)}  "
                f"peak_alloc={format_bytes(gpu_peak_alloc)}  peak_reserved={format_bytes(gpu_peak_reserved)}",
                flush=True
            )
            print("-" * 70, flush=True)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
            print("=" * 70, flush=True)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
