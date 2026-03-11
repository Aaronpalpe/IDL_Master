import os
import time
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} B"


def main():
    # Load MobileViT model
    model_name = "apple/mobilevit-small"
    model = MobileViTForImageClassification.from_pretrained(
        model_name, num_labels=10, ignore_mismatched_sizes=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Synthetic dataset (CPU tensors) - MobileViT expects 256x256 images
    batch_size = 1
    num_samples = 8

    input_images = torch.rand((num_samples, 3, 256, 256), device="cpu")
    labels = torch.randint(0, 10, (num_samples,), device="cpu")

    dataset = TensorDataset(input_images, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    # Accelerate: config_ipexbase.yaml drives cpu=True, ipex=True, mixed_precision=bf16
    profile_kwargs = ProfileKwargs(
        activities=["cpu"],
        record_shapes=True,
    )
    accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
    device = accelerator.device

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    # Memory baseline
    cpu_rss_start = get_rss_bytes()
    cpu_rss_peak = cpu_rss_start

    # Training + profiling
    num_epochs = 3
    total_seen_local = 0
    t0 = time.time()

    with accelerator.profile() as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

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
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", flush=True)

    accelerator.wait_for_everyone()
    t1 = time.time()

    # Global totals
    global_seen = int(
        accelerator.reduce(torch.tensor(total_seen_local, device=device), reduction="sum").item()
    )

    elapsed = t1 - t0
    local_throughput = total_seen_local / elapsed if elapsed > 0 else float("nan")
    global_throughput = global_seen / elapsed if elapsed > 0 else float("nan")

    cpu_rss_end = get_rss_bytes()

    if accelerator.is_main_process:
        print(
            f"\nDone. world_size={accelerator.num_processes} global_images={global_seen} "
            f"time={elapsed:.2f}s global_throughput={global_throughput:.1f} img/s\n",
            flush=True,
        )

    # Print each rank stats serially
    accelerator.wait_for_everyone()
    for rank in range(accelerator.num_processes):
        if accelerator.process_index == rank:
            print("=" * 70, flush=True)
            print(
                f"[RANK {accelerator.process_index}/{accelerator.num_processes}] device={device}\n"
                f"  local_images={total_seen_local} local_throughput={local_throughput:.1f} img/s\n"
                f"  CPU RSS: start={format_bytes(cpu_rss_start)}  end={format_bytes(cpu_rss_end)}  "
                f"peak~={format_bytes(cpu_rss_peak)}",
                flush=True,
            )
            print("-" * 70, flush=True)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
            print("=" * 70, flush=True)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
