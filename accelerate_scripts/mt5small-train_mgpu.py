"""
mt5-small finetuning for EN→ES translation on Europarl (multi-GPU).
Focus: measure training time and per-rank profiling stats.
"""
import os
import time
import socket
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator, ProfileKwargs


def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    model_name = "google/mt5-small"
    per_device_batch_size = 8
    max_length = 128
    num_epochs = 3
    num_samples = 4096
    lr = 5e-5

    # -------------------------------
    # Accelerate + profiling
    # -------------------------------
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

    # -------------------------------
    # Tokenizer & model
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------
    # Europarl EN-ES dataset (main process downloads first)
    # -------------------------------
    with accelerator.main_process_first():
        dataset = load_dataset(
            "europarl_bilingual", lang1="en", lang2="es", split="train"
        )
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        prefix = "translate English to Spanish: "

        def preprocess(examples):
            sources = [prefix + ex["en"] for ex in examples["translation"]]
            targets = [ex["es"] for ex in examples["translation"]]
            model_inputs = tokenizer(
                sources, max_length=max_length, truncation=True, padding="max_length"
            )
            labels = tokenizer(
                text_target=targets, max_length=max_length, truncation=True, padding="max_length"
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        dataset.set_format("torch")

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
    warmup_steps = 3
    it = iter(dataloader)
    for _ in range(warmup_steps):
        batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------------
    # Training + profiling
    # -------------------------------
    total_seen_local = 0
    if accelerator.is_main_process:
        print(f"Training mt5-small | world_size={world} "
              f"per_device_bs={per_device_batch_size} samples={num_samples} "
              f"epochs={num_epochs}", flush=True)

    t0 = time.time()

    with accelerator.profile() as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1
                total_seen_local += batch["input_ids"].size(0)

            avg_loss = torch.tensor(
                running_loss / max(num_batches, 1), device=device
            )
            avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", flush=True)

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    # -------------------------------
    # Global stats
    # -------------------------------
    global_seen = int(
        accelerator.reduce(
            torch.tensor(total_seen_local, device=device), reduction="sum"
        ).item()
    )
    local_throughput = total_seen_local / (t1 - t0) if (t1 - t0) > 0 else float("nan")
    global_throughput = global_seen / (t1 - t0) if (t1 - t0) > 0 else float("nan")

    if accelerator.is_main_process:
        print(
            f"\nDone. world_size={world} global_samples={global_seen} "
            f"time={t1-t0:.2f}s global_throughput={global_throughput:.1f} samples/s\n",
            flush=True,
        )

    # Per-rank profiler output (serialized to avoid interleaving)
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
                f"[RANK {rank}/{world}] host={host} local_rank={local_rank} "
                f"device={device} gpu={gpu_name} "
                f"local_samples={total_seen_local} "
                f"local_throughput={local_throughput:.1f} samples/s",
                flush=True,
            )
            print(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=15),
                flush=True,
            )
            print("=" * 70, flush=True)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
