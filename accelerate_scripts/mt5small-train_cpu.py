"""
mt5-small finetuning for EN→ES translation on Europarl (CPU-only).
Focus: measure training time, not translation quality.
"""
import time
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
    batch_size = 2
    max_length = 128
    num_epochs = 1
    num_samples = 100          # small subset for CPU
    lr = 5e-5

    # -------------------------------
    # Tokenizer & model
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------
    # Europarl EN-ES dataset
    # -------------------------------
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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------
    # Accelerate: CPU-only
    # -------------------------------
    profile_kwargs = ProfileKwargs(
        activities=["cpu"],
        record_shapes=True,
    )
    accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    device = accelerator.device

    # -------------------------------
    # Training + profiling
    # -------------------------------
    print(f"Training mt5-small on CPU | epochs={num_epochs} "
          f"batch_size={batch_size} samples={num_samples}")

    t0 = time.time()

    with accelerator.profile() as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            avg_loss = running_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    t1 = time.time()

    # -------------------------------
    # Results
    # -------------------------------
    total_samples = num_samples * num_epochs
    print(f"\nTraining completed (CPU-only). Total time: {t1-t0:.2f}s")
    print(f"Throughput: {total_samples / (t1-t0):.1f} samples/s")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    accelerator.end_training()


if __name__ == "__main__":
    main()
