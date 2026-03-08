"""from accelerate import Accelerator, ProfileKwargs

































































































































    main()if __name__ == "__main__":    accelerator.end_training()    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))    print(f"Throughput: {total_samples / (t1-t0):.1f} samples/s")    print(f"Samples processed: {total_samples}")    print(f"\nTraining completed (GPU). Total time: {t1-t0:.2f}s")    # -------------------------------    # Results    # -------------------------------    t1 = time.time()        torch.cuda.synchronize()    if torch.cuda.is_available():            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")            avg_loss = running_loss / max(num_batches, 1)                total_samples += batch["input_ids"].size(0)                num_batches += 1                running_loss += loss.item()                optimizer.step()                accelerator.backward(loss)                loss = outputs.loss                outputs = model(**batch)                optimizer.zero_grad(set_to_none=True)                batch = {k: v.to(device) for k, v in batch.items()}            for batch in dataloader:            num_batches = 0            running_loss = 0.0        for epoch in range(num_epochs):    with accelerator.profile() as prof:    t0 = time.time()          f"batch_size={batch_size} samples={num_samples}")    print(f"Training mt5-small on GPU | epochs={num_epochs} "    total_samples = 0    # -------------------------------    # Training + profiling    # -------------------------------        torch.cuda.synchronize()    if torch.cuda.is_available():        optimizer.step()        accelerator.backward(loss)        loss = outputs.loss        outputs = model(**batch)        optimizer.zero_grad(set_to_none=True)        batch = {k: v.to(device) for k, v in batch.items()}        batch = next(it)    for _ in range(warmup_steps):    it = iter(dataloader)    warmup_steps = 3    # Warmup    device = accelerator.device    model.train()    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)    accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])    )        record_shapes=True,        activities=["cuda"],    profile_kwargs = ProfileKwargs(    # -------------------------------    # Accelerate: single GPU    # -------------------------------    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    dataset.set_format("torch")    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)        return model_inputs        model_inputs["labels"] = labels["input_ids"]        )            text_target=targets, max_length=max_length, truncation=True, padding="max_length"        labels = tokenizer(        )            sources, max_length=max_length, truncation=True, padding="max_length"        model_inputs = tokenizer(        targets = [ex["es"] for ex in examples["translation"]]        sources = [prefix + ex["en"] for ex in examples["translation"]]    def preprocess(examples):    prefix = "translate English to Spanish: "    dataset = dataset.select(range(min(num_samples, len(dataset))))    )        "europarl_bilingual", lang1="en", lang2="es", split="train"    dataset = load_dataset(    # -------------------------------    # Europarl EN-ES dataset    # -------------------------------    optimizer = optim.AdamW(model.parameters(), lr=lr)    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)    tokenizer = AutoTokenizer.from_pretrained(model_name)    # -------------------------------    # Tokenizer & model    # -------------------------------    lr = 5e-5    num_samples = 2000    num_epochs = 3    max_length = 128    batch_size = 8    model_name = "google/mt5-small"    # -------------------------------    # Configuration    # -------------------------------def main():from accelerate import Accelerator, ProfileKwargsfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLMfrom datasets import load_datasetfrom torch.utils.data import DataLoaderimport torch.optim as optimimport torchimport time"""Focus: measure training time, not translation quality.mt5-small finetuning for EN→ES translation on Europarl (single GPU).import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a batch of random images (batch size 128, 3 color channels, 224x224 resolution)
batch_size = 128
input_images = torch.rand((batch_size, 3, 224, 224))  # Random image batch

# Define profiling kwargs for CPU activities
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Profile CPU activities instead of CUDA
    record_shapes=True
)

# Initialize the accelerator for CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Prepare the model for CPU execution
model = accelerator.prepare(model)

# Move inputs to CPU
device = accelerator.device
input_images = input_images.to(device)

# Profile the model execution on the CPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)  # Forward pass

# Print profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
