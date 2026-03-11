from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import ViTForImageClassification
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load ViT model
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Synthetic dataset: random images (batch size, 3 channels, 224x224)
batch_size = 1
num_samples = 8

input_images = torch.rand((num_samples, 3, 224, 224), device="cpu")
labels = torch.randint(0, 10, (num_samples,), device="cpu")

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False
)

# Accelerate: CPU-only
profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True
)

accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

model.train()
device = accelerator.device

# Training + profiling
num_epochs = 3

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

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Results
print("Training completed (CPU-only).")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

accelerator.end_training()
