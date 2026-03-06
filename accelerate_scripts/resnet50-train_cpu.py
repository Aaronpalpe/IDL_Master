from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# Model
# -------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Synthetic dataset (CPU tensors)
# -------------------------------
batch_size = 1
num_samples = 8  # > batch_size so you get multiple steps

input_images = torch.rand((num_samples, 3, 224, 224), device="cpu")
labels = torch.randint(0, 10, (num_samples,), device="cpu")

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,       # CPU parallelism
    pin_memory=False     # no CUDA
)

# -------------------------------
# Accelerate: CPU-only
# -------------------------------
profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True
)

# IMPORTANT: cpu=True forces CPU execution and disables CUDA
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

model.train()
device = accelerator.device   # will be "cpu"

# -------------------------------
# Training + profiling
# -------------------------------
num_epochs = 3

with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)   # correct Accelerate pattern
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# -------------------------------
# Results
# -------------------------------
print("Training completed (CPU-only).")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

accelerator.end_training()

