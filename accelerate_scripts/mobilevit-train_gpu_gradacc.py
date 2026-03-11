from accelerate import Accelerator, ProfileKwargs
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileViTForImageClassification

# Load MobileViT model
model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Synthetic dataset - MobileViT expects 256x256 images
batch_size = 16
seq_length = 256
input_images = torch.rand((batch_size, 3, seq_length, seq_length))
labels = torch.randint(0, 10, (batch_size,))

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define profiling kwargs for GPU activities
profile_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True
)

# Gradient accumulation steps
gradient_accumulation_steps = 4

# Initialize the accelerator for GPU with gradient accumulation
accelerator = Accelerator(
    cpu=False,
    gradient_accumulation_steps=gradient_accumulation_steps,
    kwargs_handlers=[profile_kwargs]
)

# Prepare the model, optimizer, and data loader for GPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Move model to training mode
model.train()

device = accelerator.device

# Training loop with gradient accumulation
num_epochs = 3
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with accelerator.accumulate(model):
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Print profiling results
print(f"Training completed (GPU + gradient accumulation, accum_steps={gradient_accumulation_steps}).")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
