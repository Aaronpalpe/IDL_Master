from accelerate import Accelerator, ProfileKwargs
import torch
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

# Define profiling kwargs for GPU activities
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Profile CUDA (GPU) activities
    record_shapes=True
)

# Initialize the accelerator for GPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Prepare the model for GPU execution
model = accelerator.prepare(model)

# Move inputs to GPU
device = accelerator.device
input_images = input_images.to(device)

# Profile the model execution on the GPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)  # Forward pass

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
