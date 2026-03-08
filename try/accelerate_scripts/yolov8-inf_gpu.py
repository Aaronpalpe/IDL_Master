"""
YOLOv8 nano inference benchmark (single GPU).
Uses the underlying PyTorch nn.Module from ultralytics for profiling.
Requires: pip install ultralytics
"""
import time
import torch
from accelerate import Accelerator, ProfileKwargs
from ultralytics import YOLO


def main():
    # -------------------------------
    # Load YOLOv8 nano
    # -------------------------------
    yolo = YOLO("yolov8n.pt")          # downloads weights if needed
    model = yolo.model                  # underlying nn.Module
    model.eval()

    # Random images (YOLOv8 default input: 640x640)
    batch_size = 64
    input_images = torch.rand((batch_size, 3, 640, 640))

    # -------------------------------
    # Accelerate: single GPU
    # -------------------------------
    profile_kwargs = ProfileKwargs(
        activities=["cuda"],
        record_shapes=True,
    )
    accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

    model = accelerator.prepare(model)
    device = accelerator.device
    input_images = input_images.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(input_images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------------
    # Inference + profiling
    # -------------------------------
    num_runs = 10
    print(f"YOLOv8n inference on GPU | batch_size={batch_size} runs={num_runs}")

    t0 = time.time()

    with accelerator.profile() as prof:
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    total_images = batch_size * num_runs
    print(f"\nInference completed (GPU). Total time: {t1-t0:.4f}s")
    print(f"Images processed: {total_images}")
    print(f"Throughput: {total_images / (t1-t0):.1f} images/s")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
