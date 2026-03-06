# Use the Hugging Face Accelerate GPU release image as the base
FROM huggingface/accelerate:gpu-nightly

# Set the working directory inside the container
WORKDIR /accelerate_gpu
ARG FULLPATH



# Install the required Python packages
RUN conda install -c conda-forge libstdcxx-ng=12.2.0
RUN pip install accelerate pyvim timm transformers intel_extension_for_pytorch
COPY ${FULLPATH}/accelerate_scripts/* .



# Set an entrypoint or command if needed (optional)
CMD ["bash"]
