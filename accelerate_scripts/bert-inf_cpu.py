from accelerate import Accelerator, ProfileKwargs
import torch
from transformers import BertModel, BertTokenizer

# Load BERT model and tokenizer
model_name = "bert-base-cased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Create a large batch of random long sentences (batch size 16, sequence length 512)
batch_size = 128  # Reduced for CPU efficiency
seq_length = 512

# Generate random token IDs within the model's vocabulary size
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
attention_mask = torch.ones_like(input_ids)  # Assume all tokens are attended to

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
input_ids = input_ids.to(accelerator.device)
attention_mask = attention_mask.to(accelerator.device)

# Profile the model execution on the CPU
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Print profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

