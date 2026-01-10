"""Minimal test to isolate gradient issue."""
import torch
import sys

if not torch.cuda.is_available():
    print("CUDA not available, exiting")
    sys.exit(1)

device = torch.device("cuda")
print(f"Device: {device}")

from tks_llm_core_v6 import TKSGeneralLMv6, TKSGeneralConfig

config = TKSGeneralConfig(
    vocab_size=1000,
    hidden_dim=384,
    num_layers=2,
    use_njt=True,  # Enable NJT
    njt_use_nested_memory=True,  # Test nested memory
)
model = TKSGeneralLMv6(config).to(device)

# Simulate training loop with optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

print("Testing multiple training steps with attention_mask...")
for step in range(5):
    input_ids = torch.randint(0, 100, (4, 32)).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float)  # Add mask like train_v6.py

    output = model(input_ids, attention_mask=attention_mask, step=step, return_full_trace=(step == 0))
    logits = output["logits"]

    # Compute loss
    labels = input_ids.clone()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
    )

    # Add aux loss if present
    aux_loss = output.get("routing_aux_loss", 0)
    total_loss = loss + aux_loss

    try:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(f"Step {step}: loss = {loss.item():.4f} OK")
    except RuntimeError as e:
        print(f"Step {step} FAILED: {e}")
        sys.exit(1)

print("All steps passed!")
