import torch
ckpt = torch.load('checkpoints/v3_leaderboard/best_model.pt', map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt.get('epoch')}")
