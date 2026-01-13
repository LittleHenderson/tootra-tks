import torch
import torch.nn as nn
import torch.nn.functional as F
from tks_features.noetic_fractal_coherence import NoeticFractalEncoder
from tks_features.lacunary_pattern_flow import CanonicalLacunarity

class CoherenceClassifier(nn.Module):
    def __init__(self, vocab_size: int = 130, embed_dim: int = 64,
                 hidden_dim: int = 128, noetic_dim: int = 40):
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=128)
        # N-grams removed to match checkpoint
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.nf_encoder = NoeticFractalEncoder(
            embed_dim=embed_dim,
            noetic_dim=noetic_dim
        )
        total_features = embed_dim + noetic_dim + 3 # + ngram_features (removed)
        self.coherence_head = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens: torch.Tensor) -> dict:
        x = self.embedding(tokens)
        # N-gram logic removed
        
        encoded = self.encoder(x)
        mask = (tokens != 128).float().unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        nf_output, nf_validity, nf_coords = self.nf_encoder(pooled, return_nf=True)
        
        # Lacunarity simplified for inference test
        lacunarity = self._compute_lacunarity_features(encoded, mask)
        
        features = torch.cat([pooled, nf_output, lacunarity], dim=-1)
        coherence_score = self.coherence_head(features)
        return {'coherence': coherence_score.squeeze(-1)}

    def _compute_lacunarity_features(self, encoded, mask):
        # Dummy implementation sufficient for forward pass shape
        batch_size = encoded.shape[0]
        return torch.zeros(batch_size, 3, device=encoded.device)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CoherenceClassifier().to(device)
    try:
        checkpoint = torch.load("checkpoints/coherence_classifier.pt", map_location=device)
        # Checkpoint might be whole dict or just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()

    # Simple char tokenizer from training script
    char_to_id = {chr(i): i for i in range(128)}
    char_to_id['<PAD>'] = 128
    char_to_id['<UNK>'] = 129

    def tokenize(text):
        ids = []
        for char in text[:256]:
            ids.append(char_to_id.get(char, char_to_id['<UNK>']))
        while len(ids) < 256:
            ids.append(char_to_id['<PAD>'])
        return torch.tensor([ids], dtype=torch.long).to(device)

    # Test cases - Model expects Natural Language STORIES, not equations!
    valid_story = "The equation reveals how the rhythmic pattern of material plane with sense 3 originates from Spiritual-Negative energy in the TKS framework."
    gibberish = "This A1 equation describes how o Spiritual Effect intensifies through C3 Spiritual-Idea D7 energy at intensity 9"
    
    with torch.no_grad():
        score_valid = model(tokenize(valid_story))['coherence'].item()
        score_gibberish = model(tokenize(gibberish))['coherence'].item()
    
    print(f"\nValid Story: '{valid_story}'")
    print(f"Coherence Score: {score_valid:.4f} (Expected > 0.9)")
    
    print(f"\nGibberish: '{gibberish}'")
    print(f"Coherence Score: {score_gibberish:.4f} (Expected < 0.1)")

if __name__ == "__main__":
    test()
