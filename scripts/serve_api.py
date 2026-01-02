"""
TKS Inference API - FastAPI wrapper for the TKS model.

Usage:
  .\\.venv311\\Scripts\\python.exe scripts/serve_api.py

Then visit http://localhost:8000/docs for interactive API docs.

Endpoints:
  POST /generate     - Generate/complete TKS equations
  POST /eq-to-story  - Convert equation to narrative
  POST /story-to-eq  - Convert story to equation
  POST /validate     - Validate equation canonicity
  GET  /health       - Health check
"""
import json
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.quick_train import SimpleTokenizer, SimpleTransformer
from teacher.validator import CanonicalValidator
from tks_rules.noetics import NOETIC_NAMES
app = FastAPI(title='TKS Inference API', description='API for interacting with the Tootra Kabbalistic System model', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
model = None
tokenizer = None
validator = None
device = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    prompt: str
    generated: str
    full_output: str

class EqToStoryRequest(BaseModel):
    equation: str

class EqToStoryResponse(BaseModel):
    equation: str
    story: str

class StoryToEqRequest(BaseModel):
    story: str

class StoryToEqResponse(BaseModel):
    story: str
    equation_hint: str
    model_expansion: str

class ValidateRequest(BaseModel):
    equation: str

class ValidateResponse(BaseModel):
    equation: str
    is_valid: bool
    canon_score: float
    issues: list

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
WORLD_NAMES = {'A': 'Spiritual', 'B': 'Mental', 'C': 'Emotional', 'D': 'Physical'}
OPERATOR_MEANINGS = {'+': 'combines with', '-': 'separates from', '+T': 'accumulates with', '-T': 'diminishes from', '->': 'transforms into', '<-': 'receives from', '*T': 'amplifies through', '/T': 'attenuates through', 'o': 'composes with'}

def load_default_checkpoint():
    """Load default checkpoint path from config."""
    config_path = Path('config/model_defaults.json')
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        return cfg.get('default_checkpoint') or cfg.get('checkpoint')
    return None

def generate_text(prompt: str, max_length: int=100, temperature: float=0.7) -> str:
    """Generate text continuation from prompt."""
    global model, tokenizer, device
    model.eval()
    tokens = [2]
    for c in prompt:
        tokens.append(tokenizer.token_to_id.get(c, 1))
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token in [0, 3]:
                break
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
            if tokens.shape[1] > tokenizer.max_length - 1:
                break
    return ''.join((id_to_token.get(t, '?') for t in generated))

def parse_element(element: str) -> dict:
    """Parse a TKS element like A1^2_d3."""
    import re
    match = re.match('^([ABCD])(10|[1-9])(\\^[1-9])?(_d[1-7])?$', element)
    if not match:
        return {'world': '?', 'noetic': 0, 'sense': None, 'foundation': None}
    return {'world': match.group(1), 'noetic': int(match.group(2)), 'sense': int(match.group(3)[1]) if match.group(3) else None, 'foundation': match.group(4) if match.group(4) else None}

def equation_to_story(equation: str) -> str:
    """Convert TKS equation to narrative story."""
    parts = equation.split()
    story_parts = []
    for part in parts:
        if part in OPERATOR_MEANINGS:
            story_parts.append(OPERATOR_MEANINGS[part])
        else:
            elem = parse_element(part)
            world_name = WORLD_NAMES.get(elem['world'], 'Unknown')
            noetic_name = NOETIC_NAMES.get(elem['noetic'], 'Unknown')
            desc = f'the {noetic_name} of the {world_name} realm'
            if elem['sense']:
                desc += f" (sense {elem['sense']})"
            if elem['foundation']:
                desc += f" grounded in foundation {elem['foundation']}"
            story_parts.append(desc)
    if len(story_parts) == 1:
        return f'This represents {story_parts[0]}.'
    story = story_parts[0]
    for part in story_parts[1:]:
        story += f' {part}'
    return f'In this TKS working, {story}.'

def story_to_equation_hint(story: str) -> str:
    """Extract TKS hints from a story to help generate equation."""
    hints = []
    story_lower = story.lower()
    if any((w in story_lower for w in ['spirit', 'divine', 'higher', 'soul'])):
        hints.append('A')
    if any((w in story_lower for w in ['mind', 'thought', 'mental', 'idea'])):
        hints.append('B')
    if any((w in story_lower for w in ['emotion', 'feel', 'heart', 'love', 'fear'])):
        hints.append('C')
    if any((w in story_lower for w in ['physical', 'body', 'material', 'earth'])):
        hints.append('D')
    if not hints:
        hints = ['B', 'C']
    return f'{hints[0]}1 -> {hints[-1]}1'

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(status='ok', model_loaded=model is not None, device=str(device) if device else 'none')

@app.post('/generate', response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """Generate/complete TKS equations."""
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    generated = generate_text(request.prompt, request.max_length, request.temperature)
    return GenerateResponse(prompt=request.prompt, generated=generated, full_output=request.prompt + generated)

@app.post('/eq-to-story', response_model=EqToStoryResponse)
async def eq_to_story_endpoint(request: EqToStoryRequest):
    """Convert TKS equation to narrative story."""
    story = equation_to_story(request.equation)
    return EqToStoryResponse(equation=request.equation, story=story)

@app.post('/story-to-eq', response_model=StoryToEqResponse)
async def story_to_eq_endpoint(request: StoryToEqRequest):
    """Convert narrative story to TKS equation."""
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    hint = story_to_equation_hint(request.story)
    expansion = generate_text(hint, max_length=30)
    return StoryToEqResponse(story=request.story, equation_hint=hint, model_expansion=hint + expansion)

@app.post('/validate', response_model=ValidateResponse)
async def validate_endpoint(request: ValidateRequest):
    """Validate if equation is canonical TKS."""
    if validator is None:
        raise HTTPException(status_code=503, detail='Validator not loaded')
    result = validator.validate(request.equation)
    return ValidateResponse(equation=request.equation, is_valid=result.is_valid, canon_score=result.canon_score, issues=result.issues if result.issues else [])

@app.on_event('startup')
async def startup_event():
    """Load model on startup."""
    global model, tokenizer, validator, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    ckpt = load_default_checkpoint()
    if not ckpt:
        for path in ['output/teacher_model_long_v4/final_model.pt', 'output/teacher_model_long_ft/final_model.pt', 'output/teacher_model_v3/final_model.pt']:
            if Path(path).exists():
                ckpt = path
                break
    if not ckpt:
        print('WARNING: No checkpoint found. API will return 503 for generation endpoints.')
        return
    print(f'Loading model from: {ckpt}')
    tokenizer = SimpleTokenizer(max_length=256)
    model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
    checkpoint = torch.load(ckpt, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    validator = CanonicalValidator(strict_mode=True)
    print('Model loaded successfully!')
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)