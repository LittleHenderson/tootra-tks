import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 1) Synthetic data generation
# ----------------------------

ENTITIES = ["king", "queen", "knight", "wizard", "thief"]
PROPS = ["brave", "wise", "honest", "strong", "kind"]
ACTIONS = ["fled", "laughed", "slept", "spoke", "ran"]

def tokenize(text: str) -> List[str]:
    return text.strip().split()

def make_example(rng: random.Random, max_clauses: int = 3) -> Tuple[str, str, int, bool]:
    """
    Returns: (statement, query, label, is_negated_case)
    label: 1 if property holds, 0 otherwise
    is_negated_case: True if the queried property is (single) negated in the statement.
                     (double_not contains 'not' tokens but is logically true)
    """
    ent = rng.choice(ENTITIES)
    action = rng.choice(ACTIONS)

    n = rng.randint(1, max_clauses)
    props = rng.sample(PROPS, k=n)
    qprop = rng.choice(props)

    mode = rng.choices(
        ["plain", "single_not", "double_not", "mixed_and"],
        weights=[0.35, 0.35, 0.10, 0.20],
        k=1
    )[0]

    rel_parts = []
    is_negated_case = False

    if mode == "plain":
        for i, p in enumerate(props):
            rel_parts.append(p if i == 0 else f"and {p}")
        label = 1

    elif mode == "single_not":
        for i, p in enumerate(props):
            if p == qprop:
                rel_parts.append(f"not {p}" if i == 0 else f"and not {p}")
                is_negated_case = True
            else:
                rel_parts.append(p if i == 0 else f"and {p}")
        label = 0

    elif mode == "double_not":
        for i, p in enumerate(props):
            if p == qprop:
                rel_parts.append(f"not not {p}" if i == 0 else f"and not not {p}")
            else:
                rel_parts.append(p if i == 0 else f"and {p}")
        label = 1

    else:  # mixed_and
        if len(props) < 2:
            props = [props[0], rng.choice([p for p in PROPS if p != props[0]])]
            qprop = rng.choice(props)

        neg_p = rng.choice(props)
        label = 1
        for i, p in enumerate(props):
            if p == neg_p:
                rel_parts.append(f"not {p}" if i == 0 else f"and not {p}")
                if p == qprop:
                    is_negated_case = True
                    label = 0
            else:
                rel_parts.append(p if i == 0 else f"and {p}")
                if p == qprop:
                    label = 1

    statement = f"the {ent} who was " + " ".join(rel_parts) + f" {action}"
    query = f"is the {ent} {qprop} ?"

    return statement, query, label, is_negated_case

def build_dataset(n: int, seed: int = 0, max_clauses: int = 3):
    rng = random.Random(seed)
    return [make_example(rng, max_clauses=max_clauses) for _ in range(n)]


# ----------------------------
# 2) Tokenization / vocab
# ----------------------------

SPECIALS = ["[PAD]", "[CLS]", "[SEP]"]

def build_vocab(dataset):
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for st, q, _, _ in dataset:
        for tok in tokenize(st) + tokenize(q):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def encode_pair(statement: str, query: str, vocab: Dict[str,int], max_len: int):
    """
    Input format: [CLS] statement [SEP] query [SEP]
    """
    tokens = ["[CLS]"] + tokenize(statement) + ["[SEP]"] + tokenize(query) + ["[SEP]"]
    ids = [vocab[t] for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
        tokens = tokens[:max_len]

    attn_mask = [1] * len(ids)
    pad_id = vocab["[PAD]"]
    while len(ids) < max_len:
        ids.append(pad_id)
        attn_mask.append(0)
        tokens.append("[PAD]")

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
        tokens
    )

# ----------------------------
# 3) Core transformer pieces
# ----------------------------

def split_heads(x, n_heads):
    B, T, D = x.shape
    Dh = D // n_heads
    return x.view(B, T, n_heads, Dh).transpose(1, 2)  # [B,H,T,Dh]

def merge_heads(x):
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H*Dh)

class MHSA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_attn=False):
        B, T, D = x.shape
        q = split_heads(self.wq(x), self.n_heads)  # [B,H,T,Dh]
        k = split_heads(self.wk(x), self.n_heads)
        v = split_heads(self.wv(x), self.n_heads)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]

        if attn_mask is not None:
            m = attn_mask[:, None, None, :].to(torch.bool)          # [B,1,1,T]
            scores = scores.masked_fill(~m, float("-inf"))

        if self.causal:
            cm = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(cm, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        y = attn @ v
        y = self.wo(merge_heads(y))  # [B,T,D]
        return (y, attn) if return_attn else y

class MLP(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHSA(d_model, n_heads, dropout=dropout, causal=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class SubtractiveTransformerBlock(nn.Module):
    """
    Separate inhibition attention => interpretable inhibition weights.
    """
    def __init__(self, d_model, n_heads, dropout=0.0, inhib_heads=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHSA(d_model, n_heads, dropout=dropout, causal=False)

        self.ln_inhib = nn.LayerNorm(d_model)
        self.inhib_attn = MHSA(d_model, inhib_heads, dropout=dropout, causal=False)

        self.inhib_gate = nn.Linear(d_model, d_model)
        self.alpha_sub = nn.Parameter(torch.tensor(0.10))

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_gate=False, return_inhib_attn=False):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask=attn_mask))

        z = self.ln_inhib(x)
        if return_inhib_attn:
            inhib_content, inhib_attn = self.inhib_attn(z, attn_mask=attn_mask, return_attn=True)
        else:
            inhib_content = self.inhib_attn(z, attn_mask=attn_mask)
            inhib_attn = None

        gate = torch.sigmoid(self.inhib_gate(z))  # [B,T,D]
        x = x - self.alpha_sub * gate * inhib_content

        x = x + self.drop(self.mlp(self.ln2(x)))

        outs = [x]
        if return_gate: outs.append(gate)
        if return_inhib_attn: outs.append(inhib_attn)
        return outs[0] if len(outs) == 1 else tuple(outs)

class TKS4OpBlockV2(nn.Module):
    """
    Fixes Claude's critiques:
      - Separate ADD vs SUB contexts (two different attentions)
      - Independent sigmoid gates (not softmax)
      - Optional inhibition attention return (interpretable)
    """
    def __init__(self, d_model, n_heads, dropout=0.0, eps=1e-6, inhib_heads=1):
        super().__init__()
        self.eps = eps

        self.ln_add = nn.LayerNorm(d_model)
        self.ln_inh = nn.LayerNorm(d_model)

        self.attn_add = MHSA(d_model, n_heads, dropout=dropout, causal=False)
        self.attn_inh = MHSA(d_model, inhib_heads, dropout=dropout, causal=False)

        self.add_proj = nn.Linear(d_model, d_model)
        self.sub_proj = nn.Linear(d_model, d_model)

        self.mul_gate_proj = nn.Linear(d_model, d_model)
        self.div_denom_proj = nn.Linear(d_model, d_model)

        self.g_add = nn.Linear(d_model, d_model)
        self.g_sub = nn.Linear(d_model, d_model)
        self.g_mul = nn.Linear(d_model, d_model)
        self.g_div = nn.Linear(d_model, d_model)

        self.alpha_sub = nn.Parameter(torch.tensor(0.10))
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x, attn_mask=None, return_gate_stats=False, return_inhib_attn=False):
        z_add = self.ln_add(x)
        c_add = self.attn_add(z_add, attn_mask=attn_mask)  # [B,T,D]

        z_inh = self.ln_inh(x)
        if return_inhib_attn:
            c_inh, inh_attn = self.attn_inh(z_inh, attn_mask=attn_mask, return_attn=True)
        else:
            c_inh = self.attn_inh(z_inh, attn_mask=attn_mask)
            inh_attn = None

        p_add = self.add_proj(c_add)
        p_sub = torch.tanh(self.sub_proj(c_inh))

        g_mul = torch.sigmoid(self.mul_gate_proj(c_add))
        p_mul = x * g_mul

        denom = self.eps + F.softplus(self.div_denom_proj(c_add))
        p_div = x / denom

        w_add = torch.sigmoid(self.g_add(z_add))
        w_sub = torch.sigmoid(self.g_sub(z_inh))
        w_mul = torch.sigmoid(self.g_mul(z_add))
        w_div = torch.sigmoid(self.g_div(z_add))

        y = (w_add * p_add) + (w_mul * p_mul) + (w_div * p_div) - (self.alpha_sub * w_sub * p_sub)
        x = x + self.drop(y)

        x = x + self.drop(self.mlp(self.ln2(x)))

        outs = [x]
        if return_inhib_attn: outs.append(inh_attn)
        if return_gate_stats:
            # PATCH 1: Return both scalar stats AND w_sub tensor for probing
            outs.append({
                "w_add_mean": float(w_add.mean().detach().cpu()),
                "w_sub_mean": float(w_sub.mean().detach().cpu()),
                "w_mul_mean": float(w_mul.mean().detach().cpu()),
                "w_div_mean": float(w_div.mean().detach().cpu()),
                "alpha_sub": float(self.alpha_sub.detach().cpu()),
                "w_sub_tensor": w_sub,  # [B,T,D] on device - for gate probe
            })
        return outs[0] if len(outs) == 1 else tuple(outs)


# ----------------------------
# 4) Dataset batching
# ----------------------------

@dataclass
class Batch:
    input_ids: torch.Tensor
    attn_mask: torch.Tensor
    labels: torch.Tensor
    is_negated_case: torch.Tensor
    tokens: List[List[str]]  # for probing

def make_batch(samples, vocab, max_len, device):
    ids_list, mask_list, y_list, neg_list, toks_list = [], [], [], [], []
    for st, q, y, neg_case in samples:
        ids, mask, toks = encode_pair(st, q, vocab, max_len)
        ids_list.append(ids)
        mask_list.append(mask)
        y_list.append(y)
        neg_list.append(1 if neg_case else 0)
        toks_list.append(toks)

    return Batch(
        input_ids=torch.stack(ids_list).to(device),
        attn_mask=torch.stack(mask_list).to(device),
        labels=torch.tensor(y_list, dtype=torch.long, device=device),
        is_negated_case=torch.tensor(neg_list, dtype=torch.long, device=device),
        tokens=toks_list
    )


# ----------------------------
# 5) Full model (classifier)
# ----------------------------

class ClassifierModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=64,
                 dropout=0.0, block_type="baseline", inhib_heads=1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        blocks = []
        for _ in range(n_layers):
            if block_type == "baseline":
                blocks.append(TransformerBlock(d_model, n_heads, dropout=dropout))
            elif block_type == "subtractive":
                blocks.append(SubtractiveTransformerBlock(d_model, n_heads, dropout=dropout, inhib_heads=inhib_heads))
            elif block_type == "4op_v2":
                blocks.append(TKS4OpBlockV2(d_model, n_heads, dropout=dropout, inhib_heads=inhib_heads))
            else:
                raise ValueError("block_type must be baseline | subtractive | 4op_v2")

        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, input_ids, attn_mask=None, return_probe=False, vocab_not_id=None):
        """
        If return_probe:
          - For subtractive: returns last gate + last inhibition attention
          - For 4op_v2: returns last inhibition attention + gate stats dict
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)[None, :].expand(B, T)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        last_gate = None
        last_inh_attn = None
        last_gate_stats = None

        for blk in self.blocks:
            if isinstance(blk, SubtractiveTransformerBlock) and return_probe:
                x, last_gate, last_inh_attn = blk(
                    x, attn_mask=attn_mask,
                    return_gate=True, return_inhib_attn=True
                )
            elif isinstance(blk, TKS4OpBlockV2) and return_probe:
                x, last_inh_attn, last_gate_stats = blk(
                    x, attn_mask=attn_mask,
                    return_inhib_attn=True, return_gate_stats=True
                )
            else:
                x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        cls = x[:, 0, :]
        logits = self.head(cls)

        # Gate stats for subtractive: mean gate on "not" vs other tokens
        gate_not_other = None
        if return_probe and last_gate is not None and vocab_not_id is not None:
            not_pos = (input_ids == vocab_not_id) & (attn_mask.bool() if attn_mask is not None else True)
            other_pos = (~not_pos) & (attn_mask.bool() if attn_mask is not None else True)
            g_not = last_gate[not_pos].mean().item() if not_pos.any() else float("nan")
            g_other = last_gate[other_pos].mean().item() if other_pos.any() else float("nan")
            gate_not_other = (g_not, g_other)

        # PATCH 1B: Gate stats for 4op_v2: mean w_sub on "not" vs other tokens
        if return_probe and last_gate is None and last_gate_stats is not None and vocab_not_id is not None:
            if isinstance(last_gate_stats, dict) and "w_sub_tensor" in last_gate_stats:
                w_sub = last_gate_stats["w_sub_tensor"]  # [B,T,D]
                not_pos = (input_ids == vocab_not_id) & (attn_mask.bool() if attn_mask is not None else True)
                other_pos = (~not_pos) & (attn_mask.bool() if attn_mask is not None else True)
                g_not = w_sub[not_pos].mean().item() if not_pos.any() else float("nan")
                g_other = w_sub[other_pos].mean().item() if other_pos.any() else float("nan")
                gate_not_other = (g_not, g_other)

        if return_probe:
            return logits, gate_not_other, last_inh_attn, last_gate_stats
        return logits


# ----------------------------
# 6) Evaluation + probing
# ----------------------------

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def inhibition_alignment_probe(batch: Batch, inh_attn: Optional[torch.Tensor], vocab_not_id: int):
    """
    PATCH 2: Measures avg inhibition attention weight from each 'not' token position
    to the immediately following token, BUT ONLY if that next token is actually a predicate word.

    This fixes the noise from "not not brave" where next token is "not" not "brave".

    inh_attn: [B,H,T,T] (query position is dim -2)
    Returns: (avg_not_to_pred, n_pairs)
    """
    if inh_attn is None:
        return float("nan"), 0

    input_ids = batch.input_ids
    attn_mask = batch.attn_mask.bool()
    B, T = input_ids.shape

    not_pos = (input_ids == vocab_not_id) & attn_mask  # [B,T]

    prop_set = set(PROPS)

    pairs = []
    for b in range(B):
        idxs = torch.where(not_pos[b])[0].tolist()
        toks = batch.tokens[b]
        for i in idxs:
            j = i + 1
            if j < T and attn_mask[b, j]:
                # PATCH 2: only count if next token is actually a predicate word
                if toks[j] in prop_set:
                    pairs.append((b, i, j))

    if not pairs:
        return float("nan"), 0

    vals = [float(inh_attn[b, :, i, j].mean().cpu()) for (b, i, j) in pairs]
    return sum(vals)/len(vals), len(vals)

@torch.no_grad()
def evaluate(model, data, vocab, max_len, device, batch_size=64):
    model.eval()
    total, correct = 0, 0
    neg_total, neg_correct = 0, 0
    pos_total, pos_correct = 0, 0

    not_id = vocab.get("not", None)

    g_not_vals, g_other_vals = [], []
    inh_align_vals = []

    for i in range(0, len(data), batch_size):
        batch = make_batch(data[i:i+batch_size], vocab, max_len, device)
        logits, gate_stats, inh_attn, gate_stats_dict = model(
            batch.input_ids, batch.attn_mask,
            return_probe=True, vocab_not_id=not_id
        )
        preds = logits.argmax(dim=-1)

        total += batch.labels.numel()
        correct += (preds == batch.labels).sum().item()

        neg_mask = batch.is_negated_case.bool()
        pos_mask = ~neg_mask

        if neg_mask.any():
            neg_total += neg_mask.sum().item()
            neg_correct += (preds[neg_mask] == batch.labels[neg_mask]).sum().item()
        if pos_mask.any():
            pos_total += pos_mask.sum().item()
            pos_correct += (preds[pos_mask] == batch.labels[pos_mask]).sum().item()

        if gate_stats is not None:
            g_not, g_other = gate_stats
            if not math.isnan(g_not): g_not_vals.append(g_not)
            if not math.isnan(g_other): g_other_vals.append(g_other)

        if not_id is not None:
            align, n_pairs = inhibition_alignment_probe(batch, inh_attn, not_id)
            if not math.isnan(align):
                inh_align_vals.append(align)

    acc = correct / max(1, total)
    neg_acc = neg_correct / max(1, neg_total) if neg_total > 0 else float("nan")
    pos_acc = pos_correct / max(1, pos_total) if pos_total > 0 else float("nan")

    g_not_mean = sum(g_not_vals)/len(g_not_vals) if g_not_vals else float("nan")
    g_other_mean = sum(g_other_vals)/len(g_other_vals) if g_other_vals else float("nan")
    inh_align_mean = sum(inh_align_vals)/len(inh_align_vals) if inh_align_vals else float("nan")

    return {
        "acc": acc,
        "neg_acc": neg_acc,
        "pos_acc": pos_acc,
        "neg_gap": (pos_acc - neg_acc) if (neg_total > 0 and pos_total > 0) else float("nan"),
        "gate_not_mean": g_not_mean,
        "gate_other_mean": g_other_mean,
        "inh_not_to_pred_mean": inh_align_mean,
    }


# ----------------------------
# 7) Training
# ----------------------------

def train_one(model, train_data, test_data, vocab, max_len=64,
              steps=2000, batch_size=64, lr=3e-4, device="cpu", eval_every=200):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, steps+1):
        model.train()
        batch_samples = random.sample(train_data, k=batch_size)
        batch = make_batch(batch_samples, vocab, max_len, device)

        logits = model(batch.input_ids, batch.attn_mask, return_probe=False)
        loss = F.cross_entropy(logits, batch.labels)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % eval_every == 0 or step == 1:
            tr = evaluate(model, random.sample(train_data, k=min(2000, len(train_data))), vocab, max_len, device)
            te = evaluate(model, test_data, vocab, max_len, device)
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"train acc {tr['acc']:.3f} (neg {tr['neg_acc']:.3f}) | "
                f"test acc {te['acc']:.3f} (neg {te['neg_acc']:.3f}) | "
                f"gap {te['neg_gap']:.3f} | "
                f"gate_not {te['gate_not_mean']:.3f} vs other {te['gate_other_mean']:.3f} | "
                f"inh(not→pred) {te['inh_not_to_pred_mean']:.3f}"
            )
    return model

def run_experiment(name, block_type, vocab, train_data, test_data,
                   d_model=128, n_heads=4, n_layers=2, inhib_heads=1,
                   max_len=64, dropout=0.0, steps=2000, batch_size=64, lr=3e-4, device="cpu"):
    model = ClassifierModel(
        vocab_size=len(vocab),
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        inhib_heads=inhib_heads,
        max_len=max_len, dropout=dropout,
        block_type=block_type
    )
    print(f"\n=== {name} ===")
    print("Params:", count_params(model))
    train_one(model, train_data, test_data, vocab, max_len=max_len,
              steps=steps, batch_size=batch_size, lr=lr, device=device, eval_every=max(100, steps//10))
    final = evaluate(model, test_data, vocab, max_len, device)
    return final, count_params(model)


# ----------------------------
# 8) Main
# ----------------------------

def main():
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Data
    train_data = build_dataset(10000, seed=seed, max_clauses=3)
    test_data  = build_dataset(1000,  seed=seed+1, max_clauses=3)

    vocab = build_vocab(train_data + test_data)
    print("Vocab size:", len(vocab))

    # Config (weekend-friendly defaults)
    max_len = 64
    d_model = 128
    n_heads = 4
    n_layers = 2
    inhib_heads = 1     # keep small to limit extra compute
    dropout = 0.0
    steps = 2000
    batch_size = 64
    lr = 3e-4

    results = []

    # A) Baseline
    rA, pA = run_experiment(
        "A) Baseline Transformer", "baseline",
        vocab, train_data, test_data,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        inhib_heads=inhib_heads,
        max_len=max_len, dropout=dropout,
        steps=steps, batch_size=batch_size, lr=lr, device=device
    )
    results.append(("Baseline", pA, rA))

    # B) Subtractive
    rB, pB = run_experiment(
        "B) Subtractive Transformer", "subtractive",
        vocab, train_data, test_data,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        inhib_heads=inhib_heads,
        max_len=max_len, dropout=dropout,
        steps=steps, batch_size=batch_size, lr=lr, device=device
    )
    results.append(("Subtractive", pB, rB))

    # C) 4-Op v2
    rC, pC = run_experiment(
        "C) 4-Op v2 (separate ADD/SUB + independent gates)", "4op_v2",
        vocab, train_data, test_data,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        inhib_heads=inhib_heads,
        max_len=max_len, dropout=dropout,
        steps=steps, batch_size=batch_size, lr=lr, device=device
    )
    results.append(("4-Op v2", pC, rC))

    # Summary table
    print("\n=== SUMMARY (TEST) ===")
    header = f"{'Model':<12} {'Params':>10} {'Acc':>7} {'NegAcc':>7} {'PosAcc':>7} {'Gap':>7} {'GateNot':>8} {'GateOth':>8} {'Inh(n→p)':>9}"
    print(header)
    print("-"*len(header))
    for name, params, r in results:
        print(
            f"{name:<12} {params:>10} "
            f"{r['acc']:>7.3f} {r['neg_acc']:>7.3f} {r['pos_acc']:>7.3f} {r['neg_gap']:>7.3f} "
            f"{r['gate_not_mean']:>8.3f} {r['gate_other_mean']:>8.3f} {r['inh_not_to_pred_mean']:>9.3f}"
        )

if __name__ == "__main__":
    main()
