#!/usr/bin/env python3
"""
Beam search + (optional) grammar constraints for the Story↔Equation seq2seq models.

This repo's seq2seq tokenizer is character-level, so this module implements
beam search over token IDs with efficient Transformer encoder caching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


@dataclass
class BeamHypothesis:
    tokens: List[int]
    log_prob: float
    is_complete: bool = False

    def score(self, length_penalty: float) -> float:
        # Google NMT length penalty (alpha = length_penalty)
        lp = ((5 + len(self.tokens)) / 6) ** max(length_penalty, 0.0)
        return self.log_prob / lp


class _TransformerCache:
    def __init__(self, memory: torch.Tensor, src_key_padding_mask: torch.Tensor):
        self.memory = memory
        self.src_key_padding_mask = src_key_padding_mask


def _is_transformer_seq2seq(model: torch.nn.Module) -> bool:
    return all(hasattr(model, attr) for attr in ("transformer", "encoder_embedding", "decoder_embedding"))


@torch.no_grad()
def _encode_transformer(model: torch.nn.Module, encoder_input: torch.Tensor) -> _TransformerCache:
    batch_size = encoder_input.size(0)
    enc_seq_len = encoder_input.size(1)
    enc_pos = torch.arange(enc_seq_len, device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)
    enc_emb = model.encoder_embedding(encoder_input) + model.encoder_pos(enc_pos)
    src_key_padding_mask = (encoder_input == 0)
    memory = model.transformer.encoder(enc_emb, src_key_padding_mask=src_key_padding_mask)
    return _TransformerCache(memory=memory, src_key_padding_mask=src_key_padding_mask)


@torch.no_grad()
def _decode_transformer_logits(
    model: torch.nn.Module,
    cache: _TransformerCache,
    decoder_input: torch.Tensor,
) -> torch.Tensor:
    batch_size = decoder_input.size(0)
    dec_seq_len = decoder_input.size(1)
    dec_pos = torch.arange(dec_seq_len, device=decoder_input.device).unsqueeze(0).expand(batch_size, -1)
    dec_emb = model.decoder_embedding(decoder_input) + model.decoder_pos(dec_pos)

    tgt_mask = torch.triu(
        torch.ones(dec_seq_len, dec_seq_len, device=decoder_input.device, dtype=torch.bool),
        diagonal=1,
    )
    tgt_key_padding_mask = (decoder_input == 0)

    out = model.transformer.decoder(
        tgt=dec_emb,
        memory=cache.memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=cache.src_key_padding_mask,
    )
    return model.output(out)


class TKSCharEquationConstraint:
    """
    Lightweight char-level grammar for canonical TKS equations:

      ELEMENT (space OP space ELEMENT)* [EOS]

    ELEMENT:
      [A-D] (10|[1-9]) ( ^[1-9] )? ( _d[1-7] )?

    OP token (no spaces inside):
      +  -  o  +T  -T  *T  /T  ->  <-

    This is intentionally strict to keep decoding on-format.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3

        self._id_to_token: Dict[int, str] = getattr(tokenizer, "id_to_token", {})
        self._token_to_id: Dict[str, int] = getattr(tokenizer, "token_to_id", {})

        self.space_id = self._token_to_id.get(" ")
        if self.space_id is None:
            raise ValueError("Tokenizer is missing space token")

        self.world_ids = {self._token_to_id.get(c) for c in "ABCD"} - {None}
        self.digit_ids = {self._token_to_id.get(c) for c in "0123456789"} - {None}
        self.digit_1_9_ids = {self._token_to_id.get(c) for c in "123456789"} - {None}
        self.digit_1_7_ids = {self._token_to_id.get(c) for c in "1234567"} - {None}

        self.caret_id = self._token_to_id.get("^")
        self.underscore_id = self._token_to_id.get("_")
        self.d_id = self._token_to_id.get("d")

        self.plus_id = self._token_to_id.get("+")
        self.minus_id = self._token_to_id.get("-")
        self.star_id = self._token_to_id.get("*")
        self.slash_id = self._token_to_id.get("/")
        self.o_id = self._token_to_id.get("o")
        self.lt_id = self._token_to_id.get("<")
        self.gt_id = self._token_to_id.get(">")
        self.T_id = self._token_to_id.get("T")

        required = [
            ("^", self.caret_id),
            ("_", self.underscore_id),
            ("d", self.d_id),
            ("+", self.plus_id),
            ("-", self.minus_id),
            ("*", self.star_id),
            ("/", self.slash_id),
            ("o", self.o_id),
            ("<", self.lt_id),
            (">", self.gt_id),
            ("T", self.T_id),
        ]
        missing = [name for name, tid in required if tid is None]
        if missing:
            raise ValueError(f"Tokenizer missing required chars for constraints: {missing}")

    def _tokens_to_text(self, tokens: Sequence[int]) -> str:
        pieces: List[str] = []
        for t in tokens:
            if t in (self.pad_id, self.bos_id):
                continue
            if t == self.eos_id:
                break
            pieces.append(self._id_to_token.get(int(t), ""))
        return "".join(pieces)

    def _element_next_char_ids(self, partial: str) -> Tuple[Set[int], bool]:
        """
        Returns (allowed_next_char_token_ids, eos_ok) while building an element token.
        """
        if partial == "":
            return (set(self.world_ids), False)

        w = partial[0]
        if w not in "ABCD":
            return (set(), False)

        if len(partial) == 1:
            return (set(self.digit_1_9_ids), False)

        rem = partial[1:]
        if not rem:
            return (set(self.digit_1_9_ids), False)

        # Parse noetic (1-10)
        first = rem[0]
        if first in "23456789":
            rest = rem[1:]
        elif first == "1":
            if len(rem) == 1:
                # Could be "1" or "10"
                allowed: Set[int] = set()
                allowed |= {self._token_to_id["0"]}
                if self.caret_id is not None:
                    allowed.add(self.caret_id)
                if self.underscore_id is not None:
                    allowed.add(self.underscore_id)
                allowed.add(self.space_id)
                return (allowed, True)

            second = rem[1]
            if second == "0":
                rest = rem[2:]
            elif second in "^_":
                rest = rem[1:]
            else:
                return (set(), False)
        else:
            return (set(), False)

        # Now parse suffix: (^k)? (_dF)?
        if rest == "":
            allowed = set()
            if self.caret_id is not None:
                allowed.add(self.caret_id)
            if self.underscore_id is not None:
                allowed.add(self.underscore_id)
            allowed.add(self.space_id)
            return (allowed, True)

        if rest == "^":
            return (set(self.digit_1_9_ids), False)

        if rest.startswith("^"):
            if len(rest) == 2 and rest[1] in "123456789":
                allowed = {self.space_id}
                if self.underscore_id is not None:
                    allowed.add(self.underscore_id)
                return (allowed, True)
            if len(rest) >= 3 and rest[1] in "123456789":
                after = rest[2:]
                # allow foundation after sense
                if after == "_":
                    return ({self.d_id}, False)
                if after == "_d":
                    return (set(self.digit_1_7_ids), False)
                if len(after) == 3 and after.startswith("_d") and after[2] in "1234567":
                    return ({self.space_id}, True)
                return (set(), False)

        if rest == "_":
            return ({self.d_id}, False)
        if rest == "_d":
            return (set(self.digit_1_7_ids), False)
        if len(rest) == 3 and rest.startswith("_d") and rest[2] in "1234567":
            return ({self.space_id}, True)

        return (set(), False)

    def _operator_next_char_ids(self, partial: str) -> Set[int]:
        if partial == "":
            return {self.plus_id, self.minus_id, self.o_id, self.star_id, self.slash_id, self.lt_id}

        if partial == "+":
            return {self.T_id, self.space_id}
        if partial == "-":
            return {self.T_id, self.gt_id, self.space_id}
        if partial == "*":
            return {self.T_id}
        if partial == "/":
            return {self.T_id}
        if partial == "<":
            return {self.minus_id}
        if partial in {"o", "+T", "-T", "*T", "/T", "->", "<-"}:
            return {self.space_id}

        # Second character completion
        if partial in {"+T", "-T", "*T", "/T", "->", "<-"}:
            return {self.space_id}

        return set()

    def allowed_next_token_ids(self, tokens_so_far: Sequence[int]) -> Set[int]:
        """
        Return allowed next token IDs given the currently generated token IDs.
        """
        text = self._tokens_to_text(tokens_so_far)

        # Enforce single spaces by disallowing space after space.
        if text.endswith(" "):
            completed = text.strip().split(" ") if text.strip() else []
            next_is_element = (len(completed) % 2 == 0)
            if next_is_element:
                allowed = set(self.world_ids)
            else:
                allowed = self._operator_next_char_ids("")
            return {tid for tid in allowed if tid is not None}

        parts = text.split(" ")
        idx = len(parts) - 1
        partial = parts[-1]
        if idx % 2 == 0:
            allowed, eos_ok = self._element_next_char_ids(partial)
            # EOS allowed only when element could end now (space allowed) and we are not mid-token ambiguity.
            if eos_ok:
                allowed = set(allowed)
                allowed.add(self.eos_id)
            return {tid for tid in allowed if tid is not None}

        allowed = self._operator_next_char_ids(partial)
        return {tid for tid in allowed if tid is not None}


class BeamSearchDecoder:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        *,
        beam_width: int = 5,
        max_new_tokens: int = 128,
        length_penalty: float = 0.6,
        early_stopping: bool = True,
        n_best: int = 1,
        constraint: Optional[TKSCharEquationConstraint] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = int(beam_width)
        self.max_new_tokens = int(max_new_tokens)
        self.length_penalty = float(length_penalty)
        self.early_stopping = bool(early_stopping)
        self.n_best = int(n_best)
        self.constraint = constraint

        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3

    @torch.no_grad()
    def decode_tokens(self, encoder_input: torch.Tensor) -> List[BeamHypothesis]:
        device = encoder_input.device
        self.model.eval()

        # Trim padding for speed
        if encoder_input.dim() == 2 and encoder_input.size(0) == 1:
            row = encoder_input[0].tolist()
            try:
                pad_at = row.index(self.pad_id)
            except ValueError:
                pad_at = len(row)
            pad_at = max(pad_at, 1)
            encoder_input = encoder_input[:, :pad_at]

        cache = None
        if _is_transformer_seq2seq(self.model):
            cache = _encode_transformer(self.model, encoder_input)

        beam: List[BeamHypothesis] = [BeamHypothesis(tokens=[self.bos_id], log_prob=0.0, is_complete=False)]
        completed: List[BeamHypothesis] = []

        for _step in range(self.max_new_tokens):
            if not beam:
                break

            active = [h for h in beam if not h.is_complete]
            if not active:
                break

            # Prepare padded decoder batch
            lengths = torch.tensor([len(h.tokens) for h in active], device=device, dtype=torch.long)
            max_len = int(lengths.max().item())
            dec_in = torch.full((len(active), max_len), self.pad_id, device=device, dtype=torch.long)
            for i, h in enumerate(active):
                dec_in[i, : len(h.tokens)] = torch.tensor(h.tokens, device=device, dtype=torch.long)

            if cache is not None:
                # Expand encoder cache for beam size
                enc_cache = _TransformerCache(
                    memory=cache.memory.expand(len(active), -1, -1),
                    src_key_padding_mask=cache.src_key_padding_mask.expand(len(active), -1),
                )
                logits = _decode_transformer_logits(self.model, enc_cache, dec_in)
            else:
                logits = self.model(encoder_input.expand(len(active), -1), dec_in)

            next_logits = logits[torch.arange(len(active), device=device), lengths - 1, :]
            log_probs = F.log_softmax(next_logits, dim=-1)

            candidates: List[BeamHypothesis] = []
            for i, hyp in enumerate(active):
                lp = log_probs[i]

                if self.constraint is not None:
                    allowed = self.constraint.allowed_next_token_ids(hyp.tokens)
                    mask = torch.full_like(lp, -float("inf"))
                    idxs = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
                    mask[idxs] = 0.0
                    lp = lp + mask

                # Never generate BOS again
                lp[self.bos_id] = -float("inf")

                topk = min(self.beam_width * 2, lp.numel())
                topk_logp, topk_ids = torch.topk(lp, k=topk)
                for logp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                    if math.isinf(logp) and logp < 0:
                        continue
                    new_tokens = hyp.tokens + [int(tid)]
                    new_logp = hyp.log_prob + float(logp)
                    done = int(tid) == self.eos_id
                    new_h = BeamHypothesis(tokens=new_tokens, log_prob=new_logp, is_complete=done)
                    if done:
                        completed.append(new_h)
                    else:
                        candidates.append(new_h)

            # Keep the best unfinished candidates
            candidates.sort(key=lambda h: h.score(self.length_penalty), reverse=True)
            beam = candidates[: self.beam_width]

            if self.early_stopping and completed:
                best_complete = max(completed, key=lambda h: h.score(self.length_penalty))
                best_incomplete = beam[0] if beam else None
                if best_incomplete is None or best_complete.score(self.length_penalty) >= best_incomplete.score(self.length_penalty):
                    break

        final = completed + beam
        final.sort(key=lambda h: h.score(self.length_penalty), reverse=True)
        return final[: max(self.n_best, 1)]

    def decode_text(self, prompt: str, *, max_length: Optional[int] = None) -> str:
        encoder_ids = self.tokenizer.tokenize(prompt, add_bos=True, add_eos=True)
        try:
            enc_len = encoder_ids.index(self.pad_id)
        except ValueError:
            enc_len = len(encoder_ids)
        enc_len = max(enc_len, 1)
        encoder_tensor = torch.tensor([encoder_ids[:enc_len]], dtype=torch.long, device=next(self.model.parameters()).device)

        hyps = self.decode_tokens(encoder_tensor)
        best = hyps[0]
        # Trim BOS, stop at EOS
        return self.tokenizer.decode(best.tokens).strip()

