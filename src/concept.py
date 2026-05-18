"""H2 — disease-specific concept neurons (paper §4).

For each disease:
  1. Compute mean per-neuron activation on disease-positive sentences and
     disease-negative sentences (any token; max-aggregate per neuron per
     sentence to capture peak activation as in the paper's "max-activation" probe).
  2. Rank by standardized margin:
         margin = (mean_pos - mean_neg) / pooled_std
  3. Validate top neurons by additive amplification on benign prompts:
         h_i ← h_i + m   for m in a sweep
     and check whether disease-relevant vocabulary appears in the generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .hooks import additive_intervention
from .model import LoadedModel, clear_h


@dataclass
class ConceptNeuron:
    """Top-ranked concept neuron for a disease."""
    disease: str
    layer: int
    neuron: int
    mean_pos: float
    mean_neg: float
    margin: float    # standardized (mean_pos − mean_neg) / pooled_std


@torch.no_grad()
def _max_activations(
    lm: LoadedModel,
    sentences: Sequence[str],
    layer_indices: Sequence[int],
    max_length: int = 96,
) -> Dict[int, torch.Tensor]:
    """Per-neuron max-over-tokens activation for each sentence.

    Returns ``{layer_idx: Tensor[n_sentences, d_ff]}``.
    """
    out: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_indices}
    tok = lm.tokenizer
    for s in tqdm(sentences, desc="concept-activ", leave=False):
        enc = tok(s, return_tensors="pt", truncation=True, max_length=max_length).to(lm.device)
        lm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        for L in layer_indices:
            h = lm.layers[L].mlp._h.detach().float()          # [1, T, d_ff]
            # Signed value at the token where |h| is maximal (per neuron).
            argmax = h.abs().argmax(dim=1, keepdim=True)       # [1, 1, d_ff]
            signed_max = h.gather(1, argmax).squeeze(1).squeeze(0)  # [d_ff]
            out[L].append(signed_max)
        clear_h(lm.layers)
    return {L: torch.stack(out[L], dim=0) for L in layer_indices}


def rank_concept_neurons(
    lm: LoadedModel,
    positive: Sequence[str],
    negative: Sequence[str],
    layer_range: Optional[Tuple[int, int]] = None,
    top_k: int = 3,
    disease: str = "",
) -> List[ConceptNeuron]:
    """Top-k MLP neurons whose mean activation on ``positive`` exceeds ``negative``."""
    if layer_range is None:
        layer_range = (0, lm.n_layers)
    layer_indices = list(range(*layer_range))

    act_pos = _max_activations(lm, positive, layer_indices)
    act_neg = _max_activations(lm, negative, layer_indices)

    # Aggregate margin across layers, return top-k overall.
    scored: List[ConceptNeuron] = []
    for L in layer_indices:
        ap = act_pos[L]                                       # [N_pos, d_ff]
        an = act_neg[L]                                       # [N_neg, d_ff]
        mean_p = ap.mean(0)
        mean_n = an.mean(0)
        var_p = ap.var(0, unbiased=False)
        var_n = an.var(0, unbiased=False)
        pooled = torch.sqrt(0.5 * (var_p + var_n) + 1e-6)
        margin = (mean_p - mean_n) / pooled
        topk = torch.topk(margin, k=min(top_k, margin.numel()))
        for m_val, n_idx in zip(topk.values.tolist(), topk.indices.tolist()):
            scored.append(
                ConceptNeuron(
                    disease=disease,
                    layer=L,
                    neuron=n_idx,
                    mean_pos=float(mean_p[n_idx]),
                    mean_neg=float(mean_n[n_idx]),
                    margin=float(m_val),
                )
            )
    scored.sort(key=lambda c: c.margin, reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Amplification validation
# ---------------------------------------------------------------------------


@dataclass
class AmplificationResult:
    """One (disease, neuron, benign-prompt, multiplier) generation."""
    disease: str
    layer: int
    neuron: int
    prompt: str
    multiplier: float
    generation: str
    mentions_concept: bool


@torch.no_grad()
def amplification_matrix(
    lm: LoadedModel,
    neuron: ConceptNeuron,
    benign_prompts: Sequence[str],
    multipliers: Sequence[float] = (0.0, 10.0, 40.0, 80.0, 160.0),
    max_new_tokens: int = 64,
    concept_keywords: Optional[Sequence[str]] = None,
) -> List[AmplificationResult]:
    """Additive amplification on benign prompts; flag mentions of ``concept_keywords``."""
    tok = lm.tokenizer
    keywords = [k.lower() for k in (concept_keywords or [neuron.disease.lower()])]
    out: List[AmplificationResult] = []
    for prompt in benign_prompts:
        for m in multipliers:
            with additive_intervention(lm.layers, neuron.neuron, float(m), neuron.layer):
                enc = tok(prompt, return_tensors="pt").to(lm.device)
                gen_ids = lm.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                text = tok.decode(gen_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            clear_h(lm.layers)
            tl = text.lower()
            mentions = any(k in tl for k in keywords)
            out.append(
                AmplificationResult(
                    disease=neuron.disease,
                    layer=neuron.layer,
                    neuron=neuron.neuron,
                    prompt=prompt,
                    multiplier=float(m),
                    generation=text,
                    mentions_concept=mentions,
                )
            )
    return out


# Default concept keywords for each disease (used by amplification_matrix when
# the caller doesn't override).
DISEASE_KEYWORDS: Dict[str, List[str]] = {
    "sepsis": ["sepsis", "septic", "lactate", "vasopressor"],
    "t2dm": ["diabetes", "diabetic", "hba1c", "glucose", "insulin"],
    "mi": ["myocardial", "infarction", "st-elevation", "troponin", "stemi", "heart attack"],
    "pneumonia": ["pneumonia", "consolidation", "lobar"],
    "asthma": ["asthma", "wheez", "bronchospasm", "albuterol"],
    "depression": ["depression", "depressive", "anhedonia", "suicid"],
}
