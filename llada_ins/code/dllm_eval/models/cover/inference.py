from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
import transformers

from .contextual_cover import UGAGenerator
from .modeling_llada_kv_cover import LLaDAModelLM


def init_cover_model(
    pretrained: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
    **from_pretrained_kwargs: Any,
) -> transformers.PreTrainedModel:
    """
    Initialize the COVER LLaDA model using the local implementation in this folder.
    """
    return LLaDAModelLM.from_pretrained(
        pretrained,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **from_pretrained_kwargs,
    )


def init_cover_tokenizer(
    pretrained: str,
    *,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    add_bos_token: bool | None = None,
    **tokenizer_kwargs: Any,
) -> transformers.PreTrainedTokenizerBase:
    """
    Initialize the tokenizer for a COVER run.
    """
    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "use_fast": use_fast,
    }
    if add_bos_token is not None:
        kwargs["add_bos_token"] = add_bos_token
    kwargs.update(tokenizer_kwargs)
    return transformers.AutoTokenizer.from_pretrained(pretrained, **kwargs)


def init_cover_engine(
    pretrained: str,
    *,
    device: str | torch.device | None = None,
    mask_id: int = 126336,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
    **from_pretrained_kwargs: Any,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase, UGAGenerator]:
    """
    Convenience helper that returns (model, tokenizer, generator).
    """
    tokenizer = init_cover_tokenizer(
        pretrained,
        trust_remote_code=trust_remote_code,
    )
    model = init_cover_model(
        pretrained,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **from_pretrained_kwargs,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer, UGAGenerator(model, tokenizer, mask_id=mask_id)


def _normalize_cover_gen_kwargs(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Backwards-compat: allow "threshold" to mean tau_draft.
    if gen_kwargs.get("tau_draft", None) is None and gen_kwargs.get("threshold", None) is not None:
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs["tau_draft"] = gen_kwargs["threshold"]

    sig = inspect.signature(UGAGenerator.generate_causal)
    allowed = set(sig.parameters.keys()) - {"self", "prompt"}

    out: Dict[str, Any] = {}
    for k in allowed:
        if k in gen_kwargs and gen_kwargs[k] is not None:
            out[k] = gen_kwargs[k]
    return out


@torch.no_grad()
def generate_cover(
    *,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: torch.Tensor,
    mask_id: int = 126336,
    **gen_kwargs: Any,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Run COVER generation. Returns (full_token_ids, actual_steps, flip_flop_stats).
    """
    generator = UGAGenerator(model, tokenizer, mask_id=mask_id)
    call_kwargs = _normalize_cover_gen_kwargs(gen_kwargs)
    return generator.generate_causal(prompt, **call_kwargs)
