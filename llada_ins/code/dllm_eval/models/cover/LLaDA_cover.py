import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from tqdm import tqdm

from dllm_eval.api.instance import Instance
from dllm_eval.api.registry import register_model
from dllm_eval.models.LLaDA import LLaDA as BaseLLaDA
from dllm_eval.models.utils import get_dtype

from .contextual_cover import UGAGenerator
from .modeling_llada_kv_cover import LLaDAModelLM as CoverLLaDAModelLM
from .inference import _normalize_cover_gen_kwargs


eval_logger = logging.getLogger(__name__)


@register_model("LLaDA_cover")
class LLaDA_cover(BaseLLaDA):
    """
    LLaDA wrapper that uses the COVER model implementation + verification-based generator.

    Notes:
      - The COVER generator supports batch_size > 1 with left-padding.
        With variable-length prompts, left-padding introduces different RoPE positions
        vs the unpadded single-sample case, so results may not be byte-identical to
        batch_size=1 for prompts of different lengths.
    """

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if autogptq or gptqmodel:
            raise NotImplementedError("Quantization options are not implemented for this custom class.")

        model_dtype = get_dtype(dtype)
        eval_logger.info(f"Loading COVER model with dtype: {model_dtype}")

        model_kwargs = kwargs if kwargs else {}
        if not parallelize:
            model_kwargs.update(
                self._get_accelerate_args(
                    parallelize=parallelize,
                    gpus=gpus,
                    max_memory_per_gpu=max_memory_per_gpu,
                    max_cpu_memory=max_cpu_memory,
                    offload_folder=offload_folder,
                )
            )

        # Use the COVER local implementation.
        self._model = CoverLLaDAModelLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )

        if peft:
            from peft import PeftModel

            eval_logger.info(f"Loading PEFT model from {peft}")
            self._model = PeftModel.from_pretrained(self._model, peft, torch_dtype=model_dtype)

        if not parallelize:
            self._model = self._model.to(self.device)
        self._model.eval()

    def _get_verify_generator(self) -> UGAGenerator:
        if getattr(self, "_cover_generator", None) is None:
            self._cover_generator = UGAGenerator(self.model, self.tokenizer, mask_id=self.mask_id)
        return self._cover_generator

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running COVER generate_until requests")

        ds_data = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds_data)

        gen_kwargs: Dict[str, Any] = requests[0].args[1] or {}
        # Provide sensible defaults while letting user provided kwargs win.
        defaults: Dict[str, Any] = {
            "gen_length": gen_kwargs.get("gen_length", 256),
            "block_length": gen_kwargs.get("block_length", 64),
            "temperature": gen_kwargs.get("temperature", 0.0),
            # "threshold" kept for backwards-compat; _normalize will map it to tau_draft if needed.
            "tau_draft": gen_kwargs.get("tau_draft", gen_kwargs.get("threshold", 0.7)),
            "max_unmask_per_step": gen_kwargs.get("max_unmask_per_step", 5),
            "max_reverify_times": gen_kwargs.get("max_reverify_times", 1),
            "use_attention_score": gen_kwargs.get("use_attention_score", False),
            "debug": gen_kwargs.get("debug", False),
        }
        call_kwargs = _normalize_cover_gen_kwargs({**defaults, **gen_kwargs})

        req_offset = 0
        for batch in ds.iter(batch_size=int(self.batch_size)):
            t0 = time.perf_counter()
            contexts = batch["text"]
            context_enc, _attn_masks = self.tok_batch_encode(contexts, truncation=self.truncation)
            prompt_length = int(context_enc.shape[1])

            result = self._get_verify_generator().generate_causal(
                context_enc, attention_mask=_attn_masks, **call_kwargs
            )
            if isinstance(result, tuple) and len(result) >= 3:
                out_full, nfe, flip_flop_stats = result[0], result[1], result[2]
            elif isinstance(result, tuple) and len(result) >= 2:
                out_full, nfe, flip_flop_stats = result[0], result[1], None
            else:
                out_full, nfe, flip_flop_stats = result, None, None

            generated_tokens = out_full[:, prompt_length:]
            cont_toks_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

            t1 = time.perf_counter()
            batch_time_s = t1 - t0
            batch_reqs = requests[req_offset : req_offset + len(cont_toks_list)]
            per_req_time_s = (batch_time_s / len(batch_reqs)) if batch_reqs else None
            def _stat_for_sample(stats: Dict[str, Any], key: str, index: int) -> Any:
                value = stats.get(key)
                if isinstance(value, list):
                    return value[index] if index < len(value) else None
                return value

            for i, req in enumerate(batch_reqs):
                req.inference_steps = int(nfe) if nfe is not None else None
                req.inference_time_seconds = per_req_time_s
                if flip_flop_stats is not None:
                    req.flip_flop_count = _stat_for_sample(flip_flop_stats, "flip_flop_count", i)
                    req.flip_flop_remask_count = _stat_for_sample(flip_flop_stats, "total_remask_count", i)
                    req.flip_flop_unmask_count = _stat_for_sample(flip_flop_stats, "total_unmask_count", i)
                    req.keep_count = _stat_for_sample(flip_flop_stats, "keep_count", i)
                    req.seed_total_count = _stat_for_sample(flip_flop_stats, "seed_total_count", i)
                    req.deferred_remask_count = _stat_for_sample(flip_flop_stats, "deferred_remask_count", i)
                    req.replace_count = _stat_for_sample(flip_flop_stats, "replace_count", i)
                    req.changed_after_remask_count = _stat_for_sample(
                        flip_flop_stats, "changed_after_remask_count", i
                    )
                else:
                    req.flip_flop_count = None
                    req.flip_flop_remask_count = None
                    req.flip_flop_unmask_count = None
                    req.keep_count = None
                    req.seed_total_count = None
                    req.deferred_remask_count = None
                    req.replace_count = None
                    req.changed_after_remask_count = None
            req_offset += len(batch_reqs)

            for s in cont_toks_list:
                if not self.escape_until:
                    stop_sequences = gen_kwargs.get("until", [])
                    if stop_sequences:
                        for term in stop_sequences:
                            if len(term) > 0:
                                s = s.split(term)[0]
                res.append(s)
                bar.update(1)

        bar.close()

        return res
