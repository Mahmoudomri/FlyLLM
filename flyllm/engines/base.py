"""
FlyLLM - Abstract Base Engine
"""
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from ..config import ModelConfig


class BaseEngine(ABC):

    def __init__(self, flyllm_dir: str, hf_dir: str,
                 cfg: ModelConfig, verbose: bool = True):
        self.flyllm_dir = flyllm_dir   # compressed layers
        self.hf_dir     = hf_dir       # original HF files (embed, norm, lm_head)
        self.cfg        = cfg
        self.verbose    = verbose
        self.cache      = {}

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    def generate_tokens(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
        stream:         bool  = False,
        stop_ids:       list  = None,
    ):
        # Wipe any KV cache left over from a previous generation
        if hasattr(self, "reset_cache"):
            self.reset_cache()

        if self.verbose:
            print(f"  Starting generation — prefilling {input_ids.shape[1]} prompt tokens...")

        generated   = input_ids.clone()
        next_input  = input_ids        # full prompt on step 0 (prefill)

        for step in range(max_new_tokens):
            if self.verbose and step == 0:
                print(f"  Running prefill forward pass (32 layers)...")
            logits     = self.forward(next_input)   # only NEW tokens go through
            next_logit = logits[0, -1, :]

            if temperature > 0 and temperature != 1.0:
                next_logit = next_logit / temperature

            if top_p < 1.0 and temperature > 0:
                sorted_l, sorted_i = torch.sort(next_logit, descending=True)
                probs    = F.softmax(sorted_l.float(), dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask     = cumprobs - probs > top_p
                sorted_l[mask] = float("-inf")
                next_logit = torch.zeros_like(next_logit).scatter_(0, sorted_i, sorted_l)
                probs   = F.softmax(next_logit.float(), dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = next_logit.argmax().item()

            next_tok   = torch.tensor([[next_id]])
            generated  = torch.cat([generated, next_tok], dim=1)
            next_input = next_tok      # decode steps only ever pass 1 new token

            if self.verbose:
                print(f"  token {step+1}/{max_new_tokens}", end="\r", flush=True)

            yield next_id

            if next_id == self.cfg.eos_token_id:
                break
            if stop_ids and next_id in stop_ids:
                break

        if self.verbose:
            print(" " * 40, end="\r")  # clear the progress line
