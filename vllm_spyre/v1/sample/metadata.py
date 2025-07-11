# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This is a copy of the vLLM vllm file prior to PR
# https://github.com/vllm-project/vllm/pull/16728
from dataclasses import dataclass
from typing import Optional

import torch

# TODO: Figure out if we want to apply the LogitsProcessor
# approach here and whether we want to reuse the code. That
# would require some refactoring since processors like the
# MinPLogitsProcessor have a device tensor and a CPU tensor,
# which we don't need


@dataclass
class SamplingMetadata:

    temperature: Optional[torch.Tensor]
    all_greedy: bool
    all_random: bool

    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]
    min_p: Optional[torch.Tensor]

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # req_index -> (min_tokens, stop_token_ids)
    min_tokens: dict[int, tuple[int, set[int]]]

    logit_bias: list[Optional[dict[int, float]]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: Optional[torch.Tensor]

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]
