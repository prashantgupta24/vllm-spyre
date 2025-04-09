"""Verification of continuous batching

Run `python -m pytest tests/test_spyre_cb.py`.
"""

import pytest
from spyre_util import (
    compare_results,
    generate_hf_output,
    generate_cb_spyre_vllm_output,
    get_spyre_backend_list,
    get_spyre_model_list,
)
from vllm import SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cb_handling(
    model: str,
    backend: str,
):
    """Test that the spyre worker correctly handles batches of requests that
    finish after different numbers of forward passes"""

    max_tokens1 = 20
    max_tokens2 = 17
    max_tokens3 = 11

    sampling_params1 = SamplingParams(max_tokens=max_tokens1,
                                  temperature=0.0,
                                  stop="1",
                                  ignore_eos=True)

    sampling_params2 = SamplingParams(max_tokens=max_tokens2,
                                    temperature=0.0,
                                    stop="1",
                                    ignore_eos=True)

    sampling_params3 = SamplingParams(max_tokens=max_tokens3,
                                    temperature=0.0,
                                    stop="1",
                                    ignore_eos=True)

    vllm_sampling_params = [
        sampling_params1,
        sampling_params2,
        sampling_params3,
    ]

    # Have the model count down to one and stop
    # vllm_sampling_params = SamplingParams(
    #     max_tokens=20, temperature=0, stop="1", logprobs=0
    # )
    # Importantly, these prompts are ordered so that they don't finish in the
    # order given
    prompts = [
        "7 6 5 4",
        "10 9 8 7",
        "8 7 6 5",
    ]

    # Ensure that both:
    # - The model doesn't crash
    # - The output sequences are correct
    vllm_results = generate_cb_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
    )

    print(vllm_results)

    assert vllm_results[0]["text"] == " 3 2 "
    assert vllm_results[1]["text"] == " 6 5 4 3 2 "
    assert vllm_results[2]["text"] == " 4 3 2 "
