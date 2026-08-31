"""CPU-only Dream tests using a randomly initialized tiny configuration."""

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "DMPO"))

from model_utils import align_logits_and_targets
from fast_samplers.fast_dream import DreamConfig, DreamModel
from fast_samplers.fast_dream import generate as dream_generate


@pytest.fixture
def tiny_dream_model():
    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        mask_token_id=63,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = DreamModel(config)
    model.eval()
    return model


def _always_select_non_mask(logits, *args, **kwargs):
    shape = logits.shape[:-1]
    confidence = torch.ones(shape, dtype=logits.dtype, device=logits.device)
    tokens = torch.ones(shape, dtype=torch.long, device=logits.device)
    return confidence, tokens


def test_model_alignment_preserves_llada_and_shifts_dream():
    logits = torch.randn(2, 5, 7)
    input_ids = torch.arange(10).view(2, 5)
    masked_index = input_ids.remainder(2).bool()

    llada_logits, llada_targets, llada_mask = align_logits_and_targets(
        "llada", logits, input_ids, masked_index
    )
    assert llada_logits is logits
    assert llada_targets is input_ids
    assert llada_mask is masked_index

    dream_logits, dream_targets, dream_mask = align_logits_and_targets(
        "dream", logits, input_ids, masked_index
    )
    assert dream_logits.shape == (2, 4, 7)
    assert torch.equal(dream_logits, logits[:, :-1])
    assert torch.equal(dream_targets, input_ids[:, 1:])
    assert torch.equal(dream_mask, masked_index[:, 1:])


def test_tiny_dream_forward_cache_shapes(tiny_dream_model):
    input_ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
    output = tiny_dream_model(input_ids, use_cache=True)

    assert output.logits.shape == (2, 3, 64)
    assert len(output.past_key_values) == 2
    for key, value in output.past_key_values:
        assert key.shape == (2, 3, 16)
        assert value.shape == (2, 3, 16)


def test_dream_shifted_ce_backward(tiny_dream_model):
    input_ids = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]])
    masked_index = torch.tensor(
        [[False, False, True, True], [False, True, False, True]]
    )
    perturbed_input_ids = torch.where(masked_index, 63, input_ids)

    logits = tiny_dream_model(perturbed_input_ids).logits
    logits, targets, ce_mask = align_logits_and_targets(
        "dream", logits, input_ids, masked_index
    )
    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        targets.view(-1),
        reduction="none",
    ).view(logits.shape[:-1])
    loss = losses[ce_mask].mean()
    loss.backward()

    assert tiny_dream_model.lm_head.weight.grad is not None
    assert torch.isfinite(tiny_dream_model.lm_head.weight.grad).all()


@pytest.mark.parametrize("dual_cache", [False, True])
def test_cached_generation_shapes_without_pretrained_weights(monkeypatch, tiny_dream_model, dual_cache):
    monkeypatch.setattr(dream_generate, "sample_tokens", _always_select_non_mask)
    input_ids = torch.tensor([[1, 3, 4], [0, 1, 5]])
    attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])

    output = tiny_dream_model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4,
        output_history=False,
        return_dict_in_generate=False,
        steps=4,
        temperature=0.0,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        mask_token_id=63,
        block_length=2,
        dual_cache=dual_cache,
    )

    assert output.shape == (2, 7)
    assert torch.equal(output[:, :3], input_ids)
    assert not torch.any(output[:, 3:] == 63)


@pytest.mark.parametrize("dual_cache", [False, True])
def test_cached_confidence_threshold_is_batch_safe(monkeypatch, tiny_dream_model, dual_cache):
    monkeypatch.setattr(dream_generate, "sample_tokens", _always_select_non_mask)
    input_ids = torch.tensor([[1, 3], [1, 4]])

    output = tiny_dream_model.diffusion_generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=4,
        output_history=False,
        return_dict_in_generate=False,
        steps=4,
        temperature=0.0,
        top_p=0.95,
        alg="confidence_threshold",
        alg_temp=0.0,
        threshold=0.9,
        mask_token_id=63,
        block_length=2,
        dual_cache=dual_cache,
    )

    assert output.shape == (2, 6)
    assert not torch.any(output[:, 2:] == 63)


@pytest.mark.parametrize("dual_cache", [False, True])
def test_cached_greedy_generation_matches_individual_rows(tiny_dream_model, dual_cache):
    input_ids = torch.tensor([[1, 3, 4], [0, 1, 5]])
    attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    generation_kwargs = {
        "max_new_tokens": 4,
        "output_history": False,
        "return_dict_in_generate": False,
        "steps": 4,
        "temperature": 0.0,
        "top_p": 0.95,
        "alg": "entropy",
        "alg_temp": 0.0,
        "mask_token_id": 63,
        "block_length": 2,
        "dual_cache": dual_cache,
    }

    batched_output = tiny_dream_model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    individual_output = torch.cat(
        [
            tiny_dream_model.diffusion_generate(
                input_ids[row : row + 1],
                attention_mask=attention_mask[row : row + 1],
                **generation_kwargs,
            )
            for row in range(input_ids.shape[0])
        ]
    )

    assert torch.equal(batched_output, individual_output)


def test_cached_generation_validates_block_shape(tiny_dream_model):
    with pytest.raises(AssertionError, match="must be divisible"):
        tiny_dream_model.diffusion_generate(
            torch.tensor([[1, 3]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            max_new_tokens=3,
            steps=3,
            mask_token_id=63,
            block_length=2,
        )
