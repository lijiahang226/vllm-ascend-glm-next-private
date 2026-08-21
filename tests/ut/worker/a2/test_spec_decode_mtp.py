# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UTs for the Ascend v2 MTP speculator.

Covers:
1. ``init_speculator`` dispatches MTP to ``AscendMTPSpeculator``.
2. The mixin forwards ``spec_step_idx`` to the draft model only for MTP
   (Eagle draft models have no such kwarg).
3. Draft sampling forwards the step to ``compute_logits`` for MTP.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.worker.v2.spec_decode.base import _current_draft_step
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


def _make_mtp_speculator(method: str = "mtp") -> AscendMTPSpeculator:
    """Construct an uninitialized mixin instance with mocked attributes."""
    spec = AscendMTPSpeculator.__new__(AscendMTPSpeculator)
    spec.method = method
    spec.use_local_argmax_reduction = False
    spec.current_draft_step = torch.tensor(2, dtype=torch.int64)
    spec.vllm_config = SimpleNamespace()
    spec.supports_mm_inputs = False
    spec.input_buffers = SimpleNamespace(input_ids=torch.zeros(8, dtype=torch.int64))
    spec.hidden_states = torch.zeros(8, 4)
    spec.model = MagicMock()
    spec.prefill_cudagraph_manager = None
    return spec


def test_init_speculator_dispatches_mtp():
    from vllm_ascend.worker.v2.spec_decode.eagle import init_speculator

    speculative_config = SimpleNamespace(method="mtp", use_eagle=lambda: False)
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.mtp.speculator.AscendMTPSpeculator",
        return_value="mtp-speculator",
    ) as mock_cls:
        assert init_speculator(vllm_config, torch.device("cpu")) == "mtp-speculator"
        mock_cls.assert_called_once_with(vllm_config, torch.device("cpu"))


def test_init_speculator_still_dispatches_eagle():
    from vllm_ascend.worker.v2.spec_decode.eagle import init_speculator

    speculative_config = SimpleNamespace(method="eagle", use_eagle=lambda: True)
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator",
        return_value="eagle-speculator",
    ) as mock_cls:
        assert init_speculator(vllm_config, torch.device("cpu")) == "eagle-speculator"
        mock_cls.assert_called_once_with(vllm_config, torch.device("cpu"))


def test_current_draft_step():
    spec = SimpleNamespace(current_draft_step=torch.tensor(3, dtype=torch.int64))
    assert _current_draft_step(spec) == 3


def test_greedy_sample_draft_mtp_forwards_step():
    spec = _make_mtp_speculator(method="mtp")
    spec.model.compute_logits.return_value = torch.tensor([[1.0, 3.0, 2.0]])
    out = spec._greedy_sample_draft(torch.zeros(1, 4))
    spec.model.compute_logits.assert_called_once()
    _, kwargs = spec.model.compute_logits.call_args
    assert kwargs.get("spec_step_idx") == 2
    assert out.item() == 1


def test_greedy_sample_draft_eagle_skips_step():
    spec = _make_mtp_speculator(method="eagle")
    spec.model.compute_logits.return_value = torch.tensor([[1.0, 3.0, 2.0]])
    spec._greedy_sample_draft(torch.zeros(1, 4))
    spec.model.compute_logits.assert_called_once()
    _, kwargs = spec.model.compute_logits.call_args
    assert "spec_step_idx" not in kwargs


def test_run_model_mtp_forwards_step_to_model():
    spec = _make_mtp_speculator(method="mtp")
    spec.model.return_value = (torch.zeros(8, 4), torch.zeros(8, 4))
    with patch(
        "vllm_ascend.worker.v2.spec_decode.base.set_forward_context",
        return_value=nullcontext(),
    ):
        last_hidden, hidden = spec._run_model(8, None, None, None)

    assert last_hidden.shape == (8, 4)
    assert hidden.shape == (8, 4)
    _, kwargs = spec.model.call_args
    assert kwargs.get("spec_step_idx") == 2


def test_run_model_eagle_skips_step():
    spec = _make_mtp_speculator(method="eagle")
    spec.model.return_value = (torch.zeros(8, 4), torch.zeros(8, 4))
    with patch(
        "vllm_ascend.worker.v2.spec_decode.base.set_forward_context",
        return_value=nullcontext(),
    ):
        spec._run_model(8, None, None, None)

    _, kwargs = spec.model.call_args
    assert "spec_step_idx" not in kwargs


def test_sample_draft_probabilistic_mtp_forwards_step():
    spec = _make_mtp_speculator(method="mtp")
    spec.use_fp64_gumbel = False
    spec.model.compute_logits.return_value = torch.zeros(1, 4)
    positions = torch.zeros(1, dtype=torch.int64)
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    temperature = torch.ones(1)
    seeds = torch.zeros(1, dtype=torch.int64)
    draft_step = torch.tensor(2, dtype=torch.int64)
    draft_logits = torch.zeros(1, 4)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.base.gumbel_sample",
        return_value=torch.zeros(1, dtype=torch.int64),
    ):
        spec.sample_draft(
            torch.zeros(1, 4),
            positions,
            idx_mapping,
            temperature,
            seeds,
            draft_step,
            draft_logits,
        )
    _, kwargs = spec.model.compute_logits.call_args
    assert kwargs.get("spec_step_idx") == 2
