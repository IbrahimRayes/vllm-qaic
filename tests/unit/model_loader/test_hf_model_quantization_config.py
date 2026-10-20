# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU-only tests: get_hf_model hands QEfficient the checkpoint's quantization_config.

vLLM re-tags ``hf_config.quantization_config["quant_method"]`` in place to select its
own kernels (gpt-oss ``mxfp4`` -> ``gpt_oss_mxfp4``); QEfficient does not know those
names and would skip the quantized weights. A real vLLM ModelConfig is used so the
tests follow vLLM's actual hooks.
"""

import json
from types import SimpleNamespace

import pytest
from transformers import GptOssConfig, LlamaConfig
from vllm.config import ModelConfig

import QEfficient
import vllm_qaic.quantization.quant_config  # noqa: F401  (registers mxfp6)
from vllm_qaic.model_loader import qaic

GPT_OSS_QUANT = {"quant_method": "mxfp4", "modules_to_not_convert": ["lm_head"]}


@pytest.fixture
def captured(monkeypatch):
    """Replace QEfficient's from_pretrained; record the config it receives."""
    calls = []

    def fake_from_pretrained(cls, model, **kwargs):
        calls.append(kwargs["config"])
        return SimpleNamespace(model=model)

    monkeypatch.setattr(
        QEfficient.QEFFAutoModelForCausalLM,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    return calls


def _model_config(tmp_path, config_cls, quantization_config):
    config = config_cls(num_hidden_layers=2).to_dict()
    config["architectures"] = [config_cls.__name__.replace("Config", "ForCausalLM")]
    config.pop("quantization_config", None)
    if quantization_config is not None:
        config["quantization_config"] = quantization_config  # as written on disk
    (tmp_path / "config.json").write_text(json.dumps(config))
    return ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=2048,
        quantization="mxfp6" if quantization_config is not None else None,
    )


def test_gpt_oss_retagged_quant_method_is_restored(tmp_path, captured):
    model_config = _model_config(tmp_path, GptOssConfig, GPT_OSS_QUANT)
    assert model_config.hf_config.quantization_config["quant_method"] == "gpt_oss_mxfp4"

    qaic.get_hf_model(model_config)

    assert captured[0].quantization_config == GPT_OSS_QUANT
    # vLLM's own config keeps its kernel tag.
    assert model_config.hf_config.quantization_config["quant_method"] == "gpt_oss_mxfp4"


def test_any_retagged_quant_method_is_restored(tmp_path, captured):
    model_config = _model_config(
        tmp_path, LlamaConfig, {"quant_method": "fp8", "activation_scheme": "dynamic"}
    )
    # Same pattern as vLLM's DeepSeek-V4 hook ("fp8" -> "deepseek_v4_fp8").
    model_config.hf_config.quantization_config["quant_method"] = "deepseek_v4_fp8"

    qaic.get_hf_model(model_config)

    assert captured[0].quantization_config["quant_method"] == "fp8"


@pytest.mark.parametrize(
    ("on_disk", "expected"),
    [
        ("awq", "awq"),
        ("AWQ", "awq"),  # vLLM's spelling normalization is kept
        ("compressed_tensors", "compressed-tensors"),
    ],
)
def test_not_retagged_quant_config_is_unchanged(tmp_path, captured, on_disk, expected):
    model_config = _model_config(
        tmp_path, LlamaConfig, {"quant_method": on_disk, "bits": 4, "group_size": 128}
    )

    qaic.get_hf_model(model_config)

    assert captured[0].quantization_config["quant_method"] == expected


def test_unquantized_model_gets_no_quant_config(tmp_path, captured):
    qaic.get_hf_model(_model_config(tmp_path, LlamaConfig, None))

    assert getattr(captured[0], "quantization_config", None) is None


def test_passed_quant_config_does_not_alias_vllm_state(tmp_path, captured):
    model_config = _model_config(tmp_path, GptOssConfig, GPT_OSS_QUANT)

    qaic.get_hf_model(model_config)

    arch_quant = model_config.model_arch_config.quantization_config
    passed = captured[0].quantization_config
    assert passed is not arch_quant
    assert passed["modules_to_not_convert"] is not arch_quant["modules_to_not_convert"]
