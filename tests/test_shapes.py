"""Sanity checks: BPE round-trip, LayerNorm correctness, output shapes and the causal mask."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, CausalSelfAttention, GPTConfig, LayerNorm  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

CFG = GPTConfig(vocab_size=300, block_size=16, d_model=32, n_head=4, n_layer=2, dropout=0.0)


def test_bpe_round_trip(tmp_path):
    tok = BPETokenizer()
    tok.train("the cat sat on the mat. the cat ate the rat.\n" * 20, vocab_size=300)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok = BPETokenizer().load(path)
    for s in ["the cat", "", "Hello, World!\nNew line\ttab", "émoji 🙂 ünïcödé", "zzz qqq 123"]:
        assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode("the cat sat on the mat")) < len("the cat sat on the mat".encode("utf-8"))


def test_layernorm_matches_pytorch():
    ln = LayerNorm(32)
    with torch.no_grad():
        ln.gamma.normal_()
        ln.beta.normal_()
    x = torch.randn(4, 10, 32)
    expected = F.layer_norm(x, (32,), ln.gamma, ln.beta, ln.eps)
    assert torch.allclose(ln(x), expected, atol=1e-5)


def test_attention_shape():
    x = torch.randn(2, 16, 32)
    assert CausalSelfAttention(CFG)(x).shape == (2, 16, 32)


def test_gpt_shapes_and_loss():
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, CFG.vocab_size)
    assert loss.ndim == 0


def test_causal_mask_no_future_leak():
    """Changing token t+1 must not change the output at positions <= t."""
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 16))
    changed = idx.clone()
    t = 7
    changed[0, t + 1] = (idx[0, t + 1] + 1) % CFG.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :t + 1], b[:, :t + 1], atol=1e-6)
    assert not torch.allclose(a[:, t + 1], b[:, t + 1])


def test_ablation_configs_run():
    idx = torch.randint(0, CFG.vocab_size, (2, 16))
    for flag in ["use_pos_emb", "use_layernorm", "use_residual"]:
        cfg = GPTConfig(**{**CFG.__dict__, flag: False})
        logits, _ = GPT(cfg)(idx)
        assert logits.shape == (2, 16, CFG.vocab_size)


def test_generate_length():
    model = GPT(CFG).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20, top_k=10)
    assert out.shape == (1, 21)
