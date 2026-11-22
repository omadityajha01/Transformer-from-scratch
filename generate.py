"""Generate text from the trained checkpoint (results/model.pt).

Usage:
    python generate.py
    python generate.py --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top_k 50
"""

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="\n")
parser.add_argument("--tokens", type=int, default=300)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
tok = BPETokenizer().load("data/tokenizer.json")
checkpoint = torch.load("results/model.pt", map_location="cpu")
model = GPT(GPTConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"])
model.eval()

idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print(tok.decode(out[0].tolist()))
