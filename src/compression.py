import torch
import numpy as np
import tiktoken

from gpt import GPT2

import json
import matplotlib.pyplot as plt

import pandas as pd

torch.manual_seed(42)


def load_gpt2(weights_path='pretrained/pytorch_model.bin', config_path='pretrained/config.json'):
    """Load pretrained GPT-2"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = GPT2(config)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Map weight names (HuggingFace -> our implementation)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'transformer.' prefix
        new_key = key.replace('transformer.', '')
        if 'c_attn.weight' in new_key or 'c_proj.weight' in new_key or 'c_fc.weight' in new_key:
            value = value.T
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model

def fft_compression(word, tokizer, model, removed):
    encoded_input = torch.tensor(tokizer.encode(word), dtype=torch.long)
    encoded = model.wte(encoded_input)
    spectrum = torch.fft.rfft(encoded).squeeze(0).squeeze(0)
    compress = torch.abs(spectrum).topk(removed, largest=False)
    spectrum[~compress[1]] = 0.
    compressed_s = torch.fft.irfft(spectrum)
    return torch.cosine_similarity(encoded, compressed_s).item()
    

def main():
    enc = tiktoken.get_encoding("gpt2")

    model = load_gpt2()
    

    with torch.no_grad():
        df = pd.read_csv("data/google-10000-english-usa.txt", header=None, names=["Words"]).dropna()
        df["Tokens"] = df["Words"].apply(lambda x: enc.encode(x))
        df = df.explode("Tokens").reset_index(drop=True)
        df["Tokens"] = df["Tokens"].apply(lambda x: enc.decode([x]))
        for i in range(100,101):
            df[f"removed_{i}"] = df["Tokens"].apply(lambda x: fft_compression(x, enc, model,  i))
            if df[f"removed_{i}"].mean() < .8:
                break
        print("hi")


if __name__ == "__main__":
    main()