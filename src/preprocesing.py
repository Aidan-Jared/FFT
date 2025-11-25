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

def get_spectrum(word, tokizer, model):
    encoded_input = torch.tensor(tokizer.encode(word), dtype=torch.long)
    encoded = model.wte(encoded_input)
    spectrum = torch.fft.rfft(encoded).squeeze(0).squeeze(0)
    return (spectrum.real.numpy(), spectrum.imag.numpy())


def plot_spectrum(signal):
    spectrum = torch.fft.rfft(signal).squeeze(0).squeeze(0)
    magnitude = torch.abs(spectrum)
    N = signal.shape[-1]
    freqs = torch.fft.rfftfreq(N, d = 1)

    plt.figure(figsize=(12, 4))
    
    # Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(freqs, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Linear Scale')
    plt.grid(True, alpha=0.3)
    
    # Log scale (better for wide dynamic range)
    plt.subplot(1, 2, 2)
    plt.semilogy(freqs, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log scale)')
    plt.title('Log Scale')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test(x, model, enc):
    encoded_input = torch.tensor(enc.encode(x), dtype=torch.long)
    encoded = model.wte(encoded_input)
    return encoded

def main():
    enc = tiktoken.get_encoding("gpt2")

    model = load_gpt2()
    

    with torch.no_grad():
        df = pd.read_csv("data/google-10000-english-usa.txt", header=None, names=["Words"]).dropna()
        df["Tokens"] = df["Words"].apply(lambda x: enc.encode(x))
        df = df.explode("Tokens").reset_index(drop=True)
        df["Tokens"] = df["Tokens"].apply(lambda x: enc.decode([x]))
        df['rfft'] = df["Words"].apply(lambda x: get_spectrum(x, enc, model))
        freq = pd.DataFrame(df['rfft'].to_list(), columns=["real_rfft", "imag_rfft"])
        freq["amplitude"] = freq['real_rfft']**2 + freq['imag_rfft']**2
        freq["amplitude"] = freq["amplitude"].apply(lambda x: np.sqrt(x))
        phase = []
        for i in range(df.shape[0]):
            phase.append(np.arctan2(freq['real_rfft'].iloc[i], freq['imag_rfft'].iloc[i]))
        freq['phase'] = pd.Series(phase)

        df.join(freq)

        df.to_csv("data/gpt2_rfft.csv")


if __name__ == "__main__":
    main()
