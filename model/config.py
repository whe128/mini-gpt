from dataclasses import dataclass

@dataclass
class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 256
    dropout: int = 0.0
