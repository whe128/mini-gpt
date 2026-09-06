# Mini-GPT

                 train.txt
                    │
                    ↓
             GPT-2 BPE Tokenizer
                    │
                    ↓
                Token IDs
                    │
                    ↓
          ┌────────────────────┐
          │    MiniGPT         │
          │                    │
          │ Embedding          │
          │ RMSNorm            │
          │ RoPE               │
          │ Attention          │
          │ SwiGLU             │
          │ Residual           │
          │ × 4 Blocks         │
          └─────────┬──────────┘
                    │
                    ↓
                  logits
                    │
                    ↓
             Cross Entropy
                    │
                    ↓
                 AdamW
                    │
                    ↓
              Checkpoint
                    │
                    ↓
             ┌──────────────┐
             │     SFT      │
             │              │
             │ User = -100  │
             │ Assistant =  │
             │ CrossEntropy │
             └──────┬───────┘
                    │
                    ↓
              Fine-tuned LLM
                    │
                    ↓
              ┌─────────────┐
              │ Inference   │
              │             │
              │ Prefill     │
              │ KV Cache    │
              │ Decode      │
              │ Top-K       │
              │ Top-P       │
              └──────┬──────┘
                     │
                     ↓
               FastAPI / SSE
                     │
                     ↓
                Chat UI

A small GPT-style language model implemented from scratch with PyTorch.

This project implements the main components of a decoder-only Transformer and trains the model through next-token prediction and supervised fine-tuning (SFT).

## Features

- Decoder-only Transformer architecture
- Multi-Head Self-Attention
- Causal Attention Mask
- RoPE (Rotary Positional Embedding)
- RMSNorm
- Token Embedding
- KV Cache for efficient generation
- Temperature sampling
- Top-K sampling
- Top-P (Nucleus) sampling
- Pre-training with next-token prediction
- Supervised Fine-Tuning (SFT)
- Response-only loss masking
- Checkpoint saving and loading
- GPU / CPU inference

## Model Configuration

The current model uses:

| Parameter           |  Value |
| ------------------- | -----: |
| Vocabulary Size     | 50,257 |
| Context Length      |    256 |
| Transformer Layers  |      4 |
| Attention Heads     |      4 |
| Embedding Dimension |    128 |
| Dropout             |    0.0 |
| Parameters          | ~14.3M |

The model is intentionally small so that it can be trained and deployed with relatively limited computational resources.

## Architecture

```text
Input Text
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
Transformer Block × 4
    │
    ├── RMSNorm
    ├── Multi-Head Self-Attention
    │      ├── RoPE
    │      └── Causal Mask
    ├── Residual Connection
    ├── RMSNorm
    └── Feed Forward Network
    │
    ▼
Language Model Head
    │
    ▼
Logits
    │
    ▼
Next Token Prediction
```

## Training

### 1. Pre-training

The model is trained using next-token prediction.

For example:

```text
Input:
The cat sat on the

Target:
cat sat on the mat
```

More precisely, the target sequence is shifted by one token:

```text
x = [The, cat, sat, on, the]
y = [cat, sat, on, the, mat]
```

The training objective is cross-entropy loss over the next token.

### 2. Supervised Fine-Tuning

After pre-training, the model can be fine-tuned using instruction-response data.

Example:

```text
<|user|>
Who are you?
<|assistant|>
I am a little AI assistant.
```

During SFT, the loss is calculated only on the assistant response.

```text
<|user|> Who are you? <|assistant|> I am an AI assistant. <EOS>
   -100    -100       -100          loss       loss       loss
```

`-100` positions are ignored by `CrossEntropyLoss`.

## Dataset Format

### Pre-training

The pre-training corpus is stored as plain text:

```text
A little rabbit lived in a forest...

The rabbit met a small bird...

<|endoftext|>

Another story begins here...
```

### SFT

SFT data uses JSONL format:

```json
{"instruction":"Who are you?","response":"I am a little AI assistant."}
{"instruction":"Tell me a short story about a rabbit.","response":"Once there was a little rabbit who lived in a green forest..."}
```

## Project Structure

```text
mini-gpt/
│
├── configs/
│   └── tiny.yaml
│
├── data/
│   ├── train.txt
│   └── sft.jsonl
│
├── model/
│   ├── config.py
│   ├── gpt.py
│   ├── attention.py
│   └── ...
│
├── tokenizer/
│   └── tokenizer.py
│
├── training/
│   ├── train.py
│   └── sft_train.py
│
├── inference/
│   └── generate.py
│
├── output/
│   └── minigpt.pt
│
└── README.md
```

## Installation

```bash
git clone <your-repository-url>
cd mini-gpt

pip install -r requirements.txt
```

## Pre-training

Run:

```bash
python -m training.train
```

The trained checkpoint will be saved to:

```text
output/minigpt.pt
```

## Supervised Fine-Tuning

After pre-training, run:

```bash
python -m training.sft_train
```

The SFT checkpoint will be saved to:

```text
output/minigpt_sft.pt
```

## Inference

Load the trained checkpoint and provide a prompt:

```text
User:
Tell me a story about a little rabbit.

Assistant:
Once there was a little rabbit who lived in a quiet forest...
```

The model supports autoregressive generation, where each generated token is fed back into the model to predict the next token.

## Generation

The generation process is:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
MiniGPT
  │
  ▼
Logits
  │
  ▼
Sampling
  │
  ▼
Next Token
  │
  └──────────────┐
                 │
                 ▼
              MiniGPT
```

KV caching is used during generation to avoid recomputing previous key/value states.

## Training Objective

The model is a causal language model.

For a token sequence:

```text
x₁, x₂, x₃, ..., xₙ
```

the model learns:

```text
P(x₂ | x₁)
P(x₃ | x₁, x₂)
P(x₄ | x₁, x₂, x₃)
...
P(xₙ | x₁, ..., xₙ₋₁)
```

The training loss is cross-entropy:

```text
Loss = CrossEntropy(Logits, Target Tokens)
```

## Goal

The goal of this project is to understand how a small language model works internally by implementing the major components rather than treating the model as a black box.

The project covers the complete pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Transformer
 ↓
Training
 ↓
Checkpoint
 ↓
Inference
 ↓
Generated Text
```

## License

MIT License
