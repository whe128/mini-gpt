# mini-gpt

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
