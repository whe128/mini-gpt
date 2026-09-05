# training/train.py

import os
import random
import torch
import yaml
import torch.utils.data as data
import torch.nn as nn

from tokenizer.tokenizer import Tokenizer
from model.gpt import MiniGPT
from data.dataset import ModelDataset
from model.config import GPTConfig

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, device, max_batches = 20):
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output, _ = model(x)
            loss = nn.CrossEntropyLoss()(
                # from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
                output.view(-1, output.size(-1)),
                # from [batch, seq_len] to [batch * seq_len]
                y.view(-1)
            )

            total_loss += loss.item()
            count += 1

            if count >= max_batches:
                break

    # return back to train mode
    model.train()

    return total_loss / max(count, 1)


def main():
    cfg = load_config("configs/tiny.yaml")
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print("Device:", device)

    # Tokenizer
    tokenizer = Tokenizer()
    print("Tokenizer vocab:", tokenizer.vocab_size())

    # dataset
    with open("data/train.txt", "r", encoding = "utf-8") as f:
        text = f.read()

    # txt -> token ids
    token_ids = tokenizer.encode(text, add_eos=True)

    # seperate train and validation set
    split_idx = int(len(token_ids) * cfg["train_val_split"])    # 0.9
    train_dataset = ModelDataset(token_ids[:split_idx], cfg["block_size"])
    val_dataset = ModelDataset(token_ids[split_idx:], cfg["block_size"])

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size = cfg["batch_size"],
        shuffle = True,
        drop_last = True)       # if last is not the whole batch size, drop it
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size = cfg["batch_size"],
        shuffle = False,
        drop_last=True)


    # -----------------------
    # Model
    # -----------------------

    model_config = GPTConfig(
        vocab_size = tokenizer.vocab_size(),
        block_size = cfg["block_size"],
        n_layer = cfg["n_layer"],
        n_head = cfg["n_head"],
        n_embed = cfg["n_embed"],
        dropout = cfg["dropout"]
    )

    model = MiniGPT(model_config).to(device)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameters / 1e6:.2f}M")

    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"]
    )



    # -----------------------
    # AMP
    # -----------------------
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp )


    # -----------------------
    # Load checkpoint
    # -----------------------
    checkpoint_path = "output/minigpt.pt"

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        update_step = checkpoint["step"]

        print(f"Resume training from step {update_step}")
    else:
        update_step = 0
        print("No checkpoint found. Starting training from scratch.")


    # -----------------------
    # Training
    # -----------------------
    eval_interval = cfg["eval_interval"]

    try:
        for x, y in train_dataloader:
            # stop training if we reach the max update step

            x = x.to(device)
            y = y.to(device)

            with torch.autocast(
                device_type = device.type,
                dtype = torch.float16,
                enabled = torch.cuda.is_available()

            ):
                output, _ = model(x)
                loss = nn.CrossEntropyLoss()(
                    # from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
                    output.view(-1, output.size(-1)),
                    # from [batch, seq_len] to [batch * seq_len]
                    y.view(-1)
                )
            # backward pass
            optimizer.zero_grad(set_to_none = True)
            scaler.scale(loss).backward()

            # optimizer step
            scaler.step(optimizer)
            scaler.update()

            # update step
            update_step += 1

            print( f"Step {update_step:5d}" f"| Loss: {loss.item():.4f}")


            # evaluate the model
            if update_step % eval_interval == 0:
                val_loss = evaluate(model, val_dataloader, device)
                print(f"Step {update_step:5d} " f"| Validation Loss: {val_loss:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        # -----------------------
        # Save checkpoint
        # -----------------------
    # save the model
        os.makedirs("output", exist_ok = True)
        checkpoint_path = "output/minigpt.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": update_step,
                "config": model_config.__dict__,
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved at step {update_step}: "
            f"{checkpoint_path}"
        )

        print( f"Training finished. " f"Checkpoint saved to {checkpoint_path}" )

if __name__ == "__main__": main()
