# training/train.py

import os
import random
import torch
import yaml
import json
import torch.utils.data as data
import torch.nn as nn

from tokenizer.tokenizer import Tokenizer
from model.gpt import MiniGPT
from data.dataset import SFTDataset
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
            loss = nn.CrossEntropyLoss(ignore_index = -100)(
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Tokenizer
    tokenizer = Tokenizer()
    print("Tokenizer vocab:", tokenizer.vocab_size())

    # SFT dataset
    inputs = []
    labels = []
    with open("data/sft.jsonl", "r", encoding = "utf-8") as f:
        for line in f:
            item = json.loads(line)

            instruction = item["instruction"]
            response = item["response"]

            # ==================================================
            # 1. Original prompt
            # ==================================================
            # inferences should use the same format
            prompt = "<|user|>\n" + instruction + "\n<|assistant|>\n"

            prompt_ids = tokenizer.encode(prompt, add_eos = False)
            response_ids = tokenizer.encode(response, add_eos = True)

            # input x
            x = prompt_ids + response_ids[:-1]    # the last token just needs to be apprear in the target
            # target y
            y = [-100] * (len(prompt_ids) - 1) + response_ids

            # encode the text into token IDs
            inputs.append(x)
            labels.append(y)

            # ==================================================
            # 2. Lowercase prompt
            # ==================================================
            prompt_lower = (
                "<|user|>\n"
                + instruction.lower()
                + "\n<|assistant|>\n"
            )

            prompt_lower_ids = tokenizer.encode( prompt_lower, add_eos=False)

            x_lower = prompt_lower_ids + response_ids[:-1]

            y_lower = ( [-100] * (len(prompt_lower_ids) - 1) + response_ids)

            inputs.append(x_lower)
            labels.append(y_lower)

    # sft dataset
    sft_dataset = SFTDataset(
        inputs = inputs,
        outputs = labels,
        block_size = cfg["block_size"],
        eos_token_id = tokenizer.eos_token_id(),
        pad_token_id = -100
    )

    train_dataloader = data.DataLoader(
        sft_dataset,
        batch_size = cfg["batch_size"],
        shuffle = True,
        drop_last = True)       # if last is not the whole batch size, drop it


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

    # Load pretrained checkpoint
    checkpoint = torch.load(
        "output/minigpt_sft.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model"])

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
    # Training
    # -----------------------
    eval_interval = cfg["eval_interval"]
    num_epochs = 20

    update_step = 0
    try:
        for epoch in range(num_epochs):
            print(f"\n========== Epoch {epoch + 1}/{num_epochs} ==========")

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
                    loss = nn.CrossEntropyLoss(ignore_index=-100)(
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
                # if update_step % eval_interval == 0:
                    # val_loss = evaluate(model, val_dataloader, device)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        # -----------------------
        # Save checkpoint
        # -----------------------
    # save the model
        os.makedirs("output", exist_ok = True)
        checkpoint_path = "output/minigpt_sft.pt"
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
