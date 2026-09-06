# inference/generate.py

import torch

from inference.sampling import sample_next_token


def generate(model, tokenizer, promt,
             max_new_tokens = 100, temperature = 0.8, top_k = 50, top_p = 0.95, device = "cuda"):
    """
        generate text from the model
    """

    model.eval()

    with torch.no_grad():
        # tokenize the prompt
        # this is only python list
        input_ids = tokenizer.encode(promt)
        generated = input_ids.copy()

        if len(input_ids) > model.config.block_size:
            raise ValueError(f"Input prompt is too long. Max length is {model.config.block_size} tokens.")
        if len(input_ids) == 0:
            raise ValueError("Input prompt is empty. Please provide a valid prompt.")

        # convert to tensor and add batch dimension
        # need [] to add a dimension, means the batch
        # shape: [1, seq_len]
        input_ids = torch.tensor(
            [input_ids], dtype=torch.long, device=device
        )

        # prefill the prompt to the model and generate the kv cache
        # output shape: [batch, seq_len, vocab_size]
        # only the last one is the predicted token
        # the pre are duplicated with the input
        output, past_kvs = model(input_ids, use_cache=True)

        # shape of next_token: [batch, 1]
        next_token = sample_next_token(output[:, -1, :], temperature, top_k, top_p)

        # generate new tokens
        for _ in range(max_new_tokens):
            token_id = next_token.item()
            # append the new token to the generated list
            generated.append(token_id)

            if token_id == tokenizer.eos_token_id():
                break

            # continue to generate the next token
            # next_token shape: [batch, 1]
            output, past_kvs = model(next_token, use_cache=True, past_kvs=past_kvs)

            # sample the next token
            next_token = sample_next_token(output[:, -1, :], temperature, top_k, top_p)

            if len(generated) > model.config.block_size:
                raise ValueError(f"Generated text exceeds the maximum block size of {model.config.block_size} tokens.")

    # decode the generated token ids to text
    generated_text = tokenizer.decode(generated)

    return generated_text


if __name__ == "__main__":
    from model.gpt import MiniGPT
    from model.config import GPTConfig
    from tokenizer.tokenizer import Tokenizer
    from inference.generate import generate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # checkpoint_path = "output/minigpt.pt"
    checkpoint_path = "output/minigpt_sft.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # **checkpoint["config"] is a dictionary, we need to unpack it to pass to GPTConfig
    config = GPTConfig(
        vocab_size=checkpoint["config"]["vocab_size"],
        block_size=checkpoint["config"]["block_size"],
        n_layer=checkpoint["config"]["n_layer"],
        n_head=checkpoint["config"]["n_head"],
        n_embed=checkpoint["config"]["n_embed"],
        dropout=checkpoint["config"]["dropout"]
    )

    tokenizer = Tokenizer()

    model = MiniGPT(config).to(device)

    # load the model state dict
    model.load_state_dict(checkpoint["model"])

    model.eval()

    while True:
        try:
            prompt = input("\nPrompt (q to quit): ")

            if prompt.lower() == "q":
                break
            prompt = "<|user|>\n" + prompt + "\n<|assistant|>\n"

            output = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                device=device
            )

            output = output[len(prompt):]  # remove the prompt from the output
            print("Generated:", output)
        except Exception as e:
            print(f"\nError: {e}")
            continue
