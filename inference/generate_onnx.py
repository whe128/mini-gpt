# inference/generate_onnx.py

import onnxruntime as ort
from tokenizer.tokenizer import Tokenizer
from model.config import GPTConfig
import numpy as np


print("ONNX Runtime version:", ort.__version__)

# preload DLLs to ensure that the CUDA provider is loaded correctly
ort.preload_dlls()

so = ort.SessionOptions()
so.log_severity_level = 3
"""
    Input name: input_ids  shape: ['batch_size', 'seq_len'] type: tensor(int32)
    Output name: present_k_0~7  shape: ['batch_size', 8, 'total_seq_len', 32] type: tensor(float)
    Output name: present_v_0~7  shape: ['batch_size', 8, 'total_seq_len', 32] type: tensor(float)
"""
prefill_session = ort.InferenceSession(
    "output/minigpt_prefill.onnx",
    sess_options=so,
    providers = ["CUDAExecutionProvider"]
)

"""
    Input name: input_ids  shape: ['batch_size', 'seq_len'] type: tensor(int32)
    Input name: past_k_0~7 shape: ['batch_size', 8, 'past_seq_len', 32] type: tensor(float)
    Input name: past_v_0~7 shape: ['batch_size', 8, 'past_seq_len', 32] type: tensor(float)

    Output name: logits         shape: ['batch_size', 'seq_len', 50257] type: tensor(float)
    Output name: present_k_0~7  shape: ['batch_size', 8, 'total_seq_len', 32] type: tensor(float)
    Output name: present_v_0~7  shape: ['batch_size', 8, 'total_seq_len', 32] type: tensor(float)
"""
decode_session = ort.InferenceSession(
    "output/minigpt_decode.onnx",
    sess_options=so,
    providers = ["CUDAExecutionProvider"]
)

def sample_next_token(logits, temperature = 1.0, top_k = 0, top_p = 1.0):
    if temperature <= 0:
        return np.argmax(logits, axis=-1, keepdims=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])

        # find the K-th largest value
        threshold = np.partition(
            logits,
            -top_k,
            axis=-1
        )[..., -top_k, None]

        logits = np.where(
            logits < threshold,
            -np.inf,
            logits
        )

    # Top-P filtering
    if top_p < 1.0:

        # Sort logits in descending order
        sorted_indices = np.argsort(
            -logits,
            axis=-1
        )

        sorted_logits = np.take_along_axis(
            logits,
            sorted_indices,
            axis=-1
        )

        # Calculate probabilities using stable softmax
        shifted_logits = sorted_logits - np.max(
            sorted_logits,
            axis=-1,
            keepdims=True
        )

        probs = np.exp(shifted_logits)
        probs /= np.sum(probs, axis=-1, keepdims=True)

        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(probs, axis=-1)

        # Create removal mask
        remove_mask = cumulative_probs > top_p

        # Shift mask right to preserve at least one token
        remove_mask[..., 1:] = remove_mask[..., :-1].copy()
        remove_mask[..., 0] = False

        # Remove filtered logits
        sorted_logits = np.where(
            remove_mask,
            -np.inf,
            sorted_logits
        )

        # Restore original token positions
        logits = np.full_like(logits, -np.inf)

        np.put_along_axis(
            logits,
            sorted_indices,
            sorted_logits,
            axis=-1
        )

    # Softmax
    shifted_logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    probs = np.exp(shifted_logits)
    probs /= np.sum(probs, axis=-1, keepdims= True)

    # Sample one token for each batch
    next_tokens = []

    for batch_probs in probs:
        next_token = np.random.choice(
            len(batch_probs),
            p=batch_probs
        )
        next_tokens.append(next_token)

    return np.array(next_tokens, dtype=np.int64).reshape(-1, 1)

def generate_with_onnx(tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
    # prefill the prompt to the model and generate the kv cache
    input_ids = tokenizer.encode(prompt)

    if len(input_ids) > GPTConfig.block_size:
        raise ValueError(f"Input prompt is too long. Max length is {GPTConfig.block_size} tokens.")
    if len(input_ids) == 0:
        raise ValueError("Input prompt is empty. Please provide a valid prompt.")


    # convert input_ids to numpy array and add batch dimension
    input_ids = np.array([input_ids], dtype=np.int32)

    outputs = prefill_session.run(
        # output
        ["logits",
         "present_k_0", "present_v_0",
         "present_k_1", "present_v_1",
         "present_k_2", "present_v_2",
         "present_k_3", "present_v_3",
         "present_k_4", "present_v_4",
         "present_k_5", "present_v_5",
         "present_k_6", "present_v_6",
         "present_k_7", "present_v_7"],

        {"input_ids": input_ids}
    )

    logits = outputs[0]     # shape: [batch, seq_len, vocab_size]
    past_kvs = outputs[1:]  # shape: [batch, num_heads, total_seq_len, head_dim]

    # decode the model to generate new tokens
    next_token = sample_next_token(
        logits[:, -1, :],   # get the last token's logits for sampling
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    generated = []
    for _ in range(max_new_tokens):
        token_id = next_token.item()

        generated.append(token_id)

        if token_id == tokenizer.eos_token_id():
            break

        # decode the model to generate the next token
        outputs = decode_session.run(
            ["logits",
             "present_k_0", "present_v_0",
             "present_k_1", "present_v_1",
             "present_k_2", "present_v_2",
             "present_k_3", "present_v_3",
             "present_k_4", "present_v_4",
             "present_k_5", "present_v_5",
             "present_k_6", "present_v_6",
             "present_k_7", "present_v_7"],
            {
                "input_ids": np.array([[token_id]], dtype=np.int32),
                "past_k_0": past_kvs[0],
                "past_v_0": past_kvs[1],
                "past_k_1": past_kvs[2],
                "past_v_1": past_kvs[3],
                "past_k_2": past_kvs[4],
                "past_v_2": past_kvs[5],
                "past_k_3": past_kvs[6],
                "past_v_3": past_kvs[7],
                "past_k_4": past_kvs[8],
                "past_v_4": past_kvs[9],
                "past_k_5": past_kvs[10],
                "past_v_5": past_kvs[11],
                "past_k_6": past_kvs[12],
                "past_v_6": past_kvs[13],
                "past_k_7": past_kvs[14],
                "past_v_7": past_kvs[15]
            }
        )
        logits = outputs[0]
        past_kvs = outputs[1:]

        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        if len(generated) > GPTConfig.block_size:
            raise ValueError(f"Generated text exceeds the maximum block size of {GPTConfig.block_size} tokens.")

    generated_text = tokenizer.decode(generated)

    return generated_text
# init tokenizer
tokenizer = Tokenizer()

while True:
    try:
        prompt = input("Enter your prompt (or 'q' to quit): ")

        if prompt.lower() == "q":
            break
        prompt = "<|user|>\n" + prompt + "\n<|assistant|>\n"

        output = generate_with_onnx(
            tokenizer,
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )

        print("\nGenerated text:\n", output)

    except Exception as e:
        print("Error:", e)
        continue
