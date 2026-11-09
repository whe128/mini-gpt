# inference/export_onnx.py
import torch
from model.gpt import MiniGPT
from model.gpt_onnx import MiniGPTONNX
from model.config import GPTConfig
import onnx


"""
process of exporting the model to ONNX
1: create dummy input of the model
2: create input and output names for the ONNX model (customized by the user, with the same order as the model's forward function)
3: create dynamic axes for the ONNX model
4: export the model to ONNX using torch.onnx.export
"""


checkpoint = torch.load(
    "output/minigpt_sft.pt",
    map_location="cpu",
    weights_only=True
)

config = GPTConfig(
    vocab_size=checkpoint["config"]["vocab_size"],
    block_size=checkpoint["config"]["block_size"],
    n_layer=checkpoint["config"]["n_layer"],
    n_head=checkpoint["config"]["n_head"],
    n_embed=checkpoint["config"]["n_embed"],
    dropout=checkpoint["config"]["dropout"]
)

model = MiniGPT(config)
model.load_state_dict(checkpoint["model"])
model.eval()

n_layer = config.n_layer

# use the wrapper class to export the model to ONNX
onnx_model = MiniGPTONNX(model)
onnx_model.eval()

batch_size = 1
seq_len = 16

dummy_input = torch.randint(
    low = 0,
    high = config.vocab_size,
    size = (batch_size, seq_len),
    dtype = torch.int32
)

onnx_output_name = ["logits"]
for i in range(n_layer):
    onnx_output_name.append(f"present_k_{i}")
    onnx_output_name.append(f"present_v_{i}")

onnx_dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "logits": {0: "batch_size", 1: "seq_len"}
}

for i in range(n_layer):
    # [B, H, S, D]
    onnx_dynamic_axes[f"present_k_{i}"] = {0: "batch_size", 2: "seq_len"}
    onnx_dynamic_axes[f"present_v_{i}"] = {0: "batch_size", 2: "seq_len"}


# output
torch.onnx.export(
    model = onnx_model,
    args = (dummy_input,),
    f = "output/minigpt_prefill.onnx",
    input_names = ["input_ids"],
    output_names = onnx_output_name,
    dynamic_axes = onnx_dynamic_axes,
    opset_version = 17,
    dynamo = False
)

onnx.checker.check_model("output/minigpt_prefill.onnx")

print("Prefill ONNX exported model to onnx...")

# ===========================================================================

# decode the model to ONNX
batch_size = 1
seq_len = 1
past_seq_len = 16
head_dim = config.n_embed // config.n_head


dummy_input = torch.randint(
    low = 0,
    high = config.vocab_size,
    size = (batch_size, seq_len),
    dtype = torch.int32
)

dummy_past_kvs = []
for i in range(n_layer):
    dummy_k = torch.randn(
        size = (batch_size, config.n_head, past_seq_len, head_dim),
        dtype = torch.float32
    )
    dummy_v = torch.randn(
        size = (batch_size, config.n_head, past_seq_len, head_dim),
        dtype = torch.float32
    )

    dummy_past_kvs.append(dummy_k)
    dummy_past_kvs.append(dummy_v)

input_names = ["input_ids"]
for i in range(n_layer):
    input_names.append(f"past_k_{i}")
    input_names.append(f"past_v_{i}")

output_names = ["logits"]
for i in range(n_layer):
    output_names.append(f"present_k_{i}")
    output_names.append(f"present_v_{i}")


onnx_dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "logits": {0: "batch_size", 1: "seq_len"}
}

for i in range(n_layer):
    onnx_dynamic_axes[f"past_k_{i}"] = {0: "batch_size", 2: "past_seq_len"}
    onnx_dynamic_axes[f"past_v_{i}"] = {0: "batch_size", 2: "past_seq_len"}

    onnx_dynamic_axes[f"present_k_{i}"] = {0: "batch_size", 2: "total_seq_len"}
    onnx_dynamic_axes[f"present_v_{i}"] = {0: "batch_size", 2: "total_seq_len"}


# export decode ONNX
torch.onnx.export(
    model = onnx_model,
    args = (dummy_input, *dummy_past_kvs),
    f = "output/minigpt_decode.onnx",
    input_names = input_names,
    output_names = output_names,
    dynamic_axes = onnx_dynamic_axes,
    opset_version = 17,
    dynamo = False
)

onnx.checker.check_model("output/minigpt_decode.onnx")
print("Decode ONNX exported model to onnx...")
