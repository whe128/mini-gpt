from torch import nn


class MiniGPTONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # * means: accept a variable number of positional arguments
    # k0 v0 k1 v1 k2 v2 ... kn vn
    # original kv cache is a list of tuple [(k0, v0), (k1, v1), (k2, v2), ..., (kn, vn)]
    # we want a flat list of tensors [k0, v0, k1, v1, k2, v2, ..., kn, vn]
    def forward(self, input_ids, *past_kvs):

        if len(past_kvs) == 0:
            past_kvs = None
        else:
            # convert flat inputs into per-layer KV pairs
            past_kvs = [
                (past_kvs[i * 2], past_kvs[i * 2 + 1])
                for i in range(self.model.config.n_layer)
            ]

        logits, present_kvs = self.model(
            input_ids, use_cache=True, past_kvs=past_kvs
        )

        # flatten KV outputs
        outputs = [logits]

        for key, value in present_kvs:
            outputs.append(key)
            outputs.append(value)

        # (logits, k0, v0, k1, v1, k2, v2, ..., kn, vn)
        print(len(outputs), "outputs:", [o.shape for o in outputs])
        return tuple(outputs)
