import torch

def top_k_logits(logits, top_k = 0, top_p = 1.0):
    """
                    model
                     ↓
                    logits
                     ↓
                    temperature
                     ↓
                    top-k
                     ↓
                    top-p
                     ↓
                    softmax → probabilities
                     ↓
                    torch.multinomial
                     ↓
                    next token
        only keep top k logits, set the rest to -infinity
        logits: [batch, vocab_size]
        top_k: int, number of top logits to keep
        top_p: float, cumulative probability threshold for nucleus sampling
    """

    if top_k > 0:

        top_k = min( top_k, logits.size(-1))    # Safety check

        # already sorted after topk  (descending order)
        # [batch, vacab_size] -> [batch, top_k]
        values, _ = torch.topk(logits, top_k)   # get the top k logits

        # -1, get the minmum value of each row, then expand to the shape of logits
        # [batch, top_k] -> [batch, 1]
        threshold = values[..., -1, None]       # get the threshold value

        # let the logits less than threshold to be -inf, so that they will not be sampled
        logits = torch.where(
            logits < threshold,    # condition
            torch.full_like(logits, float("-inf")),  # true case   [-inf, -inf, -inf, ...]
            logits                                   # false case  [log1, log2, log3, ...]
        )

    if top_p < 1.0:
        # sort the logits in descending order
        # prepare for the cumulative probability calculation
        # sorted_logits -> sorted in descending order, [batch, vocab_size]
        # sorted_indices -> the indices of the sorted logits, [batch, vocab_size]
        sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1)

        # calculate the cumulative probabilities of the sorted logits
        # shape: [batch, vocab_size]
        probs = torch.softmax(sorted_logits, dim = -1)

        # prefix sum
        # shape: [batch, vocab_size]
        cumulative_probs = torch.cumsum(probs, dim = -1)

        # create a mask for the tokens to be removed
        remove_mask = cumulative_probs > top_p

        # use clone to avoid modifying the original tensor
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()  # shift the mask to the right

        # keep the first token, because we want to keep at least one token
        remove_mask[..., 0] = False  # keep the first token

        # set the logits of the tokens to be removed to -inf
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        logits = torch.full_like(logits, float("-inf"))

        # scatter the sorted logits back to their original positions
        logits.scatter_(
            dim = -1,
            index = sorted_indices,
            src = sorted_logits
        )

    return logits

def sample_next_token(logits, temperature = 1.0, top_k = 0, top_p = 1.0):
    """
    sample the next token from the logits
    logits: [batch, vocab_size]
    temperature: float, temperature for sampling
    top_k: int, number of top logits to keep
    top_p: float, cumulative probability threshold for nucleus sampling
    """

    if temperature <= 0:
        # negative temperature, do not sample, just return the argmax
        # shape: [batch, vocab_size] -> [batch, 1]
        return torch.argmax(logits, dim = -1, keepdim = True)  # greedy sampling

    # apply temperature
    # higher temperature, distance close, more randomness
    # lower temperature, distance far, less randomness
    logits = logits / temperature

    # apply top-k and top-p sampling
    logits = top_k_logits(logits, top_k = top_k, top_p = top_p)

    # softmax to get probabilities
    probs = torch.softmax(logits, dim = -1)

    # sample from the distribution
    # shape: [batch, vocab_size] -> [batch, 1]
    next_token = torch.multinomial(probs, num_samples = 1)

    return next_token


