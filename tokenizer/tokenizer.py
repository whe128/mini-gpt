# tokenizer/tokenizer.py

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir = "./cache"
            )

        # if the tokenizer does not have a pad token, we add it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text, add_eos = False):
        """
            Encodes the input text into a list of token IDs.
        """
        ids = self.tokenizer.encode(text, add_special_tokens = False)

        if add_eos:
            ids.append(self.tokenizer.eos_token_id)

        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens = True)


if __name__ == "__main__":
    tokenizer = Tokenizer()

    ids = tokenizer.encode("Hello, how are you?")
    print("Encoded IDs:", ids)
    print("Decoded text:", tokenizer.decode(ids))

    ids = tokenizer.encode("Hello, how are you?", add_eos = True)
    print("Encoded IDs with EOS:", ids)
    print("Decoded text with EOS:", tokenizer.decode(ids))

    print("Vocabulary size:", tokenizer.vocab_size())
    print(tokenizer.encode("你"))
    print(tokenizer.encode("在"))
    print(tokenizer.encode("哪"))
    print(tokenizer.encode("里"))
    print(tokenizer.encode("你在哪里"))

    print(tokenizer.encode("Transformer"))
    print(tokenizer.encode("Transform"))
    print(tokenizer.encode("transform"))
    print(tokenizer.encode("er"))
    print(tokenizer.encode("trans"))
    print(tokenizer.encode("form"))
    print(tokenizer.encode("former"))
