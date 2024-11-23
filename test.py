def tokenize(string: str) -> list:
    tokens = ''.join([token for token in string if 
                  token.isalpha() is True or token == ' '])
    tokens = [token.lower() for token in tokens.split()]
    return tokens

def token_counts(string: str, k: int = 1) -> dict:
    tokens = tokenize(string)
    token_set = set(tokens)
    word_frequencies = {word:tokens.count(word) for word in token_set if 
                        tokens.count(word) > k}
    return word_frequencies

text = """The quick brown fox jumps over the lazy dog. The fox and the dog play together. 
            The fox chases the dog, but the dog runs quickly. The fox is fast, and the dog escapes."""


print(token_counts(text))