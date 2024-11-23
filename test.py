import numpy as np

text = "The quick brown fox jumps over the lazy dog!"

def tokenize(string: str) -> list:
    tokens = ''.join([token for token in string if 
                  token.isalpha() is True or token == ' '])
    tokens = [token.lower() for token in tokens.split()]
    return tokens



word_frequencies = {word:tokenize(text).count(word) for word in tokenize(text)}


def token_counts(string: str, k: int = 1) -> dict:
    tokens = tokenize(string)
    word_frequencies = {word:tokens.count(word) for word in tokens if 
                        tokens.count(word) > k}
    return word_frequencies

token_to_id = {token:id for id, token in enumerate(set(tokenize(text)))}

# Expected output: {'dog': 0, 'quick': 1, 'fox': 2, 'the': 3, 'over': 4, 'lazy': 5, 'brown': 6, 'jumps': 7}
print(token_to_id)

id_to_token = {id:token for token, id in token_to_id.items()}

print(id_to_token)

def make_vocabulary_map(documents: list) -> tuple:
    vocab = set()
    tokens = set([tokenize(document) for document in documents])


def make_vocabulary_map(documents: list) -> tuple:
    vocab = set()
    for document in documents:
        vocab = vocab.union(tokenize(document))

    token2int = {token:id for id, token in enumerate(vocab)}
    int2token = {id:token for token, id in token2int.items()}
    return token2int, int2token

def tokenize_and_encode(documents: list) -> list:
    # Hint: use your make_vocabulary_map and tokenize function
    t2i, i2t = make_vocabulary_map(documents)
    enc = []
    for document in documents:
        enc.append([t2i[word] for word in tokenize(document)])
    return enc, t2i, i2t

# Test:
enc, t2i, i2t = tokenize_and_encode([text, 'What a luck we had today!'])
" | ".join([" ".join(i2t[i] for i in e) for e in enc]) == 'the quick brown fox jumps over the lazy dog | what a luck we had today'

sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Test:
print(np.all(sigmoid(np.log([1, 1/3, 1/7])) == np.array([1/2, 1/4, 1/8])))

enc, t2i, i2t = tokenize_and_encode([text, 'What a luck we had today!'])
print(" | ".join([" ".join(i2t[i] for i in e) for e in enc]) == 'the quick brown fox jumps over the lazy dog | what a luck we had today')


def rnn_layer(w: np.array, list_of_sequences: list[np.array], sigma=sigmoid ) -> np.array:
    W = w[0:9].reshape(3, 3)
    U = w[9:18].reshape(3, 3)
    B = w[18:21].reshape(1, 3)

    nr_sequences = len(list_of_sequences)
    outputs = np.zeros(nr_sequences)

    for i in range(nr_sequences):
        X = list_of_sequences[i]
        a = 0 * X[1, ]
        for j in range(len(X)):
            a = np.matmul(W, X[j,]) + np.matmul(U, a)
    
        outputs[i] = np.matmul(B, a)

    return outputs


# Test
def rnn_layer(w: np.array, list_of_sequences: list[np.array], sigma=sigmoid ) -> np.array:
    W = w[0:9].reshape(3, 3)
    U = w[9:18].reshape(3, 3)
    B = w[18:21].reshape(1, 3)

    nr_sequences = len(list_of_sequences)
    outputs = np.zeros(nr_sequences)

    for i in range(nr_sequences):
        X = list_of_sequences[i]
        a = 0 * X[1, ]
        for j in range(len(X)):
            a = np.matmul(W, X[j,]) + np.matmul(U, a)
    
        outputs[i] = np.matmul(B, a)

    return outputs


# Test
np.random.seed(10)
list_of_sequences = [np.random.normal(size=(5,3)) for _ in range(100)]
wstart = np.random.normal(size=(3*3 + 3*3 + 3)) 
o = rnn_layer(wstart, list_of_sequences)
print(o.shape)