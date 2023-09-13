import re
from collections import Counter, defaultdict


def encode_corpus(corpus_tokens, vocab_to_id):
    encoded_corpus = []
    for token in corpus_tokens:
        if token == " ":
            encoded_corpus.append(vocab_to_id[" "])
            continue

        # Convert string to a list of characters
        token_chars = list(token)

        i = 0
        while i < len(token_chars):
            longest_sub_token = ""
            for j in range(i + 1, len(token_chars) + 1):
                sub_token = "".join(token_chars[i:j])

                if sub_token in vocab_to_id:
                    longest_sub_token = sub_token

            assert (
                len(longest_sub_token) > 0
            ), f"Could not find sub-token for {token_chars[i]}"

            # Translate the subword to its ID and append it to encoded_corpus
            encoded_corpus.append(vocab_to_id[longest_sub_token])
            i += len(longest_sub_token)  # Move to the end of the longest sub-token

    return encoded_corpus


def tokenize_for_vocab(text):
    """
    >>> tokenize_for_vocab('''"Hello, world!" It's a beautiful day.''')
    ['"', 'Hello', ',', 'world', '!', '"', 'It', "'", 's', 'a', 'beautiful', 'day', '.']
    """
    return re.findall(r"\b\w+\b|\S", text)


def add_spaces_to_tokenized(tokens):
    """
    >>> add_spaces_to_tokenized(['It', "'", 's', 'a', 'beautiful', 'day', '.'])
    ['It', "'", 's', ' ', 'a', ' ', 'beautiful', ' ', 'day', '.']
    """
    with_spaces = []
    for i in range(len(tokens)):
        pre = tokens[i]
        with_spaces.append(pre)
        if i < len(tokens) - 1:
            post = tokens[i + 1]
            if re.match(r"\w", pre) and re.match(r"\w", post):
                with_spaces.append(" ")
    return with_spaces


# Initialize vocabulary from the training corpus (a list of words)
# Here, each word is represented as a tuple of characters, followed by its frequency count
# vocab = Counter({
#     ('l', 'o', 'w'): 5,
#     ('l', 'o', 'w', 'e', 'r'): 2,
#     ('n', 'e', 'w'): 5,
#     ('n', 'e', 'w', 'e', 'r'): 2,
#     ('w', 'i', 'd', 'e', 's', 't'): 3
# })


# Function to get statistics of adjacent pairs in the vocabulary
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[word[i], word[i + 1]] += freq
    return pairs


# Function to merge the most frequent pair in the vocabulary
def merge_vocab(pair, vocab):
    new_vocab = {}
    replacement = "".join(pair)
    for word in vocab:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = vocab[word]
    return new_vocab


corpus = open("corpus/fountainhead.txt").read()
# for now let's just lowercase everything
corpus = corpus.lower()
corpus_words = tokenize_for_vocab(corpus)
vocab = Counter(map(tuple, corpus_words))

num_merges = 2000  # Number of merge operations to perform

vocab_to_id = {}
id = 0
for i in range(num_merges):
    pairs = get_stats(vocab)

    for word in vocab:
        for char in word:
            if char not in vocab_to_id:
                vocab_to_id[char] = id
                id += 1

    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)

    for word in vocab:
        for char in word:
            if char not in vocab_to_id:
                vocab_to_id[char] = id
                id += 1

# manually add space to the vocab
vocab_to_id[" "] = id
id += 1

print(vocab_to_id)
print(len(vocab_to_id))

corpus_tokens = add_spaces_to_tokenized(corpus_words)
print(corpus_tokens)

# Now, you would run the function like this:
encoded_corpus = encode_corpus(corpus_tokens, vocab_to_id)

# id_to_vocab = {v: k for k, v in vocab_to_id.items()}
# print(list(map(lambda x: id_to_vocab[x], encoded_corpus)))
