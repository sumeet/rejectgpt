import re
from collections import Counter, defaultdict


def tokenize(text):
    """
    >>> tokenize('''"Hello, world!" It's a beautiful day.''')
    ['"', 'Hello', ',', 'world', '!', '"', 'It', "'", 's', 'a', 'beautiful', 'day', '.']
    """
    return re.findall(r"\b\w+\b|\S", text)


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
corpus_words = tokenize(corpus)
vocab = Counter(map(tuple, corpus_words))

num_merges = 2_000  # Number of merge operations to perform
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)

vocab_to_id = {subword: i for i, (subword, _) in enumerate(vocab.items())}
print(vocab_to_id)

#
# def bpe_tokenize(words, vocab):
#     tokenized_text = []  # List to store the tokenized version of the text
#
#     for word in words:
#         subwords = [
#             (char,) for char in word
#         ]  # Start by treating each character as a separate subword
#
#         # Merge subwords based on the vocab
#         i = 0
#         while i < len(subwords):
#             # Check if we can merge the next two subwords
#             if i < len(subwords) - 1 and (subwords[i] + subwords[i + 1]) in vocab:
#                 merged_subword = subwords[i] + subwords[i + 1]
#                 subwords.pop(i + 1)
#                 subwords[i] = merged_subword
#             else:
#                 i += 1
#
#         # Append the resulting subwords to the tokenized text
#         tokenized_text.extend(subwords)
#
#     return tokenized_text
#
#
# f = bpe_tokenize(corpus_words, vocab)
#
# print(f)
