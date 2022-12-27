import argparse
from typing import Dict

import numpy as np


def load_embedder():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='../../aggregated_vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='../../aggregated_vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    d = (np.sum(W ** 2, 1) ** 0.5)
    W_norm = (W.T / d).T
    return W_norm, vocab


def evaluate_vectors_for_single_text(W: np.array, vocab: Dict, text: str):
    data = [word for word in text if word in vocab]
    indices = np.array([vocab[word] for word in data])
    embedded_text = W[indices:]
    return embedded_text
