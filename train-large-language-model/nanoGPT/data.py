'''
Implementation from: https://github.com/karpathy/nanoGPT

Small changes made by Terence Broad (2024) to make this code adaptable to using custom datasets.
'''


import os
import pickle
import tiktoken
import numpy as np

def prepare_dataset_gpt_tokenizer(input_file_path, dataset_path):
    with open(input_file_path, 'r') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(dataset_path, 'train.bin'))
    val_ids.tofile(os.path.join(dataset_path, 'val.bin'))


def prepare_dataset_as_chars(input_file_path, dataset_path):
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(dataset_path, 'train.bin'))
    val_ids.tofile(os.path.join(dataset_path, 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(dataset_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
