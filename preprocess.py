#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.
def clean_str(string, common_words_list):
    # remove capitalization and replace numbers
    string = string.strip().lower()
    while(any(let.isdigit() for let in string)):
        string = re.sub(r"[0-9]+", "NUMBER", string)
    if string not in common_words_list:
        string = 'RARE'
    return string

def get_common_words(file_name):
    # Get list of common words from glove file (change to trie if this is slow)
    common_words_list = []
    with codecs.open(file_name, "r", encoding="latin-1") as f:
        for line in f:
            common_words_list.append(line.split(' ')[0])
    return common_words_list

def get_tag_ids(tag_dict):
    # Construct tag to id mapping
    tag_to_id = {}
    with codecs.open(tag_dict, 'r', encoding="latin-1") as f:
        for line in f:
            tag, id_num = tuple(line.split())
            tag_to_id[tag] = int(id_num)
    tag_to_id['_'] = 0
    return tag_to_id

def convert_data(data_name, word_to_idx, tag_to_id, window_size, common_words_list, dataset):
    # Construct index/capital feature sets for each file
    features = []
    lbl = []
    window_features = []
    window_lbl = []
    with codecs.open(data_name, "r", encoding="latin-1") as f:
        # initial padding
        for _ in range(window_size/2):
            features.append([1, 0, 0, 0, 0])
            lbl.append(0)
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                # add padding word
                features.append([1, 0, 0, 0, 0])
                lbl.append(0)
            else:
                global_id, _, word, tag = tuple(line.split('\t'))
                lower_caps = int(word.islower())
                all_caps = int(word.isupper())
                first_letter_cap = int(word[0].isupper())
                has_one_cap = int(any(let.isupper() for let in word))
                word = clean_str(word, common_words_list)
                features.append([word_to_idx[word], lower_caps, all_caps, first_letter_cap, has_one_cap])
                lbl.append(tag_to_id[tag])
        # end padding
        for _ in range(window_size/2):
            features.append([1, 0, 0, 0, 0])
            lbl.append(0)
    # Convert to windowed features
    for i in range(len(features)):
        # Skip padding
        if features[i][0] == 1:
            continue
        else:
            window_idxs = range(i - window_size/2, i + window_size/2 + 1)
            window = sum([features[j] for j in window_idxs], [])
            window_features.append(window)
            window_lbl.append(lbl[i])
    return np.array(window_features, dtype=np.int32), np.array(window_lbl, dtype=np.int32)

def get_vocab(file_list, common_words_list, dataset=''):
    # Construct index feature dictionary.
    word_to_idx = {}
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    line = line.rstrip()
                    if len(line) == 0:
                        continue
                    _, _, word, _ = tuple(line.split('\t'))
                    word = clean_str(word, common_words_list)
                    if word not in word_to_idx:
                        word_to_idx[word] = idx
                        idx += 1
    return word_to_idx


FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt",
                      "data/tags.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, tag_dict = FILE_PATHS[dataset]

    # Fix this for now
    window_size = 5

    # Retrieve tag to id mapping
    tag_to_id = get_tag_ids(tag_dict)

    # Retrieve common words
    common_words_list = get_common_words('data/glove.6B.50d.txt')

    # Get index features
    word_to_idx = get_vocab([train, valid, test], common_words_list, dataset)

    # Convert data
    train_input, train_output = convert_data(train, word_to_idx, tag_to_id, window_size, common_words_list, dataset)

    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, tag_to_id, window_size, common_words_list, dataset)

    if test:
        test_input, _ = convert_data(test, word_to_idx, tag_to_id, window_size, common_words_list, dataset)

    V = len(word_to_idx) + 1 + 4
    print('Vocab size:', V - 4)

    C = np.max(train_output)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
