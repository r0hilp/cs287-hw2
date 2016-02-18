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
    cap_features = []
    lbl = []
    window_features = []
    window_cap_features = []
    window_lbl = []
    with codecs.open(data_name, "r", encoding="latin-1") as f:
        # initial padding
        features.extend([1] * (window_size/2))
        cap_features.extend([1] * (window_size/2))
        lbl.extend([0] * (window_size/2))
        
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                # add padding word
                features.extend([1] * (window_size/2))
                cap_features.extend([1] * (window_size/2))
                lbl.extend([0] * (window_size/2))
            else:
                _, _, word, tag = tuple(line.split('\t'))
                lower_caps = int(word.islower())
                all_caps = int(word.isupper())
                first_letter_cap = int(word[0].isupper())
                has_one_cap = int(any(let.isupper() for let in word))
                cap = 1
                if lower_caps:
                    cap = 2
                elif all_caps:
                    cap = 3
                elif first_letter_cap:
                    cap = 4
                elif has_one_cap:
                    cap = 5
                word = clean_str(word, common_words_list)

                features.append(word_to_idx[word])
                cap_features.append(cap)
                lbl.append(tag_to_id[tag])
        # end padding
        features.extend([1] * (window_size/2))
        cap_features.extend([1] * (window_size/2))
        lbl.extend([0] * (window_size/2))

    # Convert to windowed features
    for i in range(len(features)):
        # Skip padding
        if features[i] == 1:
            continue
        else:
            # window_idxs = range(i - window_size/2, i + window_size/2 + 1)
            i_low = i - window_size/2
            i_high = i + window_size/2 + 1
            window_features.append(features[i_low:i_high])
            window_cap_features.append(cap_features[i_low:i_high])
            window_lbl.append(lbl[i])
    return np.array(window_features, dtype=np.int32), np.array(window_cap_features, dtype=np.int32), np.array(window_lbl, dtype=np.int32)

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

def load_word_vecs(file_name, vocab):
    # Get word vecs from glove
    word_vecs = {}
    with codecs.open(file_name, "r", encoding="latin-1") as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            vals = vals[1:]
            if word in vocab:
                word_vecs[word] = vals

    return word_vecs


FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt"),
              "test": ("data/test_file.txt","data/test_file.txt","data/test_file.txt")}
TAG_DICT = "data/tags.dict"
WORD_VECS_PATH = 'data/glove.6B.50d.txt'
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
    train, valid, test = FILE_PATHS[dataset]

    # Fix this for now
    window_size = 5

    # Retrieve tag to id mapping
    print 'Get tag ids...'
    tag_to_id = get_tag_ids(TAG_DICT)

    # Retrieve common words
    print 'Getting common words...'
    common_words_list = get_common_words(WORD_VECS_PATH)

    # Get index features
    print 'Getting vocab...'
    word_to_idx = get_vocab([train, valid, test], common_words_list, dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_cap_input, train_output = convert_data(train, word_to_idx, tag_to_id, window_size, common_words_list, dataset)

    if valid:
        valid_input, valid_cap_input, valid_output = convert_data(valid, word_to_idx, tag_to_id, window_size, common_words_list, dataset)

    if test:
        test_input, test_cap_input, _ = convert_data(test, word_to_idx, tag_to_id, window_size, common_words_list, dataset)

    # +4 for cap features
    # V = len(word_to_idx) + 1 + 4
    V = len(word_to_idx) + 1
    print('Vocab size:', V)

    C = len(tag_to_id)

    # Get word vecs
    print 'Getting word vecs...'
    word_vecs = load_word_vecs(WORD_VECS_PATH, word_to_idx)
    embed = np.random.uniform(-0.25, 0.25, (V, len(word_vecs.values()[0])))
    # zero out padding
    embed[0] = 0
    for word, vec in word_vecs.items():
        embed[word_to_idx[word] - 1] = vec

    print 'Saving...'
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_cap_input'] = train_cap_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_cap_input'] = valid_cap_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
            f['test_cap_input'] = test_cap_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)

        f['word_vecs'] = embed


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
