import sklearn.linear_model as lm
import operator
from utils import *
import numpy as np
import sys

# Initialize constants/dicts
PAD = "PAD"
TAG = "TAG"
FEATURE_DICT = {0: word_tag, 1: caps_tag, 2: prefix_tag, 3: suffix_tag, 4: bigram, 5: trigram, 6: context1, 7: context2}

def read_data(file_name):
    id = []
    X = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip().split(' ')
            if len(line) == 1:
                id += line
            else:
                words = []
                tags = []
                for wt in line:
                    w, t = wt.split('_')
                    words.append(w)
                    tags.append(t)
                X += words
                y += tags
    return id, X, y

def preprocess(data, tags, features, train):
    '''
    For features:
    0 = word-tag
    1 = caps-tag
    2 = prefix-tag
    3 = suffix-tag
    4 = bigram
    5 = trigram
    6 = context-1 (x_{i-1}, y_i)
    7 = context-2 (x_{i-2}, y_i)
    [If feature_idxs not given (i.e. testing) then we
    use best-performing features on dev as default]
    '''

    # Preprocessing
    if train: # i.e. during training
        global dicts
        dicts = {}
        common_prefixes = get_common_prefix(data)
        common_suffixes = get_common_suffix(data)
        dicts['word'] = get_vocab(data)
        dicts['caps'] = {True: 1, False: 0}
        dicts['prefix'] = get_prefixes(common_prefixes)
        dicts['suffix'] = get_suffixes(common_suffixes)
        dicts['tag'] = get_tags(tags)
        print "Dictionaries loaded!"
        print "Vocab size:", len(dicts['word'])
        print "Tag size:", len(dicts['tag'])
    
    # Extract features
    X = get_features(data, tags, features, dicts)
    return X

def get_features(data, tags, features, dicts):
    for i, feature in enumerate(features):
        if i == 0:
            X = FEATURE_DICT[feature](data, tags, dicts)
        else:
            X = np.hstack((X, FEATURE_DICT[feature](data, tags, dicts)))
    return X
            
def make_model():
    return lm.LogisticRegression(solver = 'sag', max_iter = 1e3,  multi_class = 'multinomial')

#def viterbi():


if __name__ == '__main__':
    # Parse parameters
    print("Loading params...")
    train_id, train_data, train_tag = read_data(sys.argv[1])
    test_id, test_data, _ = read_data(sys.argv[2])
    model_idx = int(sys.argv[3])
    try:
        feature_idxs = [int(idx) for idx in sys.argv[4:]]
    except:
        feature_idxs = range(5)

    # Extract features
    print("Extracting features...")
    train_features = preprocess(train_data, train_tag, feature_idxs, True)
    #test_features = get_features(test_data, feature_idxs)

    # Make model
    print("Generating model...")
    model = make_model()

    # Train model
    print("Training...")
    model.fit(train_features, train_tag)

    # Save model
    
