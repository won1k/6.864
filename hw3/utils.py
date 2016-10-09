import operator
import numpy as np
import itertools

# Initialize constants
N = 100 # most common prefix/suffix
PAD = "PAD"
UNK = "UNK"
TAG = "TAG"

def get_vocab(data):
    w2idx = {}
    w2idx[UNK] = 0
    w2idx[PAD] = 1
    idx = 2
    for word in data:
        word = word.lower()
        if word not in w2idx:
           w2idx[word] = idx
           idx += 1
    return w2idx

def get_prefixes(common_prefixes):
    p2idx = {}
    idx = 0
    for prefix in common_prefixes:
        p2idx[prefix] = idx
        idx += 1
    return p2idx

def get_suffixes(common_suffixes):
    s2idx = {}
    idx = 0
    for suffix in common_suffixes:
        s2idx[suffix] = idx
        idx += 1
    return s2idx

def get_tags(tags):
    t2idx = {}
    idx = 0
    for tag in tags:
        if tag not in t2idx:
            t2idx[tag] = idx
            idx += 1
    return t2idx

def get_common_prefix(data):
    prefix_counts = {}
    for word in data:
        word = word.lower()
        for i in range(1,4):
            try:
                prefix_counts[word[:i]] += 1
            except:
                prefix_counts[word[:i]] = 1
    # Sort by number of counts
    sorted_counts = sorted(prefix_counts.items(), key = operator.itemgetter(1))
    sorted_counts.reverse()
    # Get N most common prefix
    return [k for k,v in sorted_counts[:N]]

def get_common_suffix(data):
    suffix_counts = {}
    for word in data:
        word = word.lower()
        for i in range(1,4):
            try:
                suffix_counts[word[-i:]] += 1
            except:
                suffix_counts[word[-i:]] = 1
    # Sort by number of counts
    sorted_counts = sorted(suffix_counts.items(), key = operator.itemgetter(1))
    sorted_counts.reverse()
    # Get N most common suffix
    return [k for k,v in sorted_counts[:N]]

def word_tag(data, tags, dicts):
    w2idx = dicts['word']
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(w2idx)*len(t2idx)), dtype = np.int8)
    for i,w,t in itertools.izip(range(nsamples), data, tags):
        X[i][w2idx[w.lower()]*t2idx[t]] = 1
    print "Extracted word features!"
    return X

def caps_tag(data, tags, dicts):
    c2idx = dicts['caps']
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(c2idx)*len(t2idx)), dtype = np.int8)
    for i,w,t in itertools.izip(range(nsamples), data, tags):
        X[i][c2idx[w[0].isupper()]*t2idx[t]] = 1
    print "Extracted caps features!"
    return X

def prefix_tag(data, tags, dicts):
    p2idx = dicts['prefix']
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(p2idx)*len(t2idx)), dtype = np.int8)
    for i,w,t in itertools.izip(range(nsamples), data, tags):
        for prefix in p2idx.keys():
            if w.lower().startswith(prefix):
                X[i][p2idx[prefix]*t2idx[t]] = 1
    print "Extracted prefix features!"
    return X

def suffix_tag(data, tags, dicts):
    s2idx = dicts['suffix']
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(s2idx)*len(t2idx)), dtype = np.int8)
    for i,w,t in itertools.izip(range(nsamples), data, tags):
        for suffix in s2idx.keys():
            if w.lower().endswith(suffix):
                X[i][s2idx[suffix]*t2idx[t]] = 1
    print "Extracted suffix features!"
    return X
    
def bigram(data, tags, dicts):
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(t2idx)**2), dtype = np.int8)
    for i,t1,t2 in itertools.izip(range(nsamples), [TAG] + tags, tags):
        X[i][t2idx[t1]*t2idx[t2]] = 1
    print "Extracted bigram features!"
    return X

def trigram(data, tags, dicts):
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(t2idx)**3), dtype = np.int8)
    for i,t1,t2,t3 in itertools.izip(range(nsamples), [TAG, TAG] + tags, [TAG] + tags, tags):
        X[i][t2idx[t1]*t2idx[t2]*t2idx[t3]] = 1
    print "Extracted trigram features!"
    return X

def context1(data, tags, dicts):
    w2idx = dicts['word']
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(w2idx)*len(t2idx)), dtype = np.int8)
    for i,w,t in itertools.izip(range(nsamples), [PAD] + data, tags):
        X[i][w2idx[w]*t2idx[t]] = 1
    print "Extracted context-1"
    return X

def context2(data, tags, dicts):
    w2idx = dicts['word']
    t2idx = dicts['tag']
    nsamples = len(data)
    X = np.zeros((nsamples, len(w2idx)*len(t2idx)), dtype = np.int8)
    for i,w,t in itertools.izip(range(nsamples), [PAD, PAD] + data, tags):
        X[i][w2idx[w]*t2idx[t]] = 1
    print "Extracted context-2"
    return X





def get_word(word, w2idx, vocab_size):
    # Get word idx
    try:
        idx = w2idx[word]
    except:
        idx = w2idx[UNK]
    # Turn to 1-hot
    word_idx = np.zeros(vocab_size, dtype = np.int)
    word_idx[idx] = 1
    return list(word_idx)

def get_cap(word, c2idx):
    # Get cap idx
    idx = c2idx[word[0].isupper()]
    cap_idx = np.zeros(2, dtype = np.int)
    cap_idx[idx] = 1
    return list(cap_idx)

def get_numcaps(word):
    n = 0
    for letter in word:
        if letter.isupper():
            n += 1
    return [n]

def get_prefix(word, p2idx):
    prefix_idx = np.zeros(N, dtype = np.int)
    return list(prefix_idx)

def get_suffix(word, s2idx):
    suffix_idx = np.zeros(N, dtype = np.int)
    word = word.lower()
    for suffix in s2idx.keys():
        if word.endswith(suffix):
            suffix_idx[s2idx[suffix]] = 1
    return list(suffix_idx)

def get_tag(tag, tag_size):
    idx = t2idx[tag]
    tag_idx = np.zeros(tag_size, dtype = np.int)
    tag_idx[idx] = 1
    return list(tag_idx)
