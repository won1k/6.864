import numpy as np
import languagemodel as lm
import argparse
import sys
import example
import pickle

# Train using best config
bestParams = (3, 5, 50)
trainLL, devLL, testLL, biDevLL, biTestLL, neurallm = example.nnlm(bestParams[0], bestParams[1], bestParams[2], 20)

# Generate n-grams for examples
spell1 = lm.readCorpus("data/spell.txt")
spell2 = lm.readCorpus("data/spellx.txt")
with open("data/corpus.dict","r") as f:
	w2index = pickle.load(f)

spell1grams = lm.ngramGen(spell1, w2index, bestParams[0])
spell2grams = lm.ngramGen(spell2, w2index, bestParams[0])
LL1 = 0
for ng in spell1grams:
    pr = neurallm.prob(ng)
    LL1 += np.log(pr)

print("Log-likelihood of correct: " + str(LL1))

LL2 = 0
for ng in spell2grams:
    pr = neurallm.prob(ng)
    LL2 += np.log(pr)

print("Log-likelihood of incorrect: " + str(LL2))
