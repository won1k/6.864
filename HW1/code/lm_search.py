import numpy as np
import languagemodel as lm
import argparse
import sys
import example
import pickle

ngrams = [2, 3, 5, 8]
dwords = [5, 10, 20, 50]
dhids = [10, 30, 50, 100]
trainLLs = {}
devLLs = {}

for ngram in ngrams:
	for dword in dwords:
		for dhid in dhids:
			trainLL, devLL = example.nnlm(ngram, dword, dhid, 10)
			trainLLs[(ngram, dword, dhid)] = trainLL
			devLLs[(ngram, dword, dhid)] = devLL
			print(trainLL, devLL)

with open("lm_search_train.dict","w") as f:
	pickle.dump(trainLLs, f)

with open("lm_search_dev.dict","w") as f:
	pickle.dump(devLLs, f)

