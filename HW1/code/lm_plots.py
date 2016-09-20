import numpy as np
import languagemodel as lm
import argparse
import sys
import example
import pickle
import matplotlib.pyplot as plt

with open("lm_search_train.dict","r") as f:
	trainLLs = pickle.load(f)

with open("lm_search_dev.dict","r") as f:
	devLLs = pickle.load(f)

with open("lm_search_test.dict","r") as f:
	devLLs = pickle.load(f)

maxParams = max(devLLs, key = devLLs.get)

# Plots
ngrams = [2, 3, 5]
dwords = [5, 10, 20]
dhids = [10, 30, 50]
for ngram in ngrams:
	fig = plt.figure()
	for dword in dwords:
		LLvalues = []
		for dhid in dhids:
			devLLs