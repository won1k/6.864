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
	testLLs = pickle.load(f)

maxParams = max(devLLs, key = devLLs.get)

# Plots
ngrams = [2, 3, 5]
dwords = [5, 10, 20]
dhids = [10, 30, 50]
for ngram in ngrams:
	fig = plt.figure()
	for dword in dwords:
		trainLLvalues = []
		for dhid in dhids:
			trainLLvalues.append(trainLLs[(ngram, dword, dhid)])
		plt.plot(dhids, trainLLvalues, label = str(dhid))
	plt.xlabel('hidden dim')
	plt.ylabel('average LL')
	plt.legend()
	plt.savefig('hw1_train_n' + str(ngram) + '_d' + str(dword) + '.pdf')

for ngram in ngrams:
	fig = plt.figure()
	for dword in dwords:
		testLLvalues = []
		for dhid in dhids:
			testLLvalues.append(testLLs[(ngram, dword, dhid)])
		plt.plot(dhids, testLLvalues, label = str(dhid))
	plt.xlabel('hidden dim')
	plt.ylabel('average LL')
	plt.legend()
	plt.savefig('hw1_test_n' + str(ngram) + '_d' + str(dword) + '.pdf')