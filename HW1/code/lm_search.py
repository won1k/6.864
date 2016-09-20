import numpy as np
import languagemodel as lm
import argparse
import sys
import example
import pickle

ngrams = [2, 3, 5]
dwords = [5, 10, 20]
dhids = [10, 30, 50]
trainLLs = {}
devLLs = {}
testLLs = {}
biDev = 0
biTest = 0

for ngram in ngrams:
	for dword in dwords:
		for dhid in dhids:
			trainLL, devLL, testLL, biDevLL, biTestLL = example.nnlm(ngram, dword, dhid, 20)
			trainLLs[(ngram, dword, dhid)] = trainLL
			devLLs[(ngram, dword, dhid)] = devLL
			testLLs[(ngram, dword, dhid)] = testLL
			biDev = biDevLL
			biTest = biTestLL
			print(trainLL, devLL)

with open("lm_search_train.dict","w") as f:
	pickle.dump(trainLLs, f)

with open("lm_search_dev.dict","w") as f:
	pickle.dump(devLLs, f)

with open("lm_search_test.dict","w") as f:
	pickle.dump(testLLs, f)

print("saved!")
print('Bi-gram Dev LL: ' + str(biDev))
print('Bi-gram Test LL: ' + str(biTest))