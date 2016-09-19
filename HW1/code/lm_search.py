import numpy as np
import languagemodel as lm
import argparse
import sys
import example

ngrams = [2, 3, 5, 8]
dwords = [5, 10, 20, 50, 100]
dhids = [10, 30, 50, 100]
lls = {}

for ngram in ngrams:
	for dword in dwords:
		for dhid in dhids:
			lls[(ngram, dword, dhid)] = example.nnlm(ngram, dword, dhid, 10)

print(lls)