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

maxParams = max(devLLs, key = devLLs.get)

# Plots
