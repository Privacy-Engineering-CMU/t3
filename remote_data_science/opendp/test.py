import pandas as pd
import numpy as np
from snsynth import Synthesizer
from snsynth.pytorch.nn import PATECTGAN
from snsynth.pytorch import PytorchDPSynthesizer

pums = pd.read_csv("/Users/davidzagardo/Dropbox/My Mac (Davids-MacBook-Pro.local)/Desktop/SOFTWARE_PROJECTS/t3/remote_data_science/data/preprocessed_10000_entries.csv", index_col=None) # in datasets/
print("here!")
pums = pums.drop(['UserID'], axis=1)
pums = pums.drop(['MovieName'], axis=1)
pums = pums.drop(['Timestamp'], axis=1)
pums = pums.drop(['Unnamed: 0'], axis=1)
print(pums)

synth1 = Synthesizer.create('mwem', epsilon=10000)
sample1 = synth1.fit_sample(pums)
print(sample1)

sample2 = synth1.sample(10) # get 10 synthetic rows
print(sample2)

synth2 = Synthesizer.create('dpctgan', epsilon=1.0, verbose=True)
sample3 = synth2.fit_sample(pums, preprocessor_eps=0.5)
print(sample3)