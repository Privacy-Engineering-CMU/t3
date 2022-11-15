import pandas as pd
import numpy as np
from snsynth import Synthesizer
from snsynth.pytorch.nn import PATECTGAN
from snsynth.pytorch import PytorchDPSynthesizer

pums = pd.read_csv("../data/preprocessed_10000_entries.csv", index_col=None) # in datasets/
pums = pums.drop(['UserID'], axis=1)
pums = pums.drop(['MovieName'], axis=1)
pums = pums.drop(['Timestamp'], axis=1)
pums = pums.drop(['Unnamed: 0'], axis=1)

# Calculate average rating from raw data
mean_pums = pums['rating'].mean()

# Calculate average rating from MWEM Synthetic Data
synth1 = Synthesizer.create('mwem', epsilon=1.0)
sample1 = synth1.fit_sample(pums)
mean_sample1 = sample1['rating'].mean()

# Calculate average rating from DPCTgan Synthetic Data
synth2 = Synthesizer.create('dpctgan', epsilon=1.0, verbose=True)
sample2 = synth2.fit_sample(pums, preprocessor_eps=0.5)
mean_sample2 = sample2['rating'].mean()

# Print average values
print('\n')
print('--------------------------------------------')
print('Average rating with no DP:', mean_pums)
print('Average rating with MWEM Synthetic Data:', mean_sample1)
print('Average rating with DPCT GAN Synthetic Data:', mean_sample2)
print('--------------------------------------------')
print('\n')

pums.to_csv("../data/original_data.csv")
sample1.to_csv("../data/sample1.csv")
sample2.to_csv("../data/sample2.csv")
