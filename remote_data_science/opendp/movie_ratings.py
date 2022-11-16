import pandas as pd
import numpy as np
from snsynth import Synthesizer
from snsynth.pytorch.nn import PATECTGAN
from snsynth.pytorch import PytorchDPSynthesizer

# Reading data from CSV in 'data' directory
pums = pd.read_csv("../data/preprocessed_10000_entries.csv", index_col=None) # in datasets/

# Dropping columns from dataframe
pums = pums.drop(['UserID'], axis=1)
pums = pums.drop(['MovieName'], axis=1)
pums = pums.drop(['Timestamp'], axis=1)
pums = pums.drop(['Unnamed: 0'], axis=1)

# Calculate average movie rating from raw data
mean_pums = pums['rating'].mean()

# Calculate average movie rating after generating MWEM Synthetic Data
# Multiplicative Weights Exponential Mechanism
synth1 = Synthesizer.create('mwem', epsilon=1.0)
sample1 = synth1.fit_sample(pums)
mean_sample1 = sample1['rating'].mean()

# Calculate average movie rating after generating DP-CTGAN Synthetic Data
# Differentially Private Conditional Tabular GAN
synth2 = Synthesizer.create('dpctgan', epsilon=1.0, verbose=True)
sample2 = synth2.fit_sample(pums, preprocessor_eps=0.5)
mean_sample2 = sample2['rating'].mean()

# Calculate average movie rating after generating PATE-CTGAN Synthetic Data
# Private Aggregation of Teacher Ensembles Conditional Tabular GAN
synth3 = Synthesizer.create('patectgan', epsilon=1.0, verbose=True)
sample3 = synth3.fit_sample(pums, preprocessor_eps=0.5)
mean_sample3 = sample3['rating'].mean()

# Print average values
print('\n')
print('------------------------------------------------------------------------')
print('Average movie rating with no DP:                     ', mean_pums)
print('Average movie rating with MWEM Synthetic Data:       ', mean_sample1)
print('Average movie rating with DPCT GAN Synthetic Data:   ', mean_sample2)
print('Average movie rating with PATE CTGAN Synthetic Data: ', mean_sample3)
print('------------------------------------------------------------------------')
print('\n')

pums.to_csv("../data/original_data.csv")
sample1.to_csv("../data/sample1.csv")
sample2.to_csv("../data/sample2.csv")
sample3.to_csv("../data/sample3.csv")
