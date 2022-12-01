import pandas as pd
import numpy as np
from snsynth import Synthesizer
from snsynth.pytorch.nn import PATECTGAN
from snsynth.pytorch import PytorchDPSynthesizer

# This file generates synthetic movie rating data
# using 3 different Differential Privacy Algorithms.
#
# The file preprocessed_10000_entires.csv in ./data
# contains approximately 10,000 entries from users
# that have rated movies on a 1 to 5 scale. The
# code below processes the data and feeds it into
# synthetic data generation algorithms using the
# OpenDP library.
#
# The code then calculates the average movie rating
# across each dataset, original and synthetic, and
# prints the results.

# Reading data from CSV in 'data' directory
pums = pd.read_csv("../data/preprocessed_10000_entries.csv", index_col=None) # in datasets/

# # Uncomment to train with 170,000 entries
# # Reading data from CSV in 'data' directory
# pums = pd.read_csv("../data/preprocessed.csv", index_col=None) # in datasets/

print('\n')
print('-------------------------------')
print('Dropping Columns From DataFrame')
print('-------------------------------')
print('\n')
# Dropping columns from dataframe
pums = pums.drop(['UserID'], axis=1)
pums = pums.drop(['MovieName'], axis=1)
pums = pums.drop(['Timestamp'], axis=1)
pums = pums.drop(['Unnamed: 0'], axis=1)

# Calculate average movie rating from raw data
mean_pums = pums['rating'].mean()

print('\n')
print('---------------------------------------------------------------------------')
print('Generating Synthetic Data with Multiplicative Weights Exponential Mechanism')
print('---------------------------------------------------------------------------')
print('\n')
# Calculate average movie rating after generating MWEM Synthetic Data
# Multiplicative Weights Exponential Mechanism
synth1 = Synthesizer.create('mwem', epsilon=1.0)
sample1 = synth1.fit_sample(pums)
mean_sample1 = sample1['rating'].mean()

print('\n')
print('-----------------------------------------------------------------------------')
print('Generating Synthetic Data with Differentially Private Conditional Tabular GAN')
print('-----------------------------------------------------------------------------')
print('\n')
# Calculate average movie rating after generating DP-CTGAN Synthetic Data
# Differentially Private Conditional Tabular GAN
synth2 = Synthesizer.create('dpctgan', epsilon=1.0, verbose=True)
sample2 = synth2.fit_sample(pums, preprocessor_eps=0.5)
mean_sample2 = sample2['rating'].mean()

print('\n')
print('-----------------------------------------------------------------------------------------------')
print('Generating Synthetic Data with Private Aggregation of Teacher Ensembles Conditional Tabular GAN')
print('-----------------------------------------------------------------------------------------------')
print('\n')
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

# Write to CSV files
# Compare these side by side in the ./data
# folder to see how the DP noise changed
# the individual rating values!
pums.to_csv("../data/original_data.csv")
sample1.to_csv("../data/sample1.csv")
sample2.to_csv("../data/sample2.csv")
sample3.to_csv("../data/sample3.csv")
