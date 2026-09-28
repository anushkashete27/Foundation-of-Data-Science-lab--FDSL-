import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

# 1. Load dataset
df = pd.read_csv("Election Polls DataSets.csv")

# 2. Define parameters
n = int(df.loc[0, "sample_size"])
p = df.loc[0, "rep"] / 100

print("Number of trials:", n)
print("Probability of success:", p)

# 3. Generate possible number of successes and their probabilities
x = np.arange(0, n + 1)

pmf = binom.pmf(x, n, p)
cdf = binom.cdf(x, n, p)

# 4. Compute mean and standard deviation
mean, var = binom.stats(n, p, moments='mv')
std_dev = np.sqrt(var)

print("Mean (Expected successes):", mean)
print("Variance:", var)
print("Standard Deviation:", std_dev)

# 5. Plot the Binomial Distribution
plt.figure(figsize=(8, 5))

plt.bar(x, pmf, edgecolor='black', label='PMF')
plt.axvline(mean, linestyle='--', label='Mean')

plt.title(f'Binomial Distribution (n={n}, p={p:.2f})')
plt.xlabel('Number of Voters Supporting Candidate')
plt.ylabel('Probability')
plt.legend()

plt.show()

# 6. Answer real-world probability questions

# Probability of exactly k successes
k = 500
p_exactly = binom.pmf(k, n, p)
print('P(X = 500):', p_exactly)

# Probability of at least k successes
p_at_least = 1 - binom.cdf(k - 1, n, p)
print('P(X >= 500):', p_at_least)

# Probability of at most k successes
p_at_most = binom.cdf(k, n, p)
print('P(X <= 500):', p_at_most)