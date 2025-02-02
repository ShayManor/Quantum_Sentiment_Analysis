import csv
import json
import math
import numpy as np
import dimod
from dwave.system import LeapHybridSampler
from sympy.codegen.ast import continue_

tweets = []
labels = []
num = 100
idx = 0
with open("twitter_training.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if idx > num:
            break
        if row[2] == 'Positive':
            labels.append(1)
        elif row[2] == 'Negative':
            labels.append(0)
        else:
            continue
        tweets.append(row[3])
        idx += 1
# tweets = [
#     "I love this product",  # positive sentiment
#     "This is the worst experience",  # negative sentiment
#     "Absolutely fantastic service",  # positive sentiment
#     "I am very disappointed"  # negative sentiment
# ]
# # Define labels: 1 for positive, 0 for negative
# labels = [1, 0, 1, 0]

# ------------------------------
# Simple Bag-of-Words Preprocessing
# ------------------------------
# Build a vocabulary from the tweets (in practice, use a proper tokenizer)
vocab = set()
for tweet in tweets:
    for word in tweet.lower().split():
        vocab.add(word)
vocab = list(vocab)
vocab.sort()
num_features = len(vocab)


# Convert tweets to feature vectors: binary presence/absence for each word
def tweet_to_vector(tweet, vocab):
    words = tweet.lower().split()
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)


X = np.array([tweet_to_vector(tweet, vocab) for tweet in tweets])
# Now, X is of shape (num_samples, num_features)
num_samples = len(tweets)

Q = {}

lambda_reg = 0.1

variables = ['w_{}'.format(j) for j in range(num_features)] + ['b']


# Helper: add coefficient to QUBO entry for (u,v)
def add_to_Q(u, v, coeff):
    coeff = float(coeff)
    if (u, v) in Q:
        Q[(u, v)] += coeff
    elif (v, u) in Q:
        Q[(v, u)] += coeff
    else:
        Q[(u, v)] = coeff


# Build the QUBO by summing the loss for each training sample
for i in range(num_samples):
    y = labels[i]
    xi = X[i]  # feature vector for sample i
    # Compute contribution from the linear term (prediction = sum_j w_j*xi[j] + b)
    # Loss L_i = y - 2*y*(sum_j (w_j*xi[j]) + b) + (sum_j (w_j*xi[j]) + b)^2
    # The constant term y can be dropped since it does not affect optimization.
    #
    # Add linear terms from -2*y*(sum_j (w_j*xi[j]) + b)
    for j in range(num_features):
        var = 'w_{}'.format(j)
        coeff = -2 * y * xi[j]
        add_to_Q(var, var, coeff)  # using diagonal for linear term
    add_to_Q('b', 'b', -2 * y)

    # Add quadratic terms from (sum_j (w_j*xi[j]) + b)^2
    # First, the weight-weight interaction:
    for j in range(num_features):
        for k in range(j, num_features):
            var_j = 'w_{}'.format(j)
            var_k = 'w_{}'.format(k)
            coeff = xi[j] * xi[k]
            # For j==k, this is a linear term; for j<k, it is a quadratic term.
            add_to_Q(var_j, var_k, coeff)
    # Next, the cross term with bias: 2*b*(sum_j (w_j*xi[j]))
    for j in range(num_features):
        var = 'w_{}'.format(j)
        coeff = 2 * xi[j]
        add_to_Q(var, 'b', coeff)
    # Finally, bias squared: since b is binary, b^2 = b.
    add_to_Q('b', 'b', 1)

# Optionally, add a regularization term to keep the weight vector small.
for j in range(num_features):
    add_to_Q('w_{}'.format(j), 'w_{}'.format(j), lambda_reg)

print(Q)
print("Submitting QUBO to D-Wave...")
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm, time_limit=10)
best_sample = sampleset.first.sample

weights = np.array([best_sample['w_{}'.format(j)] for j in range(num_features)])
bias = best_sample['b']

print("Learned binary weights:")
for j, w in enumerate(weights):
    print("  {}: {}".format(vocab[j], w))
print("Learned bias:", bias)


# Define a simple prediction function: prediction = sum_j (w_j * x_j) + bias
# Here we simply threshold at 0.5 to decide positive vs negative.
def predict(x, weights, bias):
    val = np.dot(weights, x) + bias
    return 1 if val >= 0.5 else 0


# Evaluate predictions on the training set
predictions = []
for i in range(num_samples):
    pred = predict(X[i], weights, bias)
    predictions.append(pred)
    print("Tweet:", tweets[i])
    print("True label:", labels[i], "Prediction:", pred)
    print("-----------")
