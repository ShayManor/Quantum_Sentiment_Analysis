import csv
import json
import math
import numpy as np
import dimod
from dwave.system import LeapHybridSampler
from collections import Counter

def sanitize_tweet(tweet: str):
    tweet = tweet.replace('.','').replace(',','').replace('@','')
    tweet = tweet.lower()
    return tweet.strip()

max_data = 500
max_vocab_size = 100

tweets = []
labels = []
with open("twitter_training.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        sentiment = row[2].strip()  # Expected to be 'Positive', 'Negative', or 'Neutral'
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            continue
        tweets.append(sanitize_tweet(row[3]))
        labels.append(sentiment)
        if len(tweets) >= max_data:
            break
print("Loaded {} tweets.".format(len(tweets)))

###############################
# VOCABULARY CONSTRUCTION     #
###############################

# Build vocabulary from tweets using a frequency count,
# and select the top max_vocab_size words.
word_counts = Counter()
for tweet in tweets:
    words = tweet.lower().split()
    word_counts.update(words)
vocab = [word for word, cnt in word_counts.most_common(max_vocab_size)]
vocab.sort()  # sort for consistency
num_features = len(vocab)
print("Vocabulary size: {}".format(num_features))


###############################
# FEATURE EXTRACTION          #
###############################

def tweet_to_vector(tweet, vocab):
    words = set(tweet.lower().split())
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)


# Create a feature matrix X of shape (num_samples, num_features)
X = np.array([tweet_to_vector(tweet, vocab) for tweet in tweets])
num_samples = len(tweets)

###############################
# QUBO BUILDING FUNCTION      #
###############################

# We use a squared error loss for a binary perceptron.
# For a given sample i with binary label y (1 if tweet belongs to the class, 0 otherwise)
# and feature vector xi, the loss is:
#    L_i = (y - (sum_j w_j * xi[j] + b))^2.
# Expanding gives a quadratic function in the binary decision variables (w_j and b).
# We add a small regularization (lambda_reg) on the weights.
lambda_reg = 0.1


def build_qubo(X, binary_labels, num_features, lambda_reg):
    # QUBO is a dictionary mapping variable pairs to coefficients.
    Q = {}

    # Helper function to add coefficients:
    def add_to_Q(u, v, coeff):
        coeff = float(coeff)
        if (u, v) in Q:
            Q[(u, v)] += coeff
        elif (v, u) in Q:
            Q[(v, u)] += coeff
        else:
            Q[(u, v)] = coeff

    num_samples = len(binary_labels)
    for i in range(num_samples):
        y = binary_labels[i]
        xi = X[i]  # feature vector for sample i
        # Loss L_i = (y - (sum_j w_j*xi[j] + b))^2.
        # Drop constant term y^2.
        # Linear term: -2*y*(sum_j w_j*xi[j] + b)
        for j in range(num_features):
            var = 'w_{}'.format(j)
            coeff = -2 * y * xi[j]
            add_to_Q(var, var, coeff)  # add on diagonal
        add_to_Q('b', 'b', -2 * y)
        # Quadratic term: (sum_j w_j*xi[j] + b)^2
        # Weight-weight interactions:
        for j in range(num_features):
            for k in range(j, num_features):
                var_j = 'w_{}'.format(j)
                var_k = 'w_{}'.format(k)
                coeff = xi[j] * xi[k]
                add_to_Q(var_j, var_k, coeff)
        # Cross term: 2*b*(sum_j w_j*xi[j])
        for j in range(num_features):
            var = 'w_{}'.format(j)
            coeff = 2 * xi[j]
            add_to_Q(var, 'b', coeff)
        # Bias squared: since b is binary, b^2 = b.
        add_to_Q('b', 'b', 1)
    # Add regularization term for each weight.
    for j in range(num_features):
        add_to_Q('w_{}'.format(j), 'w_{}'.format(j), lambda_reg)
    return Q


#####################################
# ONE-VS-REST MULTICLASS TRAINING    #
#####################################

# We train a separate binary classifier for each sentiment class.
sentiment_classes = ['Positive', 'Neutral', 'Negative']
# For each class, create binary labels (1 if tweet is in that class, 0 otherwise)
classifiers = {}  # (weights, bias)

sampler = LeapHybridSampler()

for sentiment in sentiment_classes:
    print("Training classifier for:", sentiment)
    binary_labels = [1 if lab == sentiment else 0 for lab in labels]
    Q = build_qubo(X, binary_labels, num_features, lambda_reg)
    print("QUBO for {} has {} terms.".format(sentiment, len(Q)))
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    print(Q)
    # Increase time_limit if needed (here 10 seconds per classifier)
    sampleset = sampler.sample(bqm, time_limit=15)
    if len(sampleset) == 0:
        raise ValueError("No samples returned for classifier {}".format(sentiment))
    best_sample = sampleset.first.sample
    # Extract learned weights and bias
    w = np.array([best_sample['w_{}'.format(j)] for j in range(num_features)])
    b = best_sample['b']
    classifiers[sentiment] = (w, b)
    print("Learned bias for {}: {}".format(sentiment, b))


#####################################
# PREDICTION FUNCTION               #
#####################################

def predict_tweet(tweet, vocab, classifiers):
    # Convert tweet to feature vector.
    x = tweet_to_vector(tweet, vocab)
    # For each classifier, compute the output = dot(w, x) + b.
    scores = {}
    for sentiment, (w, b) in classifiers.items():
        scores[sentiment] = np.dot(w, x) + b
    # Return the sentiment with the highest score.
    predicted = max(scores, key=scores.get)
    return predicted, scores


#####################################
# EVALUATE THE MODEL                #
#####################################
max_testing_data = 500
testing_tweets = []
testing_labels = []
with open("twitter_validation.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        sentiment = row[2].strip()  # Expected to be 'Positive', 'Negative', or 'Neutral'
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            continue
        testing_tweets.append(row[3])
        testing_labels.append(sentiment)
        if len(testing_tweets) >= max_testing_data:
            break
num_testing_samples = len(testing_tweets)
correct = 0
for i in range(num_testing_samples):
    tweet = testing_tweets[i]
    true_label = testing_labels[i]
    predicted, scores = predict_tweet(tweet, vocab, classifiers)
    if predicted == true_label:
        correct += 1
    print("Tweet:", tweet)
    print("True label:", true_label, "Predicted:", predicted, "Scores:", scores)
    print("-----------")

accuracy = correct / num_testing_samples
print("Training accuracy: {:.2f}%".format(accuracy * 100))
