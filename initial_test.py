import csv
import json
import math
import numpy as np
import dimod
from dwave.system import LeapHybridSampler
from collections import Counter


###############################
# HELPER FUNCTIONS            #
###############################

def sanitize_tweet(tweet: str):
    tweet = tweet.replace('.', '').replace(',', '').replace('@', '')
    tweet = tweet.lower()
    return tweet.strip()


def tweet_to_vector(tweet, vocab):
    words = set(tweet.lower().split())
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)


def w_bit_name(j, bit):
    """
    Return the variable name for the bit-th bit of weight w_j.
    Example: w_3_bit1
    """
    return f"w_{j}_bit{bit}"


def decode_weight(sample, j, num_bits=2):
    """
    Decode the integer weight value from the bits in the 'sample'.
    w_j = sum_{bit=0..(num_bits-1)} (2^bit * w_{j}_bit{bit}).
    """
    val = 0
    for bit in range(num_bits):
        val += (2 ** bit) * sample[w_bit_name(j, bit)]
    return val


###############################
# DATA LOADING                #
###############################

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

word_counts = Counter()
for tweet in tweets:
    words = tweet.split()  # already sanitized
    word_counts.update(words)

vocab = [word for word, cnt in word_counts.most_common(max_vocab_size)]
vocab.sort()  # sort for consistency
num_features = len(vocab)
print("Vocabulary size:", num_features)

###############################
# FEATURE MATRIX              #
###############################

X = np.array([tweet_to_vector(tw, vocab) for tw in tweets], dtype=int)
num_samples = len(X)

###############################
# BUILD QUBO (WITH 2-BIT WEIGHTS)
###############################

lambda_reg = 0.01
num_bits_per_weight = 2


def build_qubo(X, binary_labels, num_features, lambda_reg, num_bits=3):
    """
    Build a QUBO for squared error loss with integer weights:
        w_j in {0,1,...,2^num_bits - 1}, represented via 'num_bits' binary variables each.

    The classifier is:
        output = sum_j(W_j * x[i][j]) + b,
    where W_j = sum_{bit} (2^bit * w_{j}_bit{bit}).

    L_i = (y_i - (sum_j W_j*x_i[j] + b))^2
    """
    Q = {}

    def add_to_Q(u, v, coeff):
        coeff = float(coeff)
        if (u, v) in Q:
            Q[(u, v)] += coeff
        elif (v, u) in Q:
            Q[(v, u)] += coeff
        else:
            Q[(u, v)] = coeff

    n_samples = len(binary_labels)

    for i in range(n_samples):
        y = binary_labels[i]
        xi = X[i]

        # -------------------------------------
        # 1) Linear term from -2*y*( sum_j W_j*xi[j] + b )
        # -------------------------------------
        for j in range(num_features):
            # Each weight W_j = sum_{bit} 2^bit * w_{j}_bit{bit}
            for bit_j in range(num_bits):
                var_j_bit = w_bit_name(j, bit_j)
                coeff = -2.0 * y * xi[j] * (2 ** bit_j)
                add_to_Q(var_j_bit, var_j_bit, coeff)  # This is a linear term w_{j}_bit{bit}

        # bias linear term: -2*y*b
        add_to_Q('b', 'b', -2.0 * y)

        # -------------------------------------
        # 2) Quadratic term from ( sum_j W_j*xi[j] + b )^2
        #    = sum_{j,k} W_j * W_k * xi[j]*xi[k] + 2*b*(sum_j W_j*xi[j]) + b^2
        # -------------------------------------

        # 2a) Weight-weight cross terms: sum_{j,k} (W_j* W_k * x_i[j]* x_i[k])
        # Expanding W_j * W_k means sum_{bits_j} sum_{bits_k} (2^bit_j)*(2^bit_k)* w_{j}_bit{bit_j} * w_{k}_bit{bit_k}
        for j in range(num_features):
            for k in range(j, num_features):
                for bit_j in range(num_bits):
                    for bit_k in range(num_bits):
                        var_j_bit = w_bit_name(j, bit_j)
                        var_k_bit = w_bit_name(k, bit_k)
                        coeff = (xi[j] * xi[k]) * ((2 ** bit_j) * (2 ** bit_k))
                        add_to_Q(var_j_bit, var_k_bit, coeff)

        # 2b) Cross term with bias: 2*b* sum_j(W_j*x_i[j])
        # For each j, each bit: coefficient = 2*x_i[j]*2^bit_j
        for j in range(num_features):
            for bit_j in range(num_bits):
                var_j_bit = w_bit_name(j, bit_j)
                coeff = 2.0 * xi[j] * (2 ** bit_j)
                add_to_Q(var_j_bit, 'b', coeff)

        # 2c) Bias squared (b^2 = b for a binary bit)
        # For each sample, we add +1 to b,b
        add_to_Q('b', 'b', 1.0)

    # -------------------------------------
    # 3) Regularization: lambda_reg * sum_j( W_j ),
    #    or you can distribute it across each bit to discourage turning bits on.
    # -------------------------------------
    for j in range(num_features):
        for bit_j in range(num_bits):
            var_j_bit = w_bit_name(j, bit_j)
            # A simple approach: just add lambda_reg to the diagonal of each bit
            add_to_Q(var_j_bit, var_j_bit, lambda_reg)

    return Q


###############################
# ONE-VS-REST MULTICLASS      #
###############################

sentiment_classes = ['Positive', 'Neutral', 'Negative']
classifiers = {}

sampler = LeapHybridSampler()

for sentiment in sentiment_classes:
    print("\nTraining classifier for:", sentiment)
    binary_labels = [1 if lab == sentiment else 0 for lab in labels]

    Q = build_qubo(X, binary_labels, num_features, lambda_reg, num_bits=num_bits_per_weight)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

    print(f"QUBO for {sentiment} has {len(Q)} terms. Solving...")
    # Increase time_limit if needed
    sampleset = sampler.sample(bqm, time_limit=15)
    if len(sampleset) == 0:
        raise ValueError(f"No samples returned for classifier {sentiment}")

    best_sample = sampleset.first.sample

    # Decode the multi-bit weights
    w_array = np.array([decode_weight(best_sample, j, num_bits=num_bits_per_weight)
                        for j in range(num_features)])
    b = best_sample['b']

    classifiers[sentiment] = (w_array, b)
    print(f"Learned bias for {sentiment}: {b}")
    print(f"Sample of learned weights (first 10): {w_array[:10]}")


###############################
# PREDICTION                  #
###############################

def predict_tweet(tweet, vocab, classifiers):
    x = tweet_to_vector(sanitize_tweet(tweet), vocab)
    scores = {}
    for sentiment, (w, b) in classifiers.items():
        scores[sentiment] = np.dot(w, x) + b
    predicted = max(scores, key=scores.get)
    return predicted, scores


###############################
# EVALUATION                  #
###############################

max_testing_data = 500
testing_tweets = []
testing_labels = []
with open("twitter_validation.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        sentiment = row[2].strip()
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            continue
        tw = sanitize_tweet(row[3])
        testing_tweets.append(tw)
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
print("Validation accuracy: {:.2f}%".format(accuracy * 100))
