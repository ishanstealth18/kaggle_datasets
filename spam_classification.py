import pandas as pd

from collections import defaultdict

from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
df["label_int"] = (df["label"] == "spam").astype(int)

# Count word frequencies per class
spam_words = defaultdict(int)
ham_words = defaultdict(int)
for _, row in df.iterrows():
    for word in row["text"].lower().split():
        if row["label"] == "spam":
            spam_words[word] += 1
        else:
            ham_words[word] += 1

#print(df.info())
print(df.head())
X = df.drop(columns=['label'], axis=1).values
y = df['label_int'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Calculate P(spam) and P(ham)
#print(df['label_int'].value_counts())
# spam value count  = 747,total entries = 5572
# ham value count =4825, total entries = 5572
p_spam = 13.40
p_ham = 86.59

# Build a vocabulary from the training set and calculate word conditional probabilities.
# 1. Create vocab using word count
from collections import Counter
import re
import math


class NaiveBayesClassifier:
    def __init__(self):
        self.vocab= set()
        self.word_counts = {0: Counter(), 1: Counter()}
        self.total_words_in_class = {0: 0, 1: 0}
        self.class_priors = {0: 0.0, 1: 0.0}

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def train(self, training_set):
        print(training_set)
        total_data = len(training_set)
        class_counts = Counter()

        for text, label in training_set:
            print('Text: ',text)
            print('Label: ', label)

            class_counts[label] += 1
            tokens = self.tokenize(text)

            for token in tokens:
                self.vocab.add(token)
                self.word_counts[label][token] += 1
                self.total_words_in_class[label] += 1

        for label in class_counts:
            self.class_priors[label] = class_counts[label] / total_data

        print(self.word_counts)
        print(self.total_words_in_class)

    def predict(self, text):
        tokens = self.tokenize(text)
        vocab_size = len(self.vocab)

        # Use log probabilities to prevent numerical underflow
        class_scores = {}

        for label in self.class_priors:
            # Start with log of the class prior probability: log(P(Class))
            log_prob = math.log(self.class_priors[label])

            # Add log conditional probabilities for each word: log(P(Word | Class))
            for token in tokens:
                # Skip words completely if you choose, or handle as unknown
                # Applying Laplace (Add-1) Smoothing based on vocabulary size
                word_count = self.word_counts[label][token]
                total_class_words = self.total_words_in_class[label]

                smoothed_word_prob = (word_count + 1) / (total_class_words + vocab_size)
                log_prob += math.log(smoothed_word_prob)

            class_scores[label] = log_prob
        print("Class Scores: ", class_scores)

        # The predicted class is the one with the maximum score
        predicted_class = max(class_scores, key=class_scores.get)

        return predicted_class, class_scores

nb_model = NaiveBayesClassifier()
nb_model.train(X_train)

for sentence in X_test:
    prediction, scores = nb_model.predict(sentence[0])
    label_map = {0: "Ham (Normal)", 1: "Spam"}

    print(f"Sentence: \"{sentence}\"")
    print(f"-> Log Scores: Ham={scores[0]:.4f}, Spam={scores[1]:.4f}")
    print(f"-> Predicted Class: **{label_map[prediction]}**\n")