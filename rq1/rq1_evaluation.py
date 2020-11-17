import pickle
import time
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from random import randint

from sklearn.metrics import auc, roc_curve, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold

from collections import Counter


RESOURCES_PATH = "../../data/resources"
RESULTS_PATH = "../../data/results"

def get_truth_fqn(mapping, index):
    for k, v in mapping.items():
        if v == index:
            return k
    return ""


def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def benchmark(classifier, ground_truth, vectors, mapping, k_th):
    precisions = list()
    recalls = list()

    y_test = list()
    y_pred = list()
    assertions = 0

    for i, vector in enumerate(vectors):
        if i % 100 == 0:
            print(f"Analysing vector {i + 1} ...")

        probs = classifier.predict_proba([vector])
        probs_arr = [round(float(value), 2) for value in list(probs[0])]
        
        max_index = probs_arr.index(max(probs_arr))
        true_value = ground_truth[i]

        indexes = list(np.argpartition(probs[0], -k_th)[-k_th:])
        values_indexes = [get_truth_fqn(mapping, index) for index in indexes]

        if true_value in values_indexes:
            assertions += 1
            y_test.append(max_index)
            y_pred.append(max_index)
        else:
            y_test.append(randint(1, len(ground_truth)))
            y_pred.append(randint(1, len(ground_truth)))

    y_pred = np.array(y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")

    print("F1-Score: ", get_f1_score(precision, recall))
    print(f"Assertion: {assertions} out of {len(vectors)}")


def matthews_coeff_classes(classifier, X, y):
    mean_coeff = list()

    for k, (train, test) in enumerate(StratifiedKFold(n_splits=10).split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        print(f"Fold # {k + 1}")

        print("Training ...")
        classifier.fit(X_train, y_train)
        print("Done!")

        y_pred = classifier.predict(X[test])
        mean_coeff.append(matthews_corrcoef(y_test, y_pred))
    
    return mean_coeff


if __name__ == "__main__":
    print("Loading word2vec model ...")
    model_w2vec = Word2Vec.load(f"{RESULTS_PATH}/models/text/word2vec_github_coster_ml.model")

    print("Loading libraries information ...")
    libs_mapping = pickle.load(open(f"{RESULTS_PATH}/models/libraries_information/libs_mapping_github_coster_ml.pickle", "rb"))

    print("Loading the trained models ...")
    bnb = pickle.load(open(f"{RESULTS_PATH}/models/classifiers/BNB_github_coster_ml.model", "rb"))
    gnb = pickle.load(open(f"{RESULTS_PATH}/models/classifiers/GNB_github_coster_ml.model", "rb"))
    knn = pickle.load(open(f"{RESULTS_PATH}/models/classifiers/KNN_github_coster_ml.model", "rb"))

    models = [
        ("BNB", bnb),
        ("GNB", gnb),
        ("KNN", knn),
    ]

    print("Loading the file in analysis ...")
    lines = list()
    
    with open(f"{RESULTS_PATH}/datasets/coster_neighbors/COSTER-SO.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                line = line.split(",")

                lines.append(line)

    print("Convert the lines into vectors ...")
    vectors = list()
    truth_fqns = list()

    no_matches = 0
    for line in lines:
        list_words = [line[0]] + line[1].split("|")

        vector = list()
        for word in list_words:
            word = word.lower()
            if word in model_w2vec.wv.vocab:
                vector.append(model_w2vec.wv[word])
        
        if len(vector):
            vector = sum(np.array(vector)) / len(vector)
            vectors.append(vector)
            truth_fqns.append(line[-1])
        else:
            no_matches += 1

    print("No Matches: ", no_matches)
    print("Running benchmark ...")

    top_k = [1, 3, 5]

    for name, model in models:
        print(f"Model {name} ...")

        for k in top_k:
            print(f"Results for K == {k}")
            benchmark(model, truth_fqns, vectors, libs_mapping, k)
            print()
        print()
