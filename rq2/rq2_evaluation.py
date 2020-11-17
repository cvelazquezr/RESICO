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


RESOURCES_PATH = "data/resources"
RESULTS_PATH = "data/results"

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
        # if i % 100 == 0:
        #     print(f"Analysing vector {i + 1} ...")

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
            y_test.append(i)
            y_pred.append(max_index)

    y_pred = np.array(y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")

    f1_score = get_f1_score(precision, recall)
    print("F1-Score: ", f1_score)
    print(f"Assertion: {assertions} out of {len(vectors)}")

    return f1_score

# Returns those FQNs with an ocurrence higher than threshold
def filter_data(lines: list, threshold: int):
    counter_data_dict = dict()

    for line in lines:
        fqn = line[-1]
        
        if fqn in counter_data_dict:
            counter_data_dict[fqn] += 1
        else:
            counter_data_dict[fqn] = 1
    
    fqns_filtered = [fqn for fqn, presence in counter_data_dict.items() if presence >= threshold]
    lines_filtered = list()

    for line in lines:
        fqn = line[-1]

        if fqn in fqns_filtered:
            lines_filtered.append(line)
    
    return lines_filtered


if __name__ == "__main__":
    print("Loading word2vec model ...")
    configuration = "only_class_neighbors"
    model_w2vec = Word2Vec.load(f"{RESULTS_PATH}/models/text/word2vec_github_resico_{configuration}model")

    print("Loading libraries information ...")
    libs_mapping = pickle.load(open(f"{RESULTS_PATH}/models/libraries_information/libs_mapping_github_resico_{configuration}.pickle", "rb"))

    print("Loading the trained models ...")
    bnb = pickle.load(open(f"{RESULTS_PATH}/models/classifiers/BNB_github_resico_{configuration}.model", "rb"))
    gnb = pickle.load(open(f"{RESULTS_PATH}/models/classifiers/GNB_github_resico_{configuration}.model", "rb"))
    knn = pickle.load(open(f"{RESULTS_PATH}/models/classifiers/KNN_github_resico_{configuration}.model", "rb"))

    models = [
        ("BNB", bnb),
        ("GNB", gnb),
        ("KNN", knn),
    ]

    print("Loading the file in analysis ...")
    VARIANT = f"{RESULTS_PATH}/datasets/resico_{configuration}"
    DATASET = "COSTER-SO-Ext"
    lines = list()
    
    with open(f"{VARIANT}/{DATASET}.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                line = line.split(",")

                lines.append(line)

    # splits = 5
    # lines_filtered = filter_data(lines, splits)

    print("Convert the lines into vectors ...")
    vectors = list()
    truth_fqns = list()

    no_matches = 0
    no_matches_lines = pickle.load(open(f"{RESULTS_PATH}/oov_values_coster_ext.pickle", "rb"))

    for i, line in enumerate(lines):
        if not i in no_matches_lines:
            list_words = line[:-1]
            if list_words[-1].count("|"):
                list_words = list_words[:-1] + list_words[-1].split("|")

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
                # no_matches_lines.append(i)

    oov = no_matches / len(lines) # Discount from the final result this percentage
    print("No Matches: ", no_matches, oov)
    # pickle.dump(no_matches_lines, open(f"{RESULTS_PATH}/oov_values_coster_ext.pickle", "wb"))
    
    print("Running benchmark ...")
    top_k = [1, 3, 5]

    for name, model in models:
        print(f"Model {name} ...")

        for k in top_k:
            print(f"Results for K == {k}")
            f1_score = benchmark(model, truth_fqns, vectors, libs_mapping, k)
            print(f"Final F1-Score: {f1_score}")
            print()
        print()
