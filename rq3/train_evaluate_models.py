import pickle
import time
import numpy as np
import pandas as pd

from pathlib import Path
from numpy import interp
from gensim.models import Word2Vec

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold

from collections import Counter


RESOURCES_PATH = "../../data/resources"
RESULTS_PATH = "../../data/results"

def read_file(file_handler):
    lines = list()

    while True:
        line = file_handler.readline()
        if not line:
            break
        else:
            line = line.strip()
            lines.append(line)
    return lines


def obtain_data(lines: list):
    sentences = list()

    for line in lines:
        line_splitted = line.split(",")
        line_processed = line_splitted[:-1]
        line_processed_lower = list(map(lambda word: word.lower(), line_processed))

        sentences.append(line_processed_lower)
    return sentences


# Returns those FQNs with an ocurrence higher than threshold
def filter_data(lines: list, threshold: int):
    counter_data_dict = dict()

    for line in lines:
        line_splitted = line.split(",")
        fqn = line_splitted[-1]
        
        if fqn in counter_data_dict:
            counter_data_dict[fqn] += 1
        else:
            counter_data_dict[fqn] = 1
    
    fqns_filtered = [fqn for fqn, presence in counter_data_dict.items() if presence >= threshold]
    lines_filtered = list()

    for line in lines:
        line_splitted = line.split(",")
        fqn = line_splitted[-1]

        if fqn in fqns_filtered:
            lines_filtered.append(line)
    
    return lines_filtered


def get_vectors(model, lines):
    inputs = list()
    output = list()

    classes_numbered = dict()
    k = 0

    for line in lines:
        line_splitted = line.split(",")
        corpus = line_splitted[:-1]
        clazz = line_splitted[-1]

        vectors = [model.wv[word.lower()] for word in corpus]
        line_vector = sum(vectors) / len(vectors)
        inputs.append(line_vector)

        if clazz in classes_numbered:
            output.append(classes_numbered[clazz])
        else:
            classes_numbered[clazz] = k
            output.append(k)
            k += 1

    return np.array(inputs), np.array(output), classes_numbered


def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def benchmark(classifier, X, y, k_th, splits):
    scores = list()

    precisions = list()
    recalls = list()
    assertions = 0

    for train, test in StratifiedKFold(n_splits=splits).split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        classifier.fit(X_train, y_train)
        y_prediction = classifier.predict(X_test)

        y_pred = list()

        for i, x_test in enumerate(X_test):
            probs = classifier.predict_proba(np.array([x_test]))
            probs_arr = [round(float(value), 2) for value in list(probs[0])]
            
            max_index = probs_arr.index(max(probs_arr))
            true_value = y_test[i]

            if len(probs[0] < k_th): # The number of lower than the k_th
                k_th = len(probs[0])

            indexes = list(np.argpartition(probs[0], -k_th)[-k_th:])

            if true_value in indexes:
                assertions += 1
                y_pred.append(true_value)
            else:
                y_pred.append(max_index)

        y_pred = np.array(y_pred)
        precisions.append(precision_score(y_test, y_pred, average="micro"))
        recalls.append(recall_score(y_test, y_pred, average="micro"))

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)

    print(f"Assertions: {assertions} of {len(X)}")

    return get_f1_score(precision, recall)

# Returns those FQNs with an ocurrence higher than threshold
def filter_data(lines: list, threshold: int):
    counter_data_dict = dict()

    for line in lines:
        line_splitted = line.split(",")
        fqn = line_splitted[-1]
        
        if fqn in counter_data_dict:
            counter_data_dict[fqn] += 1
        else:
            counter_data_dict[fqn] = 1
    
    fqns_filtered = [fqn for fqn, presence in counter_data_dict.items() if presence >= threshold]
    lines_filtered = list()

    for line in lines:
        line_splitted = line.split(",")
        fqn = line_splitted[-1]

        if fqn in fqns_filtered:
            lines_filtered.append(line)
    
    return lines_filtered


if __name__ == "__main__":
    # 6 models are going to be in comparison here
    # The first three models are already trained and their results were already computed
    # The last three models are going to be trained/evaluated here with the classical approach of cross-validation

    print("Reading file ...")
    configuration = "only_class_neighbors"
    DATASET = "COSTER-SO"

    data = open(f'{RESULTS_PATH}/datasets/resico_{configuration}/{DATASET}.txt')
    lines = read_file(data)
    data.close()

    splits = 5
    lines_filtered = filter_data(lines, splits)
    print(len(lines_filtered), len(lines))

    print("Getting the corpus of the data to vectorize it ...")
    sentences2vec = obtain_data(lines_filtered)

    print("Vectorizing the corpus of the data ...")
    model_w2vec = Word2Vec(sentences2vec, min_count=1, sg=True)

    print("Obtaining attributes and output for the data ...")
    X, y, libs_mapping = get_vectors(model_w2vec, lines_filtered)

    print(len(libs_mapping))

    models_test = [
        # ("GNB", GaussianNB()),
        ("BNB", BernoulliNB()),
        # ("KNN", KNeighborsClassifier(n_jobs=8)),
    ]

    print("Metrics for the models:")

    for name, classifier in models_test:
        print(name)
        # print("Saving trained models ...")
        # classifier.fit(X, y)

        # model_filename = f"{RESULTS_PATH}/models/classifiers/{name}.model"
        # pickle.dump(classifier, open(model_filename, "wb"))

        print(f"Results for the classifier {name}")
        tops = [1, 3, 5]

        for top in tops:
            print(f"Results for K == {top}")

            f1_score = benchmark(classifier, X, y, top, splits)
            print(f"Final F1-Score: {f1_score}")
            print()
        print()
