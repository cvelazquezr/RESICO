import pickle
import numpy as np

from pathlib import Path
from numpy import interp
from gensim.models import Word2Vec

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


RESOURCES_PATH = "data/resources"
RESULTS_PATH = "data/results"

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
        if line:
            line_splitted = line.split(",")
            line_processed = line_splitted[:-1]

            if len(line_processed) > 1:
                line_processed = [line_processed[0], line_processed[1]] + line_processed[2].split("|")
                line_processed_lower = list(map(lambda word: word.lower(), line_processed))

                sentences.append(line_processed_lower)
    return sentences


# Returns those FQNs with an ocurrence higher than threshold
def filter_data(lines: list, threshold: int):
    counter_data_dict = dict()

    for line in lines:
        line_splitted = line.split(",")
        fqn = line_splitted[-1]
        
        if len(line_splitted[:-1]) > 1:
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
        
        if len(corpus) > 1:
            corpus = [corpus[0], corpus[1]] + corpus[2].split("|")
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


if __name__ == "__main__":
    print("Reading file ...")
    configuration = "neighbors"

    data = open(f'{RESOURCES_PATH}/apiElements_RESICO_{configuration}.txt')
    lines = read_file(data)
    data.close()

    print("Filtering the data with fqns higher than a threshold ...")
    threshold = 10
    lines_filtered = filter_data(lines, threshold)

    print(len(lines_filtered), len(lines))

    print("Getting the corpus of the data to vectorize it ...")
    sentences2vec = obtain_data(lines_filtered)

    print("Vectorizing the corpus of the data ...")
    model_w2vec = Word2Vec(sentences2vec, min_count=1, sg=True)

    print("Obtaining attributes and output for the data ...")
    X, y, libs_mapping = get_vectors(model_w2vec, lines_filtered)

    print(len(libs_mapping))
    print(X.shape, y.shape)

    print("Saving data ...")
    Path(f"{RESULTS_PATH}/models/text/").mkdir(parents=True, exist_ok=True)
    model_w2vec.save(f"{RESULTS_PATH}/models/text/word2vec_github_resico_{configuration}.model")

    Path(f"{RESULTS_PATH}/models/libraries_information/").mkdir(parents=True, exist_ok=True)
    pickle.dump(libs_mapping, open(f"{RESULTS_PATH}/models/libraries_information/libs_mapping_github_resico_{configuration}.pickle", "wb"))

    Path(f"{RESULTS_PATH}/input/").mkdir(parents=True, exist_ok=True)
    pickle.dump(X, open(f"{RESULTS_PATH}/input/X_{configuration}.pickle", "wb"))
    pickle.dump(y, open(f"{RESULTS_PATH}/input/y_{configuration}.pickle", "wb"))

    print("Getting the trained vectors in Word2vec ...")
    X = pickle.load(open(f"{RESULTS_PATH}/input/X_neighbors.pickle", "rb"))
    y = pickle.load(open(f"{RESULTS_PATH}/input/y_neighbors.pickle", "rb"))

    models_test = [
        ("GNB", GaussianNB()),
        ("BNB", BernoulliNB()),
        ("KNN", KNeighborsClassifier(n_jobs=8)),
    ]

    print("Metrics for the models:")
    Path(f"{RESULTS_PATH}/models/classifiers/").mkdir(parents=True, exist_ok=True)

    for name, classifier in models_test:
        print(name)
        print("Saving trained models ...")
        classifier.fit(X, y)

        model_filename = f"{RESULTS_PATH}/models/classifiers/{name}_github_resico_{configuration}.model"
        pickle.dump(classifier, open(model_filename, "wb"), protocol=4)
