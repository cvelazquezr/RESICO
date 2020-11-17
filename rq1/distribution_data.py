from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns


def get_lines(path: str):
    lines = list()

    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                lines.append(line)
    return lines


if __name__ == "__main__":
    RESULTS_PATH = "../../data/results"

    print("Loading the files in the analysis ...")
    fqns_coster_so = get_lines(f"{RESULTS_PATH}/datasets/COSTER-SO-fqns.txt")
    fqns_coster_so_ext = get_lines(f"{RESULTS_PATH}/datasets/COSTER-SO-Ext-fqns.txt")

    counter_so = dict(Counter(fqns_coster_so))
    counter_so_ext = dict(Counter(fqns_coster_so_ext))

    print(counter_so)
    print(len(counter_so_ext))
