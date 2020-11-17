
def get_no_neighbors(lines: list):
    modified_lines = list()

    for line in lines:
        modified_lines.append(line[:-2] + [line[-1]])

    return modified_lines


def get_only_class_neighbors(lines: list):
    modified_lines = list()

    for i, line in enumerate(lines):
        context = line[-2]

        classes_context = list()
        if context.count("|"):
            for i, token in enumerate(context.split("|")):
                if token[0].isupper():
                    classes_context.append(token)

            classes_context = "|".join(classes_context)
            modified_lines.append(line[:-2] + [classes_context, line[-1]])
        else:
            modified_lines.append(line)

    return modified_lines


def get_only_methods_neighbors(lines: list):
    modified_lines = list()

    for line in lines:
        context = line[-2]
        classes_context = list()

        if context.count("|"):
            for i, token in enumerate(context.split("|")):
                if token[0].islower():
                    classes_context.append(token)

            classes_context = "|".join(classes_context)

            modified_lines.append(line[:-2] + [classes_context, line[-1]])
        else:
            modified_lines.append(line)

    return modified_lines


def write_results(path_file: str, lines: list):
    with open(path_file, "w") as f:
        for line in lines:
            f.write(",".join(line) + "\n")


if __name__ == "__main__":
    DATASETS_FOLDER = "../data/results/datasets"
    ORIGINAL_FOLDER = f"{DATASETS_FOLDER}/resico_neighbors"
    SOURCE_FILE = "COSTER-SO"
    PATH = f"{ORIGINAL_FOLDER}/{SOURCE_FILE}.txt"

    OUTPUT_FOLDER = f"{DATASETS_FOLDER}/resico_only_methods_neighbors"
    OUTPUT_PATH = f"{OUTPUT_FOLDER}/{SOURCE_FILE}.txt"

    lines = list()
    with open(PATH) as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                line = line.split(",")
                lines.append(line)
    
    no_neighbors = get_no_neighbors(lines)
    only_class_neighbors = get_only_class_neighbors(lines)
    only_methods_neighbors = get_only_methods_neighbors(lines)

    # write_results(OUTPUT_PATH, only_methods_neighbors)
