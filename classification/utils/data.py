import pandas as pd

def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())

    texts, labels = [], []
    for (i, line) in enumerate(lines[1:]):
        line = line.split("\t")
        text_a = line[1]
        label = line[2]
        texts.append(text_a)
        labels.append(label)
    return pd.DataFrame({'text': texts, 'label':labels})