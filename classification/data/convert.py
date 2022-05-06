import pandas as pd

df = pd.read_csv("ratings_train.txt", delimiter="\t")
print(df)

text = df["document"].tolist()
label = df["label"].tolist()

new_label = []
for i in range(0, len(label)):
    if label[i] == 1:
        new_label.append("긍정")
    elif label[i] == 0:
        new_label.append("부정")

new_df = pd.DataFrame({"text": text, "label": new_label})
new_df.to_csv("train_binary_class.csv", index=False, encoding="utf8")
