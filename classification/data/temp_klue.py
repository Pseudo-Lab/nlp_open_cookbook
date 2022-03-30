import json
import pandas as pd

with open("ynat-v1.1_train.json", "r") as st_json:
    st_python = json.load(st_json)

print(len(st_python))

text, label = [], []
for i in range(0, len(st_python)):
    text.append(st_python[i]["title"])
    label.append(st_python[i]["predefined_news_category"])

df = pd.DataFrame({"text":text, "label":label})
df.to_csv("train_multi.csv", index=False)