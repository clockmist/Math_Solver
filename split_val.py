import json
import random

random.seed(42)

with open("train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)
val_size = 3000
val_data = data[:val_size]
train_data = data[val_size:]

with open("val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"Total: {len(data)}, Train: {len(train_data)}, Val: {len(val_data)}")
