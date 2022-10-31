import json
from nltk_utils import tokenize, stem, bag_of_word
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from model import Bot_Model

with open("intents.json", "r") as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []  # for dataset...😄

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)  # keep adding to the array
        xy.append((w, tag))

ignore_word = ["?", ",", ".", "!"]
# ignore some words
all_words = [stem(w) for w in all_words if w not in ignore_word]
# keep unique word
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []  # train data
y_train = []  # train label

for (pattern, tag) in xy:
    bag = bag_of_word(pattern, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # class label encoding directly


X_train = np.array(X_train)
y_train = np.array(y_train)


class Chat_Dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        word = self.x[index]
        label = self.y[index]

        # return word,label
        return {
            "word": torch.tensor(word, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }
        # !! also try this


chat_dataset = Chat_Dataset(X_train, y_train)
train_loader = DataLoader(chat_dataset, batch_size=4, shuffle=True)


# hyper parameters
epochs = 1000
input_size = len(X_train[0])
hidden_size = 8
num_classes = len(tags)


model = Bot_Model(input_size=len(X_train[0]), hidden_size=8, num_classes=len(tags))
loss_fun = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(epochs):

    for i, data in enumerate(train_loader):

        output = model(data["word"])
        loss = loss_fun(output, data["label"])

        loss.backward()
        optim.step()
        optim.zero_grad()

    if epoch % 100 == 0:
        print(f"epoch:{epoch}/{epochs} || loss:{loss}")

print(f"final-loss:{loss}")


data = {
    "model_state": model.state_dict(),
    "all_words": all_words,
    "tags": tags,
    "epochs": 1000,
    "input_size": len(X_train[0]),
    "hidden_size": 8,
    "num_classes": len(tags),
}

torch.save(data, f="model.pytorch")
