import json
from nltk_utils import tokenize, stem, bag_of_word
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

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

        return word, label
        # return {
        #     "word": torch.tensor(word, dtype=torch.float32),
        #     "label": torch.tensor(label, dtype=torch.float32),
        # } !! also try this


chat_dataset = Chat_Dataset(X_train, y_train)
train_loader = DataLoader(chat_dataset, batch_size=4, shuffle=True)
