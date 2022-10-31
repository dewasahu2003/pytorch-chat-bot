import json
import random
from black import out
import torch
from model import Bot_Model
from nltk_utils import bag_of_word, tokenize


with open("intents.json", "r") as f:
    intents = json.load(f)

FILE = "model.pytorch"
data = torch.load(FILE)

model_state = data["model_state"]
all_words = data["all_words"]
tags = data["tags"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
num_classes = data["num_classes"]


model = Bot_Model(input_size, hidden_size, num_classes)
model.load_state_dict(model_state)
model.eval()


bot_name = "jark"
while True:
    sentence = input("You: ")
    if sentence == "q":
        break
    sentence = tokenize(sentence)
    x = bag_of_word(sentence, all_words)
    x = x.reshape(-1, x.shape[0])
    x = torch.tensor(x, dtype=torch.float32)
    output = model(x)

    i, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)  # applying softmax here bcz

    probability = probabilities[0][predicted.item()]
    # same thing but getting things in terms of probability

    if probability > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                res = random.choice(intent["responses"])
                print(f"{bot_name}: {res}")
    else:
        print(f"{bot_name}: i do not understand...🙄")
