import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word: str):
    stem = PorterStemmer()
    return stem.stem(word=word.lower())


def bag_of_word(tokenised_arr_sentence, all_words):
    stem_tokenised_arr_sentence = [stem(w) for w in tokenised_arr_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in stem_tokenised_arr_sentence:
            bag[idx] = 1.0

    return bag


# t = tokenize("hello what are you doing")
# print(t)

# words = ["take", "taken", "taking"]
# stemmed_worlds = [stem(w) for w in words]
# print(stemmed_worlds)
