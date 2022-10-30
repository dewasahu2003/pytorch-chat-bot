import nltk
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word: str):
    stem = PorterStemmer()
    return stem.stem(word=word.lower())


t = tokenize("hello what are you doing")
print(t)

words = ["take", "taken", "taking"]
stemmed_worlds = [stem(w) for w in s]
print(stemmed_worlds)
