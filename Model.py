# Import the required libraries
import os
import re
import pandas as pd
import numpy as np
import glob
from nltk.tokenize import TweetTokenizer
from nltk import ngrams
from itertools import chain
import random


def nltk_tokenizer(text):
    """
    Tokenize the provided text with the imported TweetTokenizer

    Parameters:
    text (string): string of words

    Returns:
    A tokenized text
    """
    tokenized_text = TweetTokenizer(text)
    return tokenized_text.tokenize(text)


def decontracted(phrase):
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"_", "", phrase)
    phrase = re.sub(r"`", "\'", phrase)
    return phrase


def get_frequencies(ngrams):
    """
    Counts the times a word occurs

    Parameters:
    ngrams (list):  A list of ngrams

    Returns:
    A dictionary with the ngram as key and its count as value
    """
    ngram_frequencies = {}
    for i in ngrams:
        if (i in ngram_frequencies):
            # add +1 to word/ngram count
            ngram_frequencies[i] += 1
        else:
            # add word/ngram to dic with value 1
            ngram_frequencies[i] = 1

    return ngram_frequencies


def sortngram(ngram):
    return sorted(ngram.items(), key=lambda x: x[1], reverse=True)


def probabilities(num, freq):
    probs = {}
    for key in freq:
        if (num == 1):
            if "total" in probs:
                probs["total"] += freq.get(key)
            else:
                probs["total"] = freq.get(key)
        else:
            if key[:(num-1)] in probs:
                probs[key[:(num-1)]] += freq.get(key)
            else:
                probs[key[:(num-1)]] = freq.get(key)
    dict = {}
    for key in freq:
        if (num == 1):
            dict[key] = freq.get(key)/probs["total"]
        else:
            dict[key] = freq.get(key)/probs[key[:(num-1)]]
    return dict


def LoadData(books):
    """
    Load the data in a dataframe.
    This method uses case folding, decontracting.

    Parameters:
    books (list): A list containing the .txt files used.

    Returns:
    A dataframe containing tokenized text from the provided books.
    """
    sentence = []
    data = pd.DataFrame(columns=['Sentence'])

    for entry in books:
        text = open(entry, "r").readlines()
        for line in text:
            phrase = decontracted(line)
            splitline = re.split(
                r"(?<!\..[.?!])(?<!mr[.?!])(?<!mrs[.?!])(?<!\s.[.?!])(?<=[.?!])\s+", phrase.casefold().strip())
            for part in splitline:
                if part != "":
                    tokenized_text = nltk_tokenizer(part)
                    tokenized_text.insert(0, "<S>")
                    tokenized_text.append("<E>")
                    sentence.append(tokenized_text)

    data["Sentence"] = sentence
    return data


def createModel(data, n):
    """
    Creating n-gram frequency dictionaries.

    Parameters:
    data (dataframe):  A dataframe containing tokenized text.
    n (int): number n of ngram.

    Returns:
    A dictionary with the ngram as key and its count as value.
    """
    unigrams = list(chain.from_iterable(data["Sentence"]))
    ngram = list(ngrams(unigrams, n))
    freq_ngram = get_frequencies(ngram)
    return freq_ngram


def generateSentence(n, sen):
    """
    Generate a sentence based on the given ngram.

    Parameters:
    n (int): number n of ngram.

    Returns:
    A sentence (string) provided by the getNewWords method.
    """
    ngrams = []
    # Load the probabilities dictionaries of the selected ngram and all preceding ngrams
    for i in range(n):
        ngrams.append(probabilities(i+1, createModel(data, i+1)))
    sentence = ""
    if (n == 1):
        for i in range(round(20*random.random()) + 1):
            roll = random.random()
            j = 0
            for k in ngrams[n-1]:
                j += ngrams[n-1].get(k)
                if(j >= roll):
                    sentence = sentence + " " + "".join(k)
                    break
    else:
        currentWord = ("<S>",)
        k = 0  # store value of currently found <E>'s
        return getNewWords(currentWord, n, ngrams, sen, sentence, k)
    return sentence


def getNewWords(currentWord, n, ngrams, sen, sentence, k):
    """
    At a word to the current sentence based on the current sentence and the given ngram untill a sentence is created.
    Used by bigram and up.

    Parameters:
    currentWord (tuple): tuple of the current wordlist needed for the ngram.
    n (int): number n of ngram.
    ngrams (list): A list filled with the probablities dictionaries of the selected ngram and all preceding ngrams.
    sentence (string): string of the sentence currently produced by the methods.

    Returns:
    A sentence (string) when <E> is selected by the method.
    """
    roll = random.random()  # random number between 0 and 1
    j = 0  # store value of counted probabilites
    # Check if currentWord pocesses enough words to use with n (selected ngram)
    if (n - 1 == len(currentWord)):
        for i in ngrams[n-1]:
            if (i[:len(currentWord)] == currentWord):
                j += ngrams[n-1].get(i)
                if (j >= roll):
                    # Check if end of sentence was found
                    if (i[len(currentWord)] == "<E>"):
                        k += 1
                        if (k >= sen):
                            return sentence
                    currentWord = currentWord[1:]
                    sentence = sentence + " " + i[len(currentWord) + 1]
                    newWord = i[len(currentWord) + 1]
                    currentWord = currentWord + (newWord,)
                    return getNewWords(currentWord, n, ngrams, sen, sentence, k)
    # Use probabilities of lower n (ngram)
    else:
        newNgram = len(currentWord)
        for i in ngrams[newNgram]:
            if (i[:len(currentWord)] == currentWord):
                j += ngrams[newNgram].get(i)
                if (j >= roll):
                    if (i[len(currentWord)] == "<E>"):
                        k += 1
                        if (k >= sen):
                            return sentence
                    sentence = sentence + " " + i[len(currentWord)]
                    newWord = i[len(currentWord)]
                    currentWord = currentWord + (newWord,)
                    return getNewWords(currentWord, n, ngrams, sen, sentence, k)


def cleanSentences(sentence):
    sentence = re.sub(r"<S>", "", sentence)
    sentence = re.sub(r"<E>", "", sentence)
    sentence = re.sub(r"\s\.\s*", ". ", sentence)
    sentence = re.sub(r"\s!\s*", "! ", sentence)
    sentence = re.sub(r"\s\?\s*", "? ", sentence)
    sentence = re.sub(r"\s,\s*", ", ", sentence)
    sentence = re.sub(r"\s;", ";", sentence)
    sentence = re.sub(r"\s'", "'", sentence)
    sentence = re.sub(r"(?<=[.?!])'*\s", " \n", sentence)
    sentence = sentence.strip()
    sentence = sentence.capitalize()
    return sentence


# Select all text files
books = ["01 - The Fellowship Of The Ring.txt",
         "02 - The Two Towers.txt", "03 - The Return Of The King.txt"]

# Load and tokenize all provided text
data = LoadData(books)

# Generate sentence(s) accoriding to ngram
# GenerateSentence(ngram, numberOfSentences)
sentence = ("<S>" + generateSentence(7, 20) + "<E>")

text = cleanSentences(sentence)
print(text)

# np.savetxt(r"C:\Users\ruben\Desktop\NLP Project\test2.txt",
#           trigram_prob, fmt="%s")

# np.savetxt(r"C:\Users\ruben\Desktop\NLP Project\test.txt", data, fmt="%s")
