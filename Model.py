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
    text (string): String of words

    Returns:
    A tokenized text
    """
    tokenized_text = TweetTokenizer(text)
    return tokenized_text.tokenize(text)


def decontracted(phrase):
    """
    Decontract uncontracted words out of a text.

    Parameters:
    phrase (string): String of words.

    Returns:
    A decontracted text.
    """
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
    Counts the times a word occurs.

    Parameters:
    ngrams (list):  A list of ngrams.

    Returns:
    A dictionary with the ngram as key and its count as value.
    """
    ngram_frequencies = {}
    for i in ngrams:
        if (i in ngram_frequencies):
            # Add +1 to word/ngram count
            ngram_frequencies[i] += 1
        else:
            # Add word/ngram to dic with value 1
            ngram_frequencies[i] = 1

    return ngram_frequencies


def sortngram(ngram):
    """
    Sort the ngram on their values.

    Parameters:
    ngrams (dictionary):  A dictionary of a ngram.

    Returns:
    A sorted ngram dictionary.
    """
    return sorted(ngram.items(), key=lambda x: x[1], reverse=True)


def probabilities(num, freq):
    """
    Calculate the ngram probabilities.

    Parameters:
    num (int): Number n of ngram.
    freq (dictionary): Dictionary that holds the ngram combined with the number of occurrences.

    Returns:
    A dictionary of the ngram combined with the probability of it occurring.
    """
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
                r"(?<!\..[.?!])(?<!mr[.?!])(?<!mrs[.?!])(?<!\s.[.?!])(?<=[.?!])\s+", phrase.casefold().strip())  # Split into sentences
            for part in splitline:
                if part != "":
                    tokenized_text = nltk_tokenizer(part)  # Split into words
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
    n (int): Number n of ngram.

    Returns:
    A dictionary with the ngram as key and its count as value.
    """
    unigrams = list(chain.from_iterable(data["Sentence"]))
    ngram = list(ngrams(unigrams, n))
    freq_ngram = get_frequencies(ngram)
    return freq_ngram


def generateSentence(data, n, sen):
    """
    Generate a sentence based on the given ngram.

    Parameters:
    data ():
    n (int): Number n of ngram.
    sen (int): Number of sentences.

    Returns:
    A sentence (string) provided by the getNewWords method.
    """
    ngrams = []
    # Load the probabilities dictionaries of the selected ngram and all preceding ngrams
    for i in range(n):
        ngrams.append(probabilities(i+1, createModel(data, i+1)))
    sentence = ""
    if (n == 1):
        k = 0  # Store value of currently found <E>'s
        while k < sen:
            roll = random.random()  # Random number between 0 and 1
            j = 0
            for l in ngrams[n-1]:
                j += ngrams[n-1].get(l)
                if(j >= roll):
                    sentence = sentence + " " + "".join(l)
                    if (l[0] == "<E>" or l[0] == "."):
                        k += 1
                        # Exchange <E> for . otherwise the sentence breaks will not be clear
                        sentence = re.sub(r"<E>", ".", sentence)
                    break
    else:
        currentWord = ("<S>",)
        k = 0  # Store value of currently found <E>'s
        return getNewWords(currentWord, n, ngrams, sen, sentence, k)
    return sentence


def getNewWords(currentWord, n, ngrams, sen, sentence, k):
    """
    At a word to the current sentence based on the current sentence and the given ngram until a sentence is created.
    Used by bigram and up.

    Parameters:
    currentWord (tuple): Tuple of the current wordlist needed for the ngram.
    n (int): Number n of ngram.
    ngrams (list): A list filled with the probabilities dictionaries of the selected ngram and all preceding ngrams.
    sen (int): Number of sentences.
    k (int): Currently found end of sentence indicators (<E>).
    sentence (string): String of the sentence currently produced by the methods.

    Returns:
    A sentence (string) when <E> is selected by the method.
    """
    roll = random.random()  # Random number between 0 and 1
    j = 0  # Store value of counted probabilities
    # Check if currentWord possesses enough words to use with n (selected ngram)
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


def capitalizeNamesLordOfTheRings(text):
    text = re.sub(r" adanel", " Adanel", text)
    text = re.sub(r" boromir", " Boromir", text)
    text = re.sub(r" lafduf", " Lagduf", text)
    text = re.sub(r" tarcil", " Tarcil", text)
    text = re.sub(r" annael", " Annael", text)
    text = re.sub(r" angrod", " Angrod", text)
    text = re.sub(r" angrim", " Angrim", text)
    text = re.sub(r" angelimir", " Angelimir", text)
    text = re.sub(r" angelimar", " Angelimar", text)
    text = re.sub(r" gondolin", " Gondolin", text)
    text = re.sub(r" penlod", " Penlod", text)
    text = re.sub(r" tarannon falastur", " Tarannon Falastur", text)
    text = re.sub(r" morwen steelsheen", " Morwen Steelsheen", text)
    text = re.sub(r" soronto", " Soronto", text)
    text = re.sub(r" finrod", " Finrod", text)
    text = re.sub(r" fingolfin", " Fingolfin", text)
    text = re.sub(r" finduilas", " Finduilas", text)
    text = re.sub(r" findis", " Findis", text)
    text = re.sub(r" findegil", " Findegil", text)
    text = re.sub(r" finarfin", " Finarfin", text)
    text = re.sub(r" fimbrethil", " Fimbrethil", text)
    text = re.sub(r" fengel", " Fengel", text)
    text = re.sub(r" fastred", " Fastred", text)
    text = re.sub(r" farin", " Farin", text)
    text = re.sub(r" faramir", " Faramir", text)
    text = re.sub(r" falathar", " Falathar", text)
    text = re.sub(r" snaga", " Snaga", text)
    text = re.sub(r" seers", " Seers", text)
    text = re.sub(r" siriondil", " Siriondil", text)
    text = re.sub(r" shagrat", " Shagrat", text)
    text = re.sub(r" scatha", " Scatha", text)
    text = re.sub(r" salmar", " Salmar", text)
    text = re.sub(r" sandyman", " Sandyman", text)
    text = re.sub(r" salgant", " Salgant", text)
    text = re.sub(r" sagroth", " Sagroth", text)
    text = re.sub(r" saeros", " Saeros", text)
    text = re.sub(r" saelon", " Saelon", text)
    text = re.sub(r" rufus burrows", " Rufus Burrows", text)
    text = re.sub(r" rudolph bolger", " Rudolph Bolger", text)
    text = re.sub(r" rog", " Rog", text)
    text = re.sub(r" radbug", " Radbug", text)
    text = re.sub(r" proudfoot", " Proudfoot", text)
    text = re.sub(r" ivriniel", " Ivriniel", text)
    text = re.sub(r" ivorwen", " Ivorwen", text)
    text = re.sub(r" ingwion", " Ingwion", text)
    text = re.sub(r" idril", " Idril", text)
    text = re.sub(r" hyarmendacil", " Hyarmendacil", text)
    text = re.sub(r" huor", " Huor", text)
    text = re.sub(r" huan", " Huan", text)
    text = re.sub(r" erestor", " Erestor", text)
    text = re.sub(r" erellont", " Erellont", text)
    text = re.sub(r" erchirion", " Erchirion", text)
    text = re.sub(r" eradan", " Eradan", text)
    text = re.sub(r" emeldir", " Emeldir", text)
    text = re.sub(r" elwing", " Elwing", text)
    text = re.sub(r" elros", " Elros", text)
    text = re.sub(r" elphir", " Elphir", text)
    text = re.sub(r" elmo", " Elmo", text)
    text = re.sub(r" elendur", " Elendur", text)
    text = re.sub(r" elendil", " Elendil", text)
    text = re.sub(r" elemmakil", " Elemmakil", text)
    text = re.sub(r" eldarion", " Eldarion", text)
    text = re.sub(r" eldacar", " Eldacar", text)
    text = re.sub(r" elboron", " Elboron", text)
    text = re.sub(r" egalmoth", " Egalmoth", text)
    text = re.sub(r" edrahil", " Edrahil", text)
    text = re.sub(r" ecthelion", " Ecthelion", text)
    text = re.sub(r" ebor", " Ebor", text)
    text = re.sub(r" borlas", " Borlas", text)
    text = re.sub(r" pelendur", " Pelendur", text)
    text = re.sub(r" pallando", " Pallando", text)
    text = re.sub(r" ostoher", " Ostoher", text)
    text = re.sub(r" varies", " Varies", text)
    text = re.sub(r" oromendil", " Oromendil", text)
    text = re.sub(r" orodreth", " Orodreth", text)
    text = re.sub(r" orleg", " Orleg", text)
    text = re.sub(r" ori", " Ori", text)
    text = re.sub(r" Orcobal", " Orcobal", text)
    text = re.sub(r" Ondoher", " Ondoher", text)
    text = re.sub(r" hild", " Hild", text)
    text = re.sub(r" herumor", " Herumor", text)
    text = re.sub(r" herucalmo", " Herucalmo", text)
    text = re.sub(r" herion", " Herion", text)
    text = re.sub(r" hendor", " Hendor", text)
    text = re.sub(r" hathol", " Hathol", text)
    text = re.sub(r" Hareth", " Hareth", text)
    text = re.sub(r" hardang", " Hardang", text)
    text = re.sub(r" handir", " Handir", text)
    text = re.sub(r" hallatan", " Hallatan", text)
    text = re.sub(r" hallacar", " Hallacar", text)
    text = re.sub(r" haleth", " Haleth", text)
    text = re.sub(r" haldar", " Haldar", text)
    text = re.sub(r" dwalin", " Dwalin", text)
    text = re.sub(r" durin", " Durin", text)
    text = re.sub(r" draugluin", " Draugluin", text)
    text = re.sub(r" dori", " Dori", text)
    text = re.sub(r" daeron", " Daeron", text)
    text = re.sub(r" borin", " Borin", text)
    text = re.sub(r" bombur", " Bombur", text)
    text = re.sub(r" nellas", " Nellas", text)
    text = re.sub(r" nauglath", " Nauglath", text)
    text = re.sub(r" narvi", " Narvi", text)
    text = re.sub(r" naugladur", " Naugladur", text)
    text = re.sub(r" muzgash", " Muzgash", text)
    text = re.sub(r" gwaihir", " Gwaihir", text)
    text = re.sub(r" vorondil", " Vorondil", text)
    text = re.sub(r" ungoliant", " Ungoliant", text)
    text = re.sub(r" tulkastor", " Tulkastor", text)
    text = re.sub(r" mairen", " Mairen", text)
    text = re.sub(r" aragorn ", " Aragorn", text)
    text = re.sub(r" gandalf", " Gandalf", text)
    text = re.sub(r" frodo", " Frodo", text)
    text = re.sub(r" bailings", " Bailings", text)
    text = re.sub(r" legolas", " Legolas", text)
    text = re.sub(r" brodda", " Brodda", text)

    return text


def cleanSentences(sentence):
    """
    Reformat the generated text.
    Delete start and end indications, delete spaces that are in the wrong location and add sentence breaks.

    Parameters:
    sentence (string): Complete generated text.

    Returns:
    A sentence (string) with the complete reformatted generated text.
    """
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


def generateText(dataset, ngram, sentences, filter):
    """
    Generate complete new text out of the provided dataset.

    Parameters:
    dataset()
    ngram (integer): Number of N-Gram to use.
    sentences (integer): Number of sentences to generate.
    filter (function): function call to make to reformat the text.

    Returns:
    The complete reformatted generated text.
    """
    # Load and tokenize all provided text
    data = LoadData(dataset)

    # Generate sentence(s) according to ngram
    textNoMarkup = ("<S>" + generateSentence(data, ngram, sentences) + "<E>")

    # Reformat the text
    cleanText = cleanSentences(textNoMarkup)

    # Use extra filter to format text
    newText = filter(cleanText)

    # Print the generated text
    print(newText)


# ------------------------------------------------------------------------
# -------------------------------User Input-------------------------------
# ------------------------------------------------------------------------


# Insert here your complete dataset which should be used to create your N-Gram model.
books = ["Datasets/Lord_of_the_Rings/1-The-Fellowship-Of-The-Ring.txt",
         "Datasets/Lord_of_the_Rings/2-The-Two-Towers.txt", "Datasets/Lord_of_the_Rings/3-The-Return-Of-The-King.txt"]

# Insert here an extra function used for formatting of the generated text.
# Provided in this file are: capitalizeNamesLordOfTheRings and capitalizeNamesHarryPotter.
filter = capitalizeNamesLordOfTheRings

# Create N-Gram model, generate text and print this new text.
# 1st param is the dataset selected before.
# 2nd param is an integer representing the n-gram which you want to use.
# 3rd param is an integer representing the number of sentences you want to generate.
# 4th param is a function used to further add formatting to a text.
generateText(books, 6, 10, filter)
