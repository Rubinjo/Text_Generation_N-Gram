# Text_Generation_N-Gram
This project was executed as a school assignment at the University of Twente. For this project was a N-gram model created in Python by our project group. This was used to generate original Lord of the Rings texts based on all Lord of the Rings books. These new texts were then used in an enquete to test this model against a trained GPT-2 model and real texts from the books itself. This was finally all delivered in a project report.

This project generates quite convincing texts but still makes some obvious errors and doesn't perform as well as GPT-2. However, for such a simple model that doesn't use deep learning it performed above our expectations.

## Project Overview
- School: University of Twente
- Course: Natural Language Processing
- Assignment Type: Open Project
- Group Size: 2

## Extension
After delivery of this project to the university did this project see some extra refinement to the post process-text-formatting, such as capetilization and punctuation repairs, and has an extra Harry Potter dataset been added. 

## Methodology
The N-Gram model is based on looking at the probability that a sequence of words occurs. This approach to generating text is a Markov Model, which can also been expressed as a probabilistic language model. Here the length of this word sequence is determined by an integer value called n. To calculate these probabilities you look at a training set and how many times a specific word sequence occurs. With these probabilities you can then predict the probability of a word occurring when it was preceded by a certain sequence of words. This can be used to generate new words and eventually new sentences and even completely new pieces of text that this model gives as output.

## Install
To use this application you need to install **[Anaconda](https://www.anaconda.com/products/individual)** :snake:, which is an open-source distribution of Python packages that can be used for data science, machine learning, predictive analytics and more. Alternatively, the seperate packages could be installed, but this is not recommended.

## Usage
