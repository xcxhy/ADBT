import os
import sys
import spacy
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

def get_tag(text):
    sentence_taged = nltk.pos_tag(nltk.word_tokenize(text))
    return sentence_taged

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None

def get_lemmatize(text):
    tokens = word_tokenize(text) # tokenize
    tagged_sent = pos_tag(tokens) 

    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    if lemmas_sent==[]:
        return 0
    return lemmas_sent[0]
    