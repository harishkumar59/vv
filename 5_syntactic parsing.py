# Write a program to Implement syntactic parsing of a given text

import nltk
from nltk import CFG
from nltk.parse import ChartParser

# Define grammar (Phrase Structure Grammar)
grammar = CFG.fromstring("""
S -> NP VP
NP -> DT NOM | NOM
NOM -> JJ NOM | NN

VP -> VB | VB PP | VB NP | VB NP PP
PP -> IN NP

DT -> 'the' | 'a'
NN -> 'fox' | 'dog'
JJ -> 'quick' | 'brown' | 'lazy'
VB -> 'jumps' | 'saw'
IN -> 'over' | 'on'
""")

parser = ChartParser(grammar)

# Input sentence
sentence = "the quick brown fox jumps over the lazy dog".split()

print("\n=== CONSTITUENCY PARSING TREES ===\n")

for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()