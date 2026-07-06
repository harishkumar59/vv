# Write a program to Implement dependency parsing of a given text
# pip install spacy 
# python -m spacy download en_core_web_sm


import spacy
from spacy import displacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "The quick brown fox jumps over the lazy dog"

# Process the text
doc = nlp(text)

print("\n=== TOKEN LEVEL SYNTACTIC ANALYSIS ===\n")

# Print dependency relations
for token in doc:
    print(f"{token.text:10} | POS: {token.pos_:6} | DEP: {token.dep_:15} | HEAD: {token.head.text}")

print("\n=== ROOT WORD ===")
for token in doc:
    if token.dep_ == "ROOT":
        print("Root word:", token.text)

# Visualize dependency tree in browser
print("\nOpening dependency tree visualization in browser...")
displacy.serve(doc, style="dep")