# --------------------------practical 1------------
# Write a program to implement sentence segmentation and word tokenization 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize, sent_tokenize

# Now the rest of your code should run perfectly!
text = """Natural Language Processing is a fascinating field of artificial intelligence. 
It enables machines to read, understand, and derive meaning from human languages. 
By breaking down sentences into tokens, we can analyze complex patterns and data structures."""

word_tokens = word_tokenize(text)
print("Word Tokens:", word_tokens)

sentence_tokens = sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)


# --------------------------practical 2------------

# Write a program to Implement stemming and lemmatization 

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Downloads needed for the environment
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Initialize everything
words = ['run', 'runner', 'running', 'ran', 'runs', 'easily', 'fairly', 'caring', 'generous']
ps = PorterStemmer()
snowball = SnowballStemmer(language='english')
lancaster = LancasterStemmer()
wnl = WordNetLemmatizer()
# Print Header
print(f"{'Word':<12} | {'Porter':<12} | {'Snowball':<12} | {'Lancaster':<12} | {'Lemmatizer':<12}")
print("-" * 70)
# Run the comparison
for word in words:
    p_stem = ps.stem(word)
    s_stem = snowball.stem(word)
    l_stem = lancaster.stem(word)
    lemma = wnl.lemmatize(word, pos='v') # 'v' tells it to treat words as verbs 
    print(f"{word:<12} | {p_stem:<12} | {s_stem:<12} | {l_stem:<12} | {lemma:<12}")


import nltk
from nltk.stem import WordNetLemmatizer
# Necessary downloads for Colab
nltk.download('wordnet')
nltk.download('punkt_tab')
wordnet_lemmatizer = WordNetLemmatizer()
# Using standard quotes for Python
text = input("Enter words for Lemmatizing: ")
tokenization = nltk.word_tokenize(text)
print("\nResults:")
print("-" * 30)
# 'v' stands for verb, 'a' for adjective, 'n' for noun
for w in tokenization:
    lemma = wordnet_lemmatizer.lemmatize(w, pos='v')
    print("Lemma for {} is {}".format(w, lemma))
# --------------------------practical 3------------

# Write a program to Implement a tri-gram model 
from nltk.util import ngrams
# Input sentence (fixed the line break)
sentence = "Natural language processing is a field of study focused on the interactions between human language and computers."
# Tokenize the sentence into words
words = sentence.split()
# Get user input for n
try:
    n = int(input("Enter the value of n for n-grams (e.g., 2 for bigrams): "))
    # Create n-grams from the list of words
    # ngrams() returns a generator, so we can iterate through it
    ngrams_list = ngrams(words, n)
    print(f"\nGenerated {n}-grams:")
    print("-" * 30)
    # Print the n-grams
    for ngram in ngrams_list:
        # Joining the tuple for a cleaner look
        print(" ".join(ngram)) 
except ValueError:
    print("Please enter a valid number.")

# --------------------------practical 4------------

# Write a program to Implement PoS tagging using HMM ^ & Neural Model 
import nltk
# Necessary downloads for Colab environment
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng') # Note: updated for newer NLTK versions
text = "The dog barks at the mailman"
words = nltk.word_tokenize(text)
# Perform POS Tagging
tagged_words = nltk.pos_tag(words)
print("POS Tags:", tagged_words)
# A quick guide to the most common tags you'll see:
print("\n--- Tag Legend ---")
print("DT  : Determiner (the, a)")
print("NN  : Noun, singular (dog, mailman)")
print("VBZ : Verb, 3rd person singular present (barks)")

import nltk
from nltk.corpus import brown
from nltk.tag import hmm
# 1. Download data
nltk.download('brown')
nltk.download('universal_tagset')
# 2. Load and Prepare dataset
# Using the universal tagset makes the model simpler and more accurate
data = brown.tagged_sents(tagset='universal')
# 3. Split data (80% training, 20% testing is standard, but we'll stick to your counts)
train_data = data[:5000]
test_data = data[5000:5500]
# 4. Train the HMM Tagger
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)
# 5. Evaluate (using the updated accuracy method)
print(f"Model Accuracy: {tagger.accuracy(test_data):.2%}")
# 6. Test on a custom sentence
test_sentence = "I love programming with python".split()
tagged_sentence = tagger.tag(test_sentence)
print("\nTagged Sentence:")
print(tagged_sentence)

# --------------------------practical 5------------

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

# --------------------------practical 6------------


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
# --------------------------practical 7------------
# Write a program to Implement Named Entity Recognition (NER)
# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.Its headquarters are located in Cupertino, California."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, "->", ent.label_, "->", spacy.explain(ent.label_))
displacy.render(doc,style="ent",jupyter=True)




# --------------------------practical 8------------
# Write a program to Implement Text Summarization for the given sample text 
# pip install transformers torch sentencepiece accelerate


from transformers import pipeline
# We use text-generation because v5 prefers it for versatility
summarizer = pipeline("text-generation", model="facebook/opt-125m") 
text = "Summarize this: Artificial Intelligence is transforming industries like healthcare, education, and finance by automating tasks and improving decision-making."
# In v5, we often just ask the model to summarize directly
print(summarizer(text, max_new_tokens=50)[0]['generated_text'])