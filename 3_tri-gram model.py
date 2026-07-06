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