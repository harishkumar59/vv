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