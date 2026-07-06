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