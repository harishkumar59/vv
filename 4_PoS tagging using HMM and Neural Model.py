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