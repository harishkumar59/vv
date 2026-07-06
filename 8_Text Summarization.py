# Write a program to Implement Text Summarization for the given sample text 
# pip install transformers torch sentencepiece accelerate


from transformers import pipeline
# We use text-generation because v5 prefers it for versatility
summarizer = pipeline("text-generation", model="facebook/opt-125m") 
text = "Summarize this: Artificial Intelligence is transforming industries like healthcare, education, and finance by automating tasks and improving decision-making."
# In v5, we often just ask the model to summarize directly
print(summarizer(text, max_new_tokens=50)[0]['generated_text'])