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
displacy.render(doc, style="ent", jupyter=True)
