import streamlit as st
import spacy
from annotated_text import annotated_text

st.title("EntityMysticism")
st.subheader("A sorta guide to NER for beginners")
st.markdown('''
Recently **ChatGPT** is making headlines on my Twitter feed, and so many people are interested in the field of NLP recently,
this gives an unique oppurtunity to present some essential topics in NLP.
A sentence is made of of several parts such as Verb, Noun , Pronoun , Adjective and so on.
''')


annotated_text(
    "This ",
    ("is", "VERB", "#8ef"),
    " some ",
    ("annotated", "ADJ", "#faa"),
    ("text", "NOUN", "#afa"),
    " for those of ",
    ("you", "PRONOUN", "#fea"),
    " who ",
    ("like", "VERB", "#8ef"),
    " this sort of ",
    ("thing", "NOUN", "#afa"),
    "."
)

@st.cache(show_spinner=False , allow_output_mutation=True)
def load_model():
    return spacy.load('en_core_web_sm')
nlp = load_model()



s = st.text_input('Enter a sentence',"John watched an old movie at the cinema.")
doc = nlp(s)
color = {
    "PROPN" : "#fea" ,
    "VERB" : "#8ef",
    "ADJ" : "faa",
    "NOUN" : "#afa"
}

def convert(word):
    if word.pos_ in color.keys():
        return (str(word) , word.pos_ , color[word.pos_])
    else :
        return str(word)

annotated_text(*[convert(t) for t in doc])


st.markdown('''
Today I will be introducting a simple yet important topic in this field i.e. NER (Named-Entity-Recognition). 


## NER
In NLP, anything that can be referred by a proper name is known as Named Entity.
It usually has Proper Noun in their speech tag.
That can values of :
- Person
- Location
- Geopolitical Entity
- Organization

In simple terms NER is basically tagging real world entities in our document. This leads to categorizing our documents in various way.
Application of NER
- Can help in categorizing documents in our corpus.
- Used in Question-answering.
- Critical in Information extraction.


All detected entities get classifications, and systems that use NER have various categories for these words. For example, you could classify the phrase "soy candle" under "soy products" or "combustion lighting," and you would be correct both times. Named entry recognition is a type of NLP or natural language processing, a kind of artificial intelligence.


''')


def process_text(doc, anonymize=False):
    tokens = []
    for token in doc:
        if (token.ent_type_ == "PERSON"):
            tokens.append((token.text, "Person", "#faa"))
        elif (token.ent_type_ in ["GPE", "LOC"]):
            tokens.append((token.text, "Location", "#fda"))
        elif (token.ent_type_ == "ORG") :
            tokens.append((token.text, "Organization", "#afa"))
        else:
            tokens.append(" " + token.text + " ")

    if anonymize:
        anonmized_tokens = []
        for token in tokens:
            if type(token) == tuple:
                anonmized_tokens.append(("X" * len(token[0]), "MASKED", token[2]))
            else:
                anonmized_tokens.append(token)
        return anonmized_tokens

    return tokens

x = "David is an current employee at Microsoft."
y = nlp(x)

tok = process_text(y)
# st.write([(ent.text , ent.label_) for ent in y.ents])
annotated_text(*tok)

st.markdown('''
We can use this to masked important information in a given document or file.
''')

ax = st.text_area('Enter Paragraph','Alice is working for Google.')
ay = nlp(ax)
atok = process_text(ay ,anonymize=True)
annotated_text(*atok)
