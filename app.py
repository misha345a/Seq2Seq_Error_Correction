import streamlit as st
import sys
import nltk
nltk.download('punkt')
import string
import re
import random
import pickle
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk.tokenize import TweetTokenizer
import torch

#####################################

def encode_sequences(tokenizer, length, lines):
  """
  Encode training data sentences into
  padded sequences.
  """
  # integer encode sequences
  seq = tokenizer.texts_to_sequences(lines)

  # pad sequences with values of 0
  seq = pad_sequences(seq, maxlen=length, padding='post', truncating='pre',)
  return seq

incorrect_length = 20
correct_length = 20

def parenthesis_span(text):
  """
  Extract the (start, end) span of the quotation string
  within the text, if there is one.
  """
  search = [x.span() for x in re.finditer(r'\([^(]*\)', text)]

  if len(search) == 0:
    return [-9999, -9999]
  else:
    return list(search[-1])

def extract_authors(text):
  """
  Extract author names from the text.
  Ignore any occurances of names within quotes.
  """
  # apply NER model
  ignore_labels = ['LOC', 'MISC', 'ORG', 'O']
  output = classifier(text, aggregation_strategy='simple', ignore_labels=ignore_labels)

  # define quotation
  quote_start, quote_end = parenthesis_span(text)[0], parenthesis_span(text)[1]

  # return names not used in a quotation
  names = []
  for i in range(len(output)):
    if quote_start < output[i]['start'] < quote_end:
      names.append(output[i]['word'])

  # split any double names (i.e. ['John Smith'] => ['John', 'Smith'])
  names = " ".join(names).replace('and', '').split()
  names = list(set(names))
  return names

punct = '!"&(),.:;?'
punct_pattern = re.compile("[" + re.escape(punct) + "]")

def custom_tokenization(text):
  """
  Split text on a word-level using NLTK's TweetTokenizer method.
  Uniformize all words, authors, and numbers to put focus on
  punctuation and sentence structure.
  """
  # split text and punctuation on a word-level
  tokenized = [token for token in TweetTokenizer().tokenize(text.strip())]

  # extract author names
  names = extract_authors(text)

  # define variables
  word_count = 4
  output_str = ''
  map_hash = []
  org_hash = []

  # loop through text in reverse
  for i in range(1,len(tokenized)+1):
    ent = tokenized[-i] # entity

    if ent.lower() in names:
      output_str += f"$NAME{str(names.index(ent.lower()))} "
      map_hash.append(f"$NAME{str(names.index(ent.lower()))}")
      org_hash.append(ent)

    elif ent.isdigit():
      output_str += f"$NUM "
      map_hash.append(f"$NUM ")
      org_hash.append(ent)

    elif ent.lower()=='et' or ent.lower()=='al' or ent.lower()=='and':
      output_str += f"{ent.lower()} "
      map_hash.append(ent)
      org_hash.append(ent)

    elif re.search(punct_pattern, ent):
      output_str += f"{ent} "
      map_hash.append(ent)
      org_hash.append(ent)

    else:
      output_str += f"$WORD{str(word_count)} "
      map_hash.append(f"$WORD{str(word_count)}")
      org_hash.append(ent)
      word_count -= 1

    if word_count < 0:
      break

  # clip to extract only the last 5 words at most
  try:
    clipped_str = output_str[0: output_str.index("WORD0")+5]
    clipped_output = " ".join(clipped_str.split()[::-1])
  except:
    return ''

  # add name token to beginning of clipped sequence
  for name in names:
    if name in text[0:parenthesis_span(text)[0]].lower():
      clipped_output = '$NAME0 ' + clipped_output
      map_hash.insert(0, f"$NAME0 ")
      org_hash.insert(0, names[names.index(name)])
      break

  return clipped_output, map_hash, org_hash

def get_word(n, tokenizer):
  """

  """
  for word, index in tokenizer.word_index.items():
    if index == n:
      return word
  return None

def output_pred(prediction_sequence):
  """
  """
  preds_text = []
  for i in prediction_sequence:
    temp = []
    for j in range(len(i)):
      t = get_word(np.argmax(i[j]), correct_tokenizer)
      if j > 0:
        if (t==get_word(np.argmax(i[j-1]), correct_tokenizer)) or (t == None):
          temp.append('')
        else:
          temp.append(t)
      else:
        if(t == None):
          temp.append('')
        else:
          temp.append(t)
    preds_text.append(' '.join(temp).upper().replace('  ', ' ').strip())
  return preds_text

def prediction(text):
  tokenized_seq, map, org = custom_tokenization(text)
  seq = encode_sequences(incorrect_tokenizer, incorrect_length, np.array([tokenized_seq], dtype='<U87'))
  preds = model.predict(seq)
  pred_seq = output_pred(preds)[0]

  return post_processing(text, pred_seq, map, org)

def post_processing(text, sequence, map, org):
  """

  """
  df_post_processing = pd.DataFrame(zip(sequence.split(), map[::-1], org[::-1]),
                                    columns=['Pred', 'Map', 'Org'])

  error_types = ['$DELETE_PUNCT', '$ADD_SEMICOLON', '$REPLACE_SEMICOLON', '$DELETE_NAME', '$ADD_PERIOD']

  for i in df_post_processing['Pred'][::-1]:
    if i in error_types:
      error_type = error_types[error_types.index(i)]
      index = df_post_processing[df_post_processing['Pred']==error_type].index.values[0]
      org_snippet = df_post_processing.iloc[index-1:index+2,:]['Org'].tolist()

      edit = re.compile('('+re.escape(org_snippet[0])+')'+'(\s+)?'+'('+re.escape(org_snippet[1])+')'+'(\s+)?'+'('+re.escape(org_snippet[2])+')')
      if error_type == '$DELETE_PUNCT':
        text = re.sub(edit, r'\1\2\4\5', text)

      elif error_type == '$ADD_SEMICOLON':
        text = re.sub(edit, r'\1; \3\4\5', text)

      elif error_type == '$REPLACE_SEMICOLON':
        text = re.sub(edit, r'\1;\4\5', text)

      elif error_type == '$DELETE_NAME':
          if '$DELETE_NAME $NAME1' in sequence:
              text = re.sub(edit, r'\1\2\5', text)
              error_type =  "$DELETE_FIRST_NAME"
          else:
              text = re.sub(edit, r'\3\4\5', text)

      elif error_type == '$ADD_PERIOD':
        text = re.sub(edit, r'\1. \3\4\5', text)

      text = re.sub(r"\s{2,}", "", text).strip()
      return text, error_type

  return None, None

error_messages = {'$DELETE_PUNCT': 'Deleted incorrectly placed punctuation.',
                  '$ADD_SEMICOLON': 'Inserted semicolon to correctly seperate works.',
                  '$REPLACE_SEMICOLON': 'Replaced incorrect punctuation type with semicolon.',
                  '$DELETE_NAME': 'Author already mentioned in text, removed name.',
                  '$DELETE_FIRST_NAME': "Only author's last name should be used. Removed first name.",
                  '$ADD_PERIOD': 'Inserted missing period.'}

#####################################

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def load_ner_model():
    model_name = 'elastic/distilbert-base-uncased-finetuned-conll03-english'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp_pipe = pipeline(task='ner', model=model, tokenizer=tokenizer, framework='pt')
    return nlp_pipe

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def load_seq2seq_model():
    return keras.models.load_model('./model')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def load_tokenizers():
    with open('./tokenizers/incorrect_tokenizer.pickle', 'rb') as handle:
        incorrect_tokenizer = pickle.load(handle)
    with open('./tokenizers/correct_tokenizer.pickle', 'rb') as handle:
        correct_tokenizer = pickle.load(handle)
    return incorrect_tokenizer, correct_tokenizer

# load models and display temporary message
with st.spinner("Please be patient. Loading models..."):
    classifier = load_ner_model()
    model = load_seq2seq_model()
    incorrect_tokenizer, correct_tokenizer = load_tokenizers()

st.title("Demo: Correction of MLA In-Text Citations")

st.subheader("Motivation")
st.markdown("Existing services, such as Easybib, CitationMachine, Scribbr, \
help create in-text citations, but many students still struggle to \
correctly incorporate them within their works. \n This AI-powered \
prototype seeks to detect and correct common mistakes:")

image = Image.open('./images/example.PNG')
st.image(image,
         caption='Image sourced from lumenlearning.com',
         output_format='PNG')

st.subheader("Under the Hood")
st.markdown("Training data was created using pattern-based error generation \
to replicate common errors (3 million+ observations).\n \
A Seq2Seq LSTM model in Keras was trained to translate faulty observations into correct ones.\n \
A pretrained BERT NER model is leveraged during pre and post-processing steps to identify names.")

# st.subheader("Future Directions")
# st.markdown("1. Improve model robustness by training on a greater range of errors.\n \
# 2. Error detection for other writing formats - APA, Chicago, etc.\n\n \
# This demo is is far from perfect, and you will surely see a great many mistakes. But it's a start!")

examples = ['Parents play an important role in helping children learn techniques for coping with bullying. (Lang).',
            'One study found that the most important element in comprehending non-native speech is familiarity with the topic (Lee and Martin, 163).',
            '"Margaret had never spoken of Helstone since she left it" (Moore, 100).',
            'However, this study showed promising results for gene therapy. (Davies et al. 21).',
            'The authors claim that surface reading looks at what is "evident, perceptible, apprehensible in texts" (Best and Marcus, 9).',
            'As indicated by Williams, it is c extensively explored the role of emotion in the creative process (Williams 263).',
            'Smith demonstrates this psychological phenomenon in his work (Smith 22).',
            'Thus, the main character exemplified "aspects of posthumanism" (Alex Shelby 31).',
            'There are ways to document this "overwhelming need for self-study." (Jones et al. 18).',
            'In response, Mary replied to the politician with disgust! (Blackwell 43).',
            'Global warning is the biggest threat to our planet (Brown 40 Smith 50)',
            'Green imagines a "new type of oligarchy." (232).',
            'This novel recently drew critical attention (Talbot and Martin 22).',
            'Livestock farming is one of the biggest global contributors to climate change. (Raul 64; Douglas 14).',
            'As a painter, Andrea was a romantic (Margaret Flint 98).']

with st.form(key='my_form'):
    st.text("Enter your sentence below:")
    input_text = st.text_area(label='',
                              max_chars=250)
    input_button = st.form_submit_button(label='Run Input')

with st.form(key='my_form1'):
    st.text("or... use a random example:")
    generate_button = st.form_submit_button(label='Generate Example and Run')
    if generate_button:
        example_text = examples[random.randrange(0,len(examples))]
        st.session_state.input = example_text
    generate_text = st.text_area(label='',
                                 key='input',
                                 max_chars=250)

# display prediction
st.subheader('Prediction')

def handle_output(text):
    # error handling
    if len(text.split()) < 5:
        st.error('Please enter a longer sentence.')
    elif text.isspace() or text.isnumeric() or not text.isascii():
        st.error('Please enter a valid textual input.')
    else:
        try:
            post_text, error = prediction(text)
            if error is None:
              st.write("All correct here!")
            else:
              st.write("Error detected.")
              st.write(f"Revision: {error_messages[error]}")
              st.write(post_text)
        except:
            st.error("Sorry, an error has occured. Try again.")

if generate_button:
    text = generate_text.strip()
    handle_output(text)

if input_button:
    text = input_text.strip()
    handle_output(text)
