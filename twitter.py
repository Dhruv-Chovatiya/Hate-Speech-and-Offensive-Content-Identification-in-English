import streamlit as st
from model import *
import nltk
#nltk.download('all')
from sklearn.preprocessing import scale
import pickle
import numpy as np
import itertools
from gensim.models.word2vec import Text8Corpus
#from glove import Corpus, Glove
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from keras.models import Model
# from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
# from keras.optimizers import RMSprop
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
#from keras.utils import to_categorical
#from keras.callbacks import EarlyStopping
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
#%matplotlib inline

# provide sql-like data manipulation tools. very handy.
import pandas as pd
pd.options.mode.chained_assignment = None

# high dimensional vector computing library.
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle
import torch

import joblib
import gensim
from transformers import DistilBertModel, DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertModel


# the word2vec model gensim class
from gensim.models.word2vec import Word2Vec

# we'll talk about this down below
#LabeledSentence = gensim.models.doc2vec.LabeledSentence

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# a tweet tokenizer from nltk.
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
#from gensim.models.wrappers import FastText
#from gensim.models import FastText
#import fastText
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
st.set_page_config(  # Alternate names: setup_page, page, layout
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit".
	page_icon=None,  # String, anything supported by st.image, or None.
)
st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Cooper Black'; color: White;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Hate Speech And Offensive Speech Detection</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    tweet = st.text_input('Tweet')
    if(tweet)!='':
        ctweet = clean_text(tweet)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        clf = joblib.load('D:/study/Collage/Sem-8/Hate Speech And Offensive Speech Detection/distil_logi.pkl')
        encoded_input = tokenizer.batch_encode_plus([ctweet], max_length=300, truncation=True,padding='max_length')
        input_id = torch.LongTensor(encoded_input.input_ids)
        atte_mask = torch.LongTensor(encoded_input.attention_mask)

        output = model(input_id, atte_mask)
        last_hidden_states = output.last_hidden_state[:, 0]
        results = clf.predict(last_hidden_states.detach().numpy())
        if results[0]==0:
            original_title = '<p style="font-family:Courier; color:Red; font-size: 20px;">Offensive Tweet</p>'
            st.markdown(original_title, unsafe_allow_html=True)
        else:
            original_title = '<p style="font-family:Courier; color:Green; font-size: 20px;">Not Offensive Tweet</p>'
            st.markdown(original_title, unsafe_allow_html=True)

    else:
        st.write('Write Something')




