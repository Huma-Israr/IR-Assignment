import random
import re
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
#spacy.load('en_core_web_sm')
#spacy.load('en')
from spacy.lang.en import English
import string
parser = English()
import gensim
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import movie_reviews

def tokenize(text):
    lda_tokens = []
    text.translate(dict.fromkeys(string.punctuation))
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.orth_.startswith('@'):
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


for w in ['dogs', 'ran', 'discouraged']:
    print(w, get_lemma(w), get_lemma2(w))

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

text_data = []
short_pos = open("dataset/combinedtweet.txt","r",encoding="ANSI").read()
tweet = short_pos.split('\n')
document = []
for t in tweet:
    item = t.split('|')
if len(item) >= 3:
    clean = re.sub('r' "http\\S+", item[2])
    document.append(clean)
print ("Done Cleaning")

for line in document:
    tokens = prepare_text_for_lda(line)
    if random.random() > .99:
     print(tokens)
    text_data.append(tokens)
print ("Done...")

len(text_data)


dictionary = corpora.Dictionary(text_data)

corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim

NUM_TOPICS = 4
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=25)
ldamodel.save('model4.gensim')



topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)


def predict_class(new_tweet):
    new_tweet = prepare_text_for_lda(new_tweet)
    new_tweet_bow = dictionary.doc2bow(new_tweet)
    #     print(new_doc_bow)
    pred = ldamodel.get_document_topics(new_tweet_bow)
    print (pred)
    max = pred[0][1]
    label = 0
    for i in range(1, 4):
      if (pred[i][1] > max):
       max = pred[i][1]
       label = i
    return label
    print (label)
    print('Label is %d with proabbility of %f' % (label, max))

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model4.gensim')


nltk.download('movie_reviews')
documents_final = [(list(movie_reviews.words(fileid)), category)
                   for category in movie_reviews.categories()
                   for fileid in movie_reviews.fileids(category)]
random.shuffle(documents_final)


csv = open("dataset/health_tweets_labeled.csv", "a")

columnTitleRow = "tweet, class"
csv.write(columnTitleRow)
for i in range(13618, len(text_data)):
  tweet = text_data[i]
  tweet_str = ' '.join(str(e) for e in tweet)
#     re.sub(r'[^\\x00-\\x7F]+',' ', tweet_str)
  #row = tweet_str.replace(",","\") + "," + str(predict_class(tweet_str)) + "\"
  #csv.write(row)
csv.close()

#print(text_data[13617])

#csv.close()
