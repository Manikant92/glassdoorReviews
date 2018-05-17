# import necessary packages for analysis
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import collections
from operator import itemgetter
from wordcloud import WordCloud

# import these packages if you want play with word cloud images and to view
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# import requests
# from io import BytesIO

# i have used jupyter notebook to analyse and directly using code here for what is necessary in this file
# refer to jupyter notebook for analysis
df = pd.read_csv('accentureGlassdoorReviews.csv')
# filling NAN with none text, this file contains only in advice column,
# you can simply apply on whole dataframe (by removing column specified below) as well if it contains in other columns
#df.fillna('None', inplace=True)
df['advice'].fillna('None', inplace=True)


# cleaning data function
def pre_processing(text):
    '''
    pre processing and cleaning the data that is unwanted
     '''
    # convert all words to smaller case
    text = text.lower()
    # replace none words
    text = text.replace('none', '')
    # text = text.replace('na', '')
    text = text.replace('nothing', '')
    text = text.replace('n/a', '')
    text = text.replace('nil', '')
    # text = text.replace('no','')
    # replace urls if any
    text = re.sub(r"http.?://[^\s]+[\s]?", '', text)
    # replace email id's if any
    text = re.sub(r"\S+@\S+", '', text)
    # replace numbers and digits with space
    text = re.sub('[0-9]', '', text)
    # replace any single chars with space, ex: a, i, m, u
    text = re.sub(r"\b[a-z]\b", '', text)

    # replace punctuations with spaces
    text = text.replace("(", '').replace(":", '').replace(")", '').replace(".", '').replace("'", '').replace(",",
                                                                                                             '').replace(
        '"', '').replace("<", '').replace(">", '')
    text = text.replace(",", '').replace(":", '').replace("/", '').replace("=", '').replace("&", '').replace(";",
                                                                                                             '').replace(
        "%", '').replace("$", '').replace("%", '')
    text = text.replace("@", '').replace("^", '').replace("*", '').replace("{", '').replace("}", '').replace("[",
                                                                                                             '').replace(
        "]", '').replace("|", '').replace("\\", '')
    text = text.replace("//", '').replace("-", '').replace("!", '').replace("`", '').replace("~", '').replace("?",
                                                                                                              '').replace(
        "--", '').replace("---", '').replace("#", '')
    text = text.replace("+", '')
    # trail all spaces
    text = re.sub('\s+', ' ', text).strip()

    return text


# convert pro reviews to list then preprocess and store it in another list
sentences = df['pro'].tolist()
pos_sentences = []
for sentence in sentences:
    processSent = pre_processing(sentence)
    pos_sentences.append(processSent)

# remove spaces list if any in the list
pos_sentences = [x for x in pos_sentences if x != '']

# convert con reviews to list then preprocess and store it in another list
sentences = df['con'].tolist()
con_sentences = []
for sentence in sentences:
    processSent = pre_processing(sentence)
    con_sentences.append(processSent)

# remove spaces list if any in the list
con_sentences = [x for x in con_sentences if x != '']

sentences = df['advice'].tolist()
adv_sentences = []
for sentence in sentences:
    processSent = pre_processing(sentence)
    adv_sentences.append(processSent)

adv_sentences = [x for x in adv_sentences if x != '']

# tokenize the sentences into words and store in list
pos_words = []
for sentence in pos_sentences:
    tokens = word_tokenize(sentence)
    pos_words.append(tokens)

# list contains sublists, convert all sublists into one list
pos_words = [sl for li in pos_words for sl in li]

con_words = []
for sentence in con_sentences:
    tokens = word_tokenize(sentence)
    con_words.append(tokens)

con_words = [sl for li in con_words for sl in li]

adv_words = []

for sentence in adv_sentences:
    tokens = word_tokenize(sentence)
    adv_words.append(tokens)

adv_words = [sl for li in adv_words for sl in li]

#removing stopwords and adding few more stopwords to this words
stopwords = stopwords.words('english')
addstopwords = ['please','make','still']
stopwords.extend(addstopwords)

#filter word tokens by removing stopwords

filtered_pos_words = []
for word in pos_words:
    if word not in stopwords:
        filtered_pos_words.append(word)

filtered_con_words = []
for word in con_words:
    if word not in stopwords:
        filtered_con_words.append(word)

filtered_adv_words = []
for word in adv_words:
    if word not in stopwords:
        filtered_adv_words.append(word)

#count frequent words used in the reviews and sort them based on highest occurency
pos_unigrams = collections.Counter(nltk.ngrams(filtered_pos_words,1))
pos_bigrams = collections.Counter(nltk.ngrams(filtered_pos_words,2))
pos_trigrams = collections.Counter(nltk.ngrams(filtered_pos_words,3))

sorted_pos_unigrams = sorted(pos_unigrams.items(), key=itemgetter(1),reverse=True)
sorted_pos_bigrams = sorted(pos_bigrams.items(), key=itemgetter(1), reverse=True)
sorted_pos_trigrams = sorted(pos_trigrams.items(), key=itemgetter(1), reverse=True)

con_unigrams = collections.Counter(nltk.ngrams(filtered_con_words,1))
con_bigrams = collections.Counter(nltk.ngrams(filtered_con_words,2))
con_trigrams = collections.Counter(nltk.ngrams(filtered_con_words,3))

sorted_con_unigrams = sorted(con_unigrams.items(), key=itemgetter(1),reverse=True)
sorted_con_bigrams = sorted(con_bigrams.items(), key=itemgetter(1), reverse=True)
sorted_con_trigrams = sorted(con_trigrams.items(), key=itemgetter(1), reverse=True)

adv_unigrams = collections.Counter(nltk.ngrams(filtered_adv_words,1))
adv_bigrams = collections.Counter(nltk.ngrams(filtered_adv_words,2))
adv_trigrams = collections.Counter(nltk.ngrams(filtered_adv_words,3))

sorted_adv_unigrams = sorted(adv_unigrams.items(), key=itemgetter(1),reverse=True)
sorted_adv_bigrams = sorted(adv_bigrams.items(), key=itemgetter(1), reverse=True)
sorted_adv_trigrams = sorted(adv_trigrams.items(), key=itemgetter(1), reverse=True)

#above variables will be in format of tuples in a list
#converting them to dictionaries to pass it to wordcloud
pos_uni_dict = {}
for i in range(len(sorted_pos_unigrams)):
    pos_uni_dict[''.join(sorted_pos_unigrams[i][0])] = sorted_pos_unigrams[i][1]

pos_bi_dict = {}
for i in range(len(sorted_pos_bigrams)):
    pos_bi_dict[''.join(sorted_pos_bigrams[i][0])] = sorted_pos_bigrams[i][1]

con_uni_dict = {}
for i in range(len(sorted_con_unigrams)):
    con_uni_dict[''.join(sorted_con_unigrams[i][0])] = sorted_con_unigrams[i][1]

con_bi_dict = {}
for i in range(len(sorted_con_bigrams)):
    con_bi_dict[''.join(sorted_con_bigrams[i][0])] = sorted_con_bigrams[i][1]

adv_uni_dict = {}
for i in range(len(sorted_adv_unigrams)):
    adv_uni_dict[''.join(sorted_adv_unigrams[i][0])] = sorted_adv_unigrams[i][1]

adv_bi_dict = {}
for i in range(len(sorted_adv_bigrams)):
    adv_bi_dict[''.join(sorted_adv_bigrams[i][0])] = sorted_adv_bigrams[i][1]

#instantiate wordcloud object into variable

#to display max of highest 30 words, image height, width and background color to display
wordCloud = WordCloud(max_words=30, height=1000, width=1500, background_color='white')

#generate the word cloud and store it as image files in current project location
poswc_unigrams = wordCloud.generate_from_frequencies(pos_uni_dict)
poswc_unigrams.to_file('poswc_unigrams.png')
conwc_unigrams = wordCloud.generate_from_frequencies(con_uni_dict)
conwc_unigrams.to_file('conwc_unigrams.png')
advwc_unigrams = wordCloud.generate_from_frequencies(adv_uni_dict)
advwc_unigrams.to_file('advwc_unigrams.png')
poswc_bigrams = wordCloud.generate_from_frequencies(pos_bi_dict)
poswc_bigrams.to_file('poswc_bigrams.png')
conwc_bigrams = wordCloud.generate_from_frequencies(con_bi_dict)
conwc_bigrams.to_file('conwc_bigrams.png')
advwc_bigrams = wordCloud.generate_from_frequencies(adv_bi_dict)
advwc_bigrams.to_file('advwc_bigrams.png')

#to view in console using matplotlib
# plt.title('Pro Unigrams words')
# plt.imshow(poswc_unigrams, interpolation='bilinear')
# plt.axis("off")
# plt.show()

#below code is to play with forming different structure images like building word cloud in accenture logo format etc.
#if you have file in your local, you can ignore this response variable. This is for getting image from url
#response = requests.get('https://stanfordbases.files.wordpress.com/2015/03/accenture-logo.png')
#opening image from url and reading it. If file in local, you can BytesIO and pass image string directly into open
#image = Image.open(BytesIO(response.content))
#you need to convert images into grey for that structure
#convert colored images to greyscale for wordcloud
#image = image.convert('L')
#image.mode = 'L'
#image = image.point(lambda x:0 if x<128 else 255)
#if u want to store in ur local
#image.save('accenture.png')
#read image in array format
#masking = np.array(image)
#pass mask argument in wordcloud for that shape. refer to documentation for more details
#wordcloud = WordCloud(background_color='white',max_words=30,mask= masking,random_state=42).generate_from_frequencies(poswc_bigrams)


