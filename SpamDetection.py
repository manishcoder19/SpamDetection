
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r'C:\Users\hp\Desktop\spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df.head()


# In[3]:


df.groupby('label').describe()


# In[4]:


sns.countplot(data=df, x='label')


# In[5]:


#Lets move directly to creating spam filter
#Our approach:

#Clean and Normalize text
#Convert text into vectors (using bag of words model) that machine learning models can understand
#Train and test Classifier
#Clean and normalize textIt will be done in following steps:
#Remove punctuationsRemove all stopwordsApply stemming (converting to normal form of word).
#For example, 'driving car' and 'drives car' becomes drive car


# In[6]:


import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
def process(text):
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming
    st = Stemmer()
    text = [st.stem(t) for t in text]
    # return token list
    return text


# In[7]:


process('It\'s holiday and we are playing cricket. Jeff is playing very well!!!')


# In[ ]:





# In[8]:


# Test with our dataset
df['message'][:20].apply(process)


# In[9]:


#Convert each message to vectors that machine learning models can understand.
#We will do that using bag-of-words model
#We will use TfidfVectorizer. It will convert collection of text documents (SMS corpus) into 2D matrix.
#One dimension represent documents and other dimension repesents each unique word in SMS corpus . .
#If nth term t has occured p times in mth document, (m, n) value in this matrix will be TF-IDF(t),

#where [TF-IDF(t)](https://en.wikipedia.org/wiki/Tf–idf) = Term Frequency (TF) * Inverse Document Frequency (IDF)

#Term Frequency (TF) is a measure of how frequent a term occurs in a document.

#TF(t)= Number of times term t appears in document (p) / Total number of terms in that document

#Inverse Document Frequency (IDF) is measure of how important term is. For TF, all terms are equally treated. But, in IDF, for words that occur frequently like 'is' 'the' 'of' are assigned less weight. While terms that occur rarely that can easily help identify class of input features will be weighted high.

#Inverse Document Frequency, IDF(t)= loge(Total number of documents / Number of documents with term t in it)

#At end we will have for every message, vectors normalized to unit length equal to size of vocalbulary (number of unique terms from entire SMS corpus)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


tfidfv = TfidfVectorizer(analyzer=process)
data = tfidfv.fit_transform(df['message'])


# In[12]:


mess = df.iloc[2]['message']
print(mess)


# In[13]:


print(tfidfv.transform([mess]))


# In[14]:


j = tfidfv.transform([mess]).toarray()[0]
print('index\tidf\ttfidf\tterm')
for i in range(len(j)):
    if j[i] != 0:
        print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'), tfidfv.get_feature_names()[i],sep='\t')


# In[15]:


#Having messages in form of vectors, we are ready to train our classifier.
#We will use Naive Bayes which is well known classifier while working with text data.
#Before that we will use pipeline feature of sklearn to create a pipeline of TfidfVectorizer followed by Classifier.
#Input will be message passed to first stage TfidfVectorizer which will transform it and pass it to Naive Bayes Classifier to get output label


# In[16]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
spam_filter = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)), # messages to weighted TFIDF score
    ('classifier', MultinomialNB())                    # train on TFIDF vectors with Naive Bayes
])


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.20, random_state = 21)
spam_filter.fit(x_train, y_train)


# In[18]:


predictions = spam_filter.predict(x_test)
count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)


# In[19]:


x_test[y_test != predictions]


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))


# In[21]:


def detect_spam(s):
    return spam_filter.predict([s])[0]
detect_spam('Your cash-balance is currently 500 pounds - to maximize your cash-in now, send COLLECT to 83600.')


# In[ ]:




