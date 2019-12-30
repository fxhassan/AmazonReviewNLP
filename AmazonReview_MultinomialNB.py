import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
print("at step one")


dataset = pd.read_json("Appliances.json",lines=True)
dataset = shuffle(dataset,random_state=21)
dataset = dataset.iloc[:10000, [0,8]]
dataset = pd.DataFrame.dropna(dataset)
dataset = pd.DataFrame.reset_index(dataset)

print("at step two")

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
corpus = [] 
for i in range(0,len(dataset)):
    stemmer = SnowballStemmer("english")
    review = re.sub("[^a-zA-z]"," ",dataset["reviewText"][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    print(i)
    corpus.append(review)
    
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer   
 
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer()),
                      ('clf-svm', MultinomialNB())])
  
X = corpus
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier.fit(X_train, y_train)
    
y_pred = classifier.predict(X_test)

np.mean(y_pred == y_test)
