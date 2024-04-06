import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

data = pd.read_csv("C:/Programs/Emotion Text Classification/text.csv")

stemmer = PorterStemmer() 

def remove_punctuation(sentence):
    pure_text = re.sub(r'(\s)?@\w+', r'\1', sentence)
    pure_text = re.sub(r'[^\w\s]', '', pure_text)
    pure_text = re.sub(r'\b.*com\b', '', pure_text)
    pure_text = re.sub(r'\bhttp\w+', '', pure_text)
    pure_text = pure_text.replace("  ", " ")
    pure_text = pure_text.lower()
    
    return pure_text

def preprocess(text):    
    text = remove_punctuation(text)
    
    text = remove_stopwords(text)
    
    return text

data = data[['text', 'label']]

data['preprocessed_text'] = data['text'].apply(lambda a: preprocess(a))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['preprocessed_text'])

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#xgb = XGBClassifier()
#xgb.fit(X_train, y_train)
#pickle.dump(xgb, open('xgb_model.pkl', 'wb'))

xgb = pickle.load(open('xgb_model.pkl', 'rb'))

xgb_pred = xgb.predict(X_test)

accuracy = accuracy_score(y_test, xgb_pred)

print(f"Accuracy is {accuracy}")

#class_report = classification_report(y_test, xgb_pred)
#print(class_report)

emotions = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

new_sentence =input("Please give me a Sentence - ")

preprocessed_text = preprocess(new_sentence)
vectorized = vectorizer.transform([preprocessed_text])
output = xgb.predict(vectorized)
print(f"Emotion of you're Sentence is - {emotions[output[0]]}")