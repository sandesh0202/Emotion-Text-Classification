import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from flask import Flask, request, render_template

app = Flask(__name__)

data = pd.read_csv("C:/Programs/Emotion Text Classification/text.csv")

stemmer = PorterStemmer() 

def remove_punctuation(sentence):
    sentence = str(sentence)
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

emotions = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}



@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/', methods=['GET', 'POST'])
def emotion_classify():
    text = request.form.get('text')
    preprocessed_text = preprocess(text)
    vectorized = vectorizer.transform([preprocessed_text])
    emotion = xgb.predict(vectorized)
    if emotion == 0:
        gif_address = "https://res.cloudinary.com/djhtor2yi/image/upload/v1712488177/gemabrnlwkziymnczghq.gif"
        output = "Why are you 'Sad', Be Happy"
    elif emotion == 1:
        gif_address = "https://res.cloudinary.com/djhtor2yi/image/upload/v1712488179/xzd0rv0ztlq4ggsehnmj.gif"
        output = "Great, What you said is 'Joyfull'" 
    elif emotion == 2:
        gif_address = "https://res.cloudinary.com/djhtor2yi/image/upload/v1712488177/xdd0osvzcnnpuoxukkxx.gif"
        output = "Wow, You're such a Full of 'Love' person"
    elif emotion == 3:
        gif_address = "https://res.cloudinary.com/djhtor2yi/image/upload/v1712488177/fx8wmb1iyquwubscgyt5.gif"
        output = "Please Reduce your 'Anger', Be Calm!"
    elif emotion == 4:
        gif_address = "https://res.cloudinary.com/djhtor2yi/image/upload/v1712488177/edncu39l4hv3bbeywwmy.gif"
        output = "'Fear' is just a Mirage created by Mind, do not fear"
    elif emotion == 5:
        gif_address = "https://res.cloudinary.com/djhtor2yi/image/upload/v1712488178/bnuhmmhdmjmrdfm6msyx.gif"
        output = "You look 'Surprised' "
    print(emotions[emotion[0]])
    
    return render_template("index.html", gif_address = gif_address, output = output)
    

if __name__ == "__main__":
    app.run()