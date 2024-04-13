## Emotion Text Classification

Emotion Classification Project is Natural Language Processing based on project that can detect different Emotions. 
1. Joyfull 2. Anger 3. Surprise 4. Fear 5. Sadness 6. Love
After user enters the Text, Application Preprocesses the Text, and Detects the Emotion using XGBoost Model.

### Dataset - 
- Dataset contains 40000 Human-annotated tweets from twitter and uploaded on Kaggle. 

### Preprocessing - 
- Application removes punctuations, websites, numbers, lowers the words to create clean words. 
- And Application also removes stopwords, Stopwords are words like the, it, him, which do not show any emotions.

- Preprocessing also includes process like Lemmatization and Stemming which are used to take words to their base forms. But We observed that the accuracy of the model reduces after we apply Lemmatization and Stemming.

### Model - 
We use XGBoost Model for, XGBoost uses Ensemble Learning Technique Gradient Boosting. We have also tried working on other Classifier Models like Logistic Regression, SVR, Decision Tree, Random Forest.

### Deployment
Application is deployed using Flask App and HTML & CSS
