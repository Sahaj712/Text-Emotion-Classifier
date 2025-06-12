import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Tf-idf - Term Frequency Inverse Document Frequency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('text.csv')

df = df.drop(columns=['Unnamed: 0'])
df = df.rename(columns={
    'text': 'tweets',
    'label': 'label_id'
})

label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

df['emotions'] = df['label_id'].map(label_map)
#print(df[['tweets', 'label_id', 'emotions']].head())

emotion_counts = df['emotions'].value_counts()
print(emotion_counts)

#Bar chart
plt.figure(figsize=(8,5))
emotion_counts.plot(kind='bar', color='orange')
plt.title('Emotion Distribution')
plt.xlabel('Emotion')
plt.ylabel('Tweet Count')
plt.xticks(rotation=0)
plt.tight_layout()
#plt.show()

#Tweet Length Analysis
df['length'] = df['tweets'].str.split().str.len()
print(df[['tweets', 'length']].head())

#Tweets length chart
plt.figure(figsize=(8,5))
plt.hist(df['length'], bins=30, color = 'lightgreen', edgecolor = 'black')
plt.title('Tweet Length Distribution (Word Count)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Tweets')
plt.tight_layout()
#plt.show()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['clean_tweet'] = df['tweets'].apply(clean_text)
print(df[['tweets', 'clean_tweet']].head())

#Testing and Training Data 
x = df['clean_tweet']   
y = df['emotions']

xtrain, xtest, ytrain, ytest = train_test_split(
    x,y,
    test_size = 0.2,
    random_state = 42,
    stratify = y    
)

print("Training samples:", len(xtrain))
print("Testing samples:", len(xtest))

#Tf_IDF vectorizer 
vectorizer  =  TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(xtrain)
X_test_tfidf = vectorizer.transform(xtest)

print("TF-IDF matrix shape (train):", X_train_tfidf.shape)
print("TF-IDF matrix shape (test):", X_test_tfidf.shape)

#Logistic Regression
#Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, ytrain)

#Model Prediction
y_pred = model.predict(X_test_tfidf) #Accuracy

#Evaluation
print("\n\nAccuracy", accuracy_score(ytest, y_pred))
print("\n\n Classification Report: \n")
print(classification_report(ytest,y_pred))

#Save trained model
import joblib
joblib.dump(model, 'emotion_model.pkl') #Saving the model
joblib.dump(vectorizer, 'tfdif_vectorizer.pkl') #Saving the vectorizer
