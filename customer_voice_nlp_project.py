
# Customer Voice Intelligence System
# Complete NLP Project: Sentiment Analysis and Opinion Mining

# --------------------------
# Step 1: Import Libraries
# --------------------------
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation

import spacy
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# Step 2: Load Dataset
# --------------------------
df = pd.read_csv("Reviews.csv")  # Amazon Fine Food Reviews
df = df[['Score', 'Time', 'Text', 'ProductId']].dropna()

# --------------------------
# Step 3: Data Cleaning
# --------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['Text'].astype(str).apply(clean_text)

# --------------------------
# Step 4: Sentiment Labeling
# --------------------------
def label_sentiment(score):
    if score > 3:
        return 'positive'
    elif score < 3:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['Score'].apply(label_sentiment)
df = df[df['sentiment'] != 'neutral']
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# --------------------------
# Step 5: TF-IDF Vectorization
# --------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# --------------------------
# Step 6: Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Step 7: Sentiment Classification
# --------------------------
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

# --------------------------
# Step 8: WordClouds
# --------------------------
def plot_wordcloud(sentiment):
    text = " ".join(df[df['sentiment'] == sentiment]['clean_text'])
    wc = WordCloud(width=800, height=400).generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{sentiment.capitalize()} Word Cloud")
    plt.show()

for s in ['positive', 'negative']:
    plot_wordcloud(s)

# --------------------------
# Step 9: N-Gram Analysis
# --------------------------
cv = CountVectorizer(ngram_range=(2,2), max_features=20)
bigrams = cv.fit_transform(df['clean_text'])
bi_freq = zip(cv.get_feature_names_out(), bigrams.sum(axis=0).tolist()[0])

sns.barplot(x=[x[1] for x in bi_freq], y=[x[0] for x in bi_freq])
plt.title("Top Bigrams")
plt.xlabel("Frequency")
plt.show()

# --------------------------
# Step 10: Topic Modeling (LDA)
# --------------------------
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
words = tfidf.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"\nTopic #{idx + 1}:", [words[i] for i in topic.argsort()[:-11:-1]])

# --------------------------
# Step 11: Sentiment Trend Over Time
# --------------------------
df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['month'] = df['Time'].dt.to_period('M')
sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)

sentiment_trend.plot(figsize=(12,6), title="Sentiment Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Review Count")
plt.show()

# --------------------------
# Step 12: Named Entity Recognition (NER)
# --------------------------
nlp = spacy.load("en_core_web_sm")
text_sample = df['Text'].iloc[0]
doc = nlp(text_sample)

print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
