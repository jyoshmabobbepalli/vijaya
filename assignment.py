#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import os
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Define the file path for training labels
file_path = r'C:\Users\bhava\Downloads\archive (10)\Doctor’s Handwritten Prescription BD dataset\Training\training_labels.csv'
df = pd.read_csv(file_path)
print(f"Dataset loaded successfully. Preview:\n{df.head()}")

df['MEDICINE_NAME'] = df['MEDICINE_NAME'].str.lower().str.translate(str.maketrans('', '', string.punctuation))
df['GENERIC_NAME'] = df['GENERIC_NAME'].str.lower().str.translate(str.maketrans('', '', string.punctuation))

df['word'] = df['MEDICINE_NAME']  
df['label'] = df['MEDICINE_NAME']  

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
X = df['word']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

    
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

    


# In[ ]:




