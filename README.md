# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output
```

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: POOJA.S
RegisterNumber: 212223040146  
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))
```


## Output:
![Screenshot 2024-05-01 120157](https://github.com/Shubhavi17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150005085/a86c2ea3-3ef8-496a-9692-3b587bc64e70)
![Screenshot 2024-05-01 120206](https://github.com/Shubhavi17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150005085/b2ca1cc3-2368-416a-b407-9979a34e316d)
![Screenshot 2024-05-01 120214](https://github.com/Shubhavi17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150005085/45f95f58-7c37-453e-aa23-502410e34ef4)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
