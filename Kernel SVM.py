import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
path = r'C:\Users\santo\Desktop\ML Course\Part 3 - Classification\Section 17 - Kernel SVM\Python\Social_Network_Ads.csv'
df = pd.read_csv(path)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,random_state=0)

sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

clf = SVC(kernel='rbf', random_state=0)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
y_pred = y_pred.reshape(len(y_pred),1)
y_test= y_test.reshape(len(y_test),1)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred,y_test),1))

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm,acc)

