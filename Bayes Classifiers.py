import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

path = r'C:\Users\santo\Desktop\ML Course\Part 3 - Classification\Section 18 - Naive Bayes\Python\Social_Network_Ads.csv'
df = pd.read_csv(path)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.25, random_state=0)

y_te = np.array(y_te)
sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_te = sc.transform(x_te)

clf = GaussianNB()
clf = clf.fit(x_tr,y_tr)

y_pred = clf.predict(x_te)
y_pred = y_pred.reshape(len(y_pred),1)
y_te= y_te.reshape(len(y_te),1)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred,y_te),1))

cm = confusion_matrix(y_te,y_pred)
acc = accuracy_score(y_te,y_pred)
print(cm,acc)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_te), y_te
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)
plt.title('Naive Bayes')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()