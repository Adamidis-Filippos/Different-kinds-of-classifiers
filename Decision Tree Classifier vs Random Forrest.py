import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier

path = r'C:\Users\santo\Desktop\ML Course\Part 3 - Classification\Section 19 - Decision Tree Classification\Python\Social_Network_Ads.csv'
df = pd.read_csv(path)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

dtc = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dtc = dtc.fit(x_train,y_train)
rfc = RandomForestClassifier(n_estimators=5,random_state=0)
rfc = rfc.fit(x_train,y_train)

y_dtc = dtc.predict(x_test)
y_rfc = rfc.predict(x_test)

y_dtc = y_dtc.reshape(len(y_dtc),1)
y_rfc = y_rfc.reshape(len(y_rfc),1)
y_test = y_test.reshape(len(y_test),1)


np.set_printoptions(precision=2)
print(np.concatenate((y_rfc,y_test),1))
print(np.concatenate((y_dtc,y_test),1))

cm = confusion_matrix(y_test,y_dtc)
cm1 = confusion_matrix(y_test,y_rfc)
acc= accuracy_score(y_test,y_dtc)
acc1 = accuracy_score(y_test, y_rfc)
print('Decision Tree results:')
print(cm,acc)
print('Random Forrest Results')
print(cm1,acc1)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_dtc
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step =1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, dtc.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)
plt.title('Decision Tree')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = sc.inverse_transform(x_test), y_rfc
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step =1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, rfc.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)
plt.title('Random Forrest')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
