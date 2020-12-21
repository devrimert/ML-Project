import pandas as pd
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

pokemons = pd.read_csv('Pokemon.csv', sep=r'\s*,\s*', header=0, engine='python')

print(pokemons.head())
print(pokemons.shape)
print(pokemons['Type 1'].unique())
print(pokemons.groupby('Type 1').size())

import seaborn as sns
sns.countplot(pokemons['Type 1'],label="Count")
plt.show()

pokemons.drop('Type 2', axis=1).plot(kind='box', subplots=True, layout=(9,9),
                                     sharex=False, sharey=False, figsize=(9,9), title='Pokemon Type Plot Box')
plt.savefig('pokemon_Type2Fig')
plt.show()

feature_names = ['Total', 'Attack', 'Defense', 'Speed']
X = pokemons[feature_names]
y = pokemons['Type 1']

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X, y, random_state=0)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)


#_____________________Decision Tree(CART)_________________________
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(trainX, trainY)
print('Accuracy of Decision Tree on training: {:.2f}'.format(dtc.score(trainX, trainY)))
print('Accuracy of Decision Tree on test: {:.2f}'.format(dtc.score(testX, testY)))
dtc2 = DecisionTreeClassifier(max_depth=3).fit(trainX, trainY)
print('Accuracy of Decision Tree on training: {:.2f}'.format(dtc2.score(trainX, trainY)))
print('Accuracy of Decision Tree on test: {:.2f}'.format(dtc2.score(testX, testY)))

#____________________/DesicionTree(CART)_________________________


#________________Linear Discriminant Analysis(LDA)________________

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(trainX, trainY)
print('Accuracy of LDA on training: {:.2f}'.format(lda.score(trainX, trainY)))
print('Accuracy of LDA on test: {:.2f}'.format(lda.score(testX, testY)))

#________________/Linear Discriminant Analysis(LDA)________________


#___________________Logistic Regression(LR)________________________

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(trainX, trainY)
print('Accuracy of Logistic regression on training: {:.2f}'.format(logreg.score(trainX, trainY)))
print('Accuracy of Logistic regression on test: {:.2f}'.format(logreg.score(testX, testY)))
#___________________/Logistic Regression(LR)________________________


#_______________________K-Nearest Neighbors(KNN)_________________________

from sklearn.neighbors import KNeighborsClassifier
knb = KNeighborsClassifier()
knb.fit(trainX, trainY)
print('Accuracy of K-NN on training: {:.2f}'.format(knb.score(trainX, trainY)))
print('Accuracy of K-NN on test: {:.2f}'.format(knb.score(testX, testY)))
#_______________________/K-Nearest Neighbors(KNN)_________________________


#________________________Gaussian Naive Bayes(NB)__________________________

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(trainX, trainY)
print('Accuracy of GNB on training: {:.2f}'.format(gnb.score(trainX, trainY)))
print('Accuracy of GNB on test: {:.2f}'.format(gnb.score(testX, testY)))
#________________________/Gaussian Naive Bayes(NB)__________________________


#________________________Support Vector Machine(SVM)________________________

from sklearn.svm import SVC
svm = SVC()
svm.fit(trainX, trainY)
print('Accuracy of SVM on training: {:.2f}'.format(svm.score(trainX, trainY)))
print('Accuracy of SVM on test: {:.2f}'.format(svm.score(testX, testY)))
#________________________/Support Vector Machine(SVM)________________________


#________________________Classification Report_______________________________

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
predict = knb.predict(testX)
print(confusion_matrix(testY, predict))
print(classification_report(testY, predict))

#________________________/Classification Report_______________________________