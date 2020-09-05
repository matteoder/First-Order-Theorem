import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

train_set = pd.read_csv("C:\Users\Utente\Desktop\\theorem proof\ml-prove\\train.csv")
valid_set = pd.read_csv("C:\Users\Utente\Desktop\\theorem proof\ml-prove\\validation.csv")
test_set = pd.read_csv("C:\Users\Utente\Desktop\\theorem proof\ml-prove\\test.csv")

n_features = 51
X = train_set.iloc[:, 0:51].values
y = train_set.iloc[:, 51:57].values
X_val = valid_set.iloc[:, 0:51].values
y_val = valid_set.iloc[:, 51:57].values
X_test = test_set.iloc[:, 0:51].values
y_test = test_set.iloc[:, 51:57].values

all_data = np.concatenate((X, X_val, X_test))
n_dati = all_data.shape

# conversione dei valori di classificazione nelle etichette delle classi
ytrain = []
for i in range(3058):
    classe = np.argmax(y[i])
    ytrain.append(classe + 1)
yval = []
for j in range(1528):
    classe = np.argmax(y_val[j])
    yval.append(classe + 1)
ytest = []
for k in range(1529):
    classe = np.argmax(y_test[k])
    ytest.append(classe + 1)

all_target = np.concatenate((ytrain, yval, ytest))

# suddivisione del data set in training set e test set
train_data, X_test, train_target, ytest = train_test_split(all_data, all_target, test_size=0.2, random_state=42)
n_train_data = len(train_data)
n_test_data = len(X_test)

########################################################################################################################
# SVM
########################################################################################################################
pca = PCA()
svc = SVC(kernel='rbf', decision_function_shape='ovr')
svm = make_pipeline(pca, svc)
param_grid = {'pca__n_components': [18, 40, 19, 20, 35, 22, 43, 10, 41, 37], 'svc__C': [50, 80, 100, 60, 25],
              'svc__gamma': [0.01, 0.1, 0.2, 0.3, 0.4, 0.001]}

grid = GridSearchCV(svm, param_grid, cv=10, scoring='f1_weighted',verbose=True,n_jobs=4)
grid.fit(train_data, train_target)
print(grid.best_params_)

########################################################################################################################
# MLPClassifier
########################################################################################################################
pca = PCA()
mlpclassifier = MLPClassifier(solver='sgd', learning_rate_init=0.01, random_state=1)
mlp = make_pipeline(pca, mlpclassifier)
param_grid1 = {'pca__n_components': [18, 40, 19, 20, 35, 22, 43, 10, 41, 37], 'mlpclassifier__activation': ['relu', 'tanh'],
               'mlpclassifier__alpha': [1e-2, 1e-4, 1e-3],
               'mlpclassifier__hidden_layer_sizes': [(100, 30), (100, 20)],'mlpclassifier__early_stopping':[True,False]
               }

grid1 = GridSearchCV(mlp, param_grid1, cv=10, scoring='f1_weighted', n_jobs=4,verbose=True)
grid1.fit(train_data, train_target)
print(grid1.best_params_)

#######################################################################################################################
# K-NN
#######################################################################################################################
pca = PCA()
kneighborsclassifier = neighbors.KNeighborsClassifier()
nbrs = make_pipeline(pca, kneighborsclassifier)
parameters = {'pca__n_components': [18, 40, 19, 20, 35, 22, 43, 10, 41, 37],
              'kneighborsclassifier__n_neighbors': [5, 10, 15], 'kneighborsclassifier__p': [1, 2],
              'kneighborsclassifier__weights': ['distance', 'uniform']}
grid2 = GridSearchCV(nbrs, parameters, scoring='f1_weighted', cv=10, verbose=True, n_jobs=4)
grid2.fit(train_data, train_target)
print(grid2.best_params_)

########################################################################################################################
# TRAINING E TEST
########################################################################################################################

model = grid.best_estimator_
model.fit(train_data, train_target)
ypred = model.predict(X_test)
cf = confusion_matrix(ytest, ypred)
plt.figure()
sns.heatmap(cf, annot=True, cmap=plt.cm.Reds, cbar=True)
plt.show()
print(classification_report(ytest, ypred))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(ytest, ypred) * 100, 2)) + '%')
print('f1 score nel test set: ' + str(np.round(f1_score(ytest, ypred, average='weighted') * 100, 2)) + '%')

model1 = grid1.best_estimator_
model1.fit(train_data, train_target)
ypred1 = model1.predict(X_test)
cf1 = confusion_matrix(ytest, ypred1)
plt.figure()
sns.heatmap(cf1, annot=True, cmap=plt.cm.Reds, cbar=True)
plt.show()
print "----------------------------------------------------------------------------------------------------------------"
print(classification_report(ytest, ypred1))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(ytest, ypred1) * 100, 2)) + '%')
print('f1 score nel test set: ' + str(np.round(f1_score(ytest, ypred1, average='weighted') * 100, 2)) + '%')

model2 = grid2.best_estimator_
ypred2 = model2.predict(X_test)
cf2 = confusion_matrix(ytest, ypred2)
plt.figure()
sns.heatmap(cf2, annot=True, cmap=plt.cm.Reds, cbar=True)
plt.show()
print "----------------------------------------------------------------------------------------------------------------"
print(classification_report(ytest, ypred2))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(ytest, ypred2) * 100, 2)) + '%')
print('f1 score nel test set: ' + str(np.round(f1_score(ytest, ypred2, average='weighted') * 100, 2)) + '%')
