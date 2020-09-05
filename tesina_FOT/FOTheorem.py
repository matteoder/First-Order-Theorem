# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, \
    balanced_accuracy_score
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# lettura dei dataset
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
    ytrain.append(classe+1)
yval = []
for j in range(1528):
    classe = np.argmax(y_val[j])
    yval.append(classe+1)
ytest = []
for k in range(1529):
    classe = np.argmax(y_test[k])
    ytest.append(classe+1)

all_target = np.concatenate((ytrain, yval, ytest))
# classi sbilanciate
c1=0
c2=0
c3=0
c4=0
c5=0
c6=0
for i in range(6115):
    if all_target[i] == 1:
        c1 = c1+1
    if all_target[i] == 2:
        c2 = c2 + 1
    if all_target[i] == 3:
        c3 = c3 + 1
    if all_target[i] == 4:
        c4 = c4 + 1
    if all_target[i] == 5:
        c5 = c5 + 1
    if all_target[i] == 6:
        c6 = c6 + 1
distrib=[]
distrib.extend([c1,c2,c3,c4,c5,c6])
num=[]
for i in range(6):
    num.append(i)
plt.bar(num, distrib, 0.6, color='goldenrod')
plt.title("distribuzione nelle classi")
plt.xlabel("classi")
plt.ylabel("dati")
plt.show()

# suddivisione del data set in training set e test set
train_data, X_test, train_target, ytest = train_test_split(all_data, all_target, test_size=0.2, random_state=42)
n_train_data = len(train_data)
n_test_data = len(X_test)

#######################################################################################################################
#EDA
#######################################################################################################################

# visualizzazione in due dimensioni delle features in due dimensioni mediante l'utilizzo della PCA
pca = PCA(n_components=2)
X_train = pca.fit(train_data, train_target).transform(X)
X_set = X_train
fig = plt.figure()
ax = fig.gca()
unique = list(set(ytrain))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
colors[2]='hotpink'
colors[5]='limegreen'
for i, u in enumerate(unique):
    xi = [X_set[j,0] for j  in range(len(X_set[:,0])) if ytrain[j] == u]
    yi = [X_set[j,1] for j  in range(len(X_set[:,1])) if ytrain[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.title("PCA con 2 componenti")
plt.xlabel("componente 1")
plt.ylabel("componente 2")
plt.legend()
plt.show()

# conferma che i dati non sono linearmente separabili
linear_svm = SVC(kernel='linear', C=1, decision_function_shape='ovr')
linear_svm.fit(train_data,train_target)
y_pred = linear_svm.predict(train_data)
print(classification_report(train_target, y_pred))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(train_target, y_pred) * 100, 2)) + '%')
print "----------------------------------------------------------------------------------------------------------------"

# rappresentazione grafica della matrice di correlazione; il codice è commentato perchè non è una rappresentazione molto
# informativa, ma per completezza è stata comunque messa

X1 = train_data.transpose()
cr_matrix = np.corrcoef(X1)

# plt.figure(figsize=(120, 100))
# sns.heatmap(cr_matrix, annot=False, cmap=plt.cm.Reds, cbar=True)
# plt.show()

cr = cr_matrix
count = 0
index = []
correlated = []
numeri = []
boolean = True
for i in range(n_features):
    numeri.append(i)
print "selezione delle features altamente correlate"
while boolean:
 for i in range(n_features):
     n_corr=0
     for j in range(n_features):
         if ((cr[i, j] > 0.75 or cr[i, j] < -0.75) and i != j):
             #print(str(i) + '-' + str(j) + 'valore:' + str(cr[i, j]))
             count = count + 1
             n_corr = n_corr + 1
     correlated.append(n_corr)

 plt.bar(numeri, correlated, 0.6, color='limegreen')
 plt.title("numero di variabili altamente correlate")
 plt.xticks(numeri,[])
 plt.xlabel("features")
 plt.ylabel("numero variabili")
 plt.show()
 m=correlated.index(max(correlated))
 if max(correlated) > 2:
    index.append(m)
    print "valore: "+str(max(correlated))+" -- indice: "+str(m)
    for i in range(n_features):
      for j in range(n_features):
          if i==m or j==m:
              cr[i,j] = 0
 if max(correlated) <= 2:
  print "fine dell'operazione"
  boolean = False
 correlated=[]
print "gli indici selezionati sono "+str(index)

variance = np.var(train_data, axis=0)
plt.bar(numeri, variance, 0.6)
plt.title("valori della varianza")
plt.xlabel("features")
plt.ylabel("varianza")
plt.show()


for i in range(n_features):
    for j in range(n_features):
        if ((cr[i, j] > 0.9 or cr[i, j] < -0.9) and i != j):
            print(str(i) + '-' + str(j) + 'valore:' + str(cr[i, j]))
            if variance[i]<variance[j]:
            #if variance[i] > variance[j]:
                index.append(i)
            count = count + 1
# questo indice non viene inserito con l'operazione precedente e viene inserito manualmente
index.append(13)
print "gli indici selezionati sono "+str(index)


#######################################################################################################################
# FEATURE SELECTION
#######################################################################################################################
# OPZIONE 1: nessuna operazione sulle features
all_train_data = train_data

# OPZIONE 2: rimozione delle features ad alto coefficiente di correlazione
# togliere il commento per effettuare la rimozione

all_train_data = np.delete(train_data, index, 1)
X_test = np.delete(X_test, index, 1)

########################################################################################################################
# MODEL SELECTION -- SVM
########################################################################################################################

svm = SVC(kernel='rbf', decision_function_shape='ovr')
param_grid = {'C': [50, 30, 100, 60, 25, 80],
              'gamma': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5,0.001]}

grid = GridSearchCV(svm, param_grid, cv=10, scoring='f1_weighted',n_jobs=4,verbose=True)
grid.fit(all_train_data, train_target)
print "i migliori parametri del SVC sono: "+str((grid.best_params_))

########################################################################################################################
# MODEL SELECTION -- MLPClassifier
########################################################################################################################

mlp = MLPClassifier(solver='sgd', learning_rate_init=0.01, random_state=1)
param_grid1 = {'activation': ['relu', 'tanh'], 'alpha': [1e-2, 1e-4, 1e-5, 1e-3],
               'hidden_layer_sizes': [(100, 30), (100, 10), (100, 20),(80,30)],'early_stopping':[True,False]}

grid1 = GridSearchCV(mlp, param_grid1, cv=10, scoring='f1_weighted',n_jobs=4,verbose=True)
grid1.fit(all_train_data, train_target)
print"i migliori parametri del MLP sono: "+str((grid1.best_params_))

#######################################################################################################################
# MODEL SELECTION -- K-NN
#######################################################################################################################

nbrs = neighbors.KNeighborsClassifier()
parameters = {'n_neighbors': [5, 10, 15, 20, 30], 'p': [1, 2], 'weights': ['distance', 'uniform']}
grid2 = GridSearchCV(nbrs, parameters, scoring='f1_weighted', cv=10,n_jobs=4,verbose=True)
grid2.fit(all_train_data, train_target)
print "i migliori parametri del K-NN sono: "+str((grid2.best_params_))

########################################################################################################################
# TRAINING E TEST
########################################################################################################################
print "----------------------------------------------------------------------------------------------------------------"
print "TEST:"
model = grid.best_estimator_
model.fit(all_train_data, train_target)
ypred = model.predict(X_test)
cf = confusion_matrix(ytest, ypred)
plt.figure()
sns.heatmap(cf, annot=True, cmap=plt.cm.Reds, cbar=True)
plt.show()
print(classification_report(ytest, ypred))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(ytest, ypred) * 100, 2)) + '%')
print('f1 score nel test set: ' + str(np.round(f1_score(ytest,ypred,average='weighted') * 100, 2)) + '%')

model1 = grid1.best_estimator_
model1.fit(all_train_data, train_target)
ypred1 = model1.predict(X_test)
cf1 = confusion_matrix(ytest, ypred1)
plt.figure()
sns.heatmap(cf1, annot=True, cmap=plt.cm.Reds, cbar=True)
plt.show()
print "----------------------------------------------------------------------------------------------------------------"
print(classification_report(ytest, ypred1))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(ytest, ypred1) * 100, 2)) + '%')
print('f1 score nel test set: ' + str(np.round(f1_score(ytest,ypred1,average='weighted')*100,2)) + '%')

model2 = grid2.best_estimator_
ypred2 = model2.predict(X_test)
cf2 = confusion_matrix(ytest, ypred2)
plt.figure()
sns.heatmap(cf2, annot=True, cmap=plt.cm.Reds, cbar=True)
plt.show()
print "----------------------------------------------------------------------------------------------------------------"
print(classification_report(ytest, ypred2))
print('Accuracy nel test set: ' + str(np.round(balanced_accuracy_score(ytest, ypred2) * 100, 2))+'%')
print('f1 score nel test set: ' + str(np.round(f1_score(ytest,ypred2,average='weighted')*100,2)) + '%')