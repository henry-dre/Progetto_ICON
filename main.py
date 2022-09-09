# libreria che permette di gestire dati tabulari in modo facile come ad esempio file exel (il nostro dataset)
import pandas as pd

# libreria usata per facilitare e ottimizzare le operazioni sugli array
import numpy as np

# import modelli di classificazione
from pgmpy.estimators import BicScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel

# import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# import di metrics per calcolare l'accuratezza del modello
from sklearn import metrics
from sklearn.model_selection import KFold

# Librerie per fare inferenza
from pgmpy.inference import VariableElimination

# import del dataset
milkQuality = pd.read_csv('milknew.csv')

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# trasforma in matrice gli elementi del dataframe sostituendo i nomi delle colonne in tutto il dataset con indici vettoriali
Xval = milkQuality.to_numpy()
# fa la stessa cosa di sopra ma solo per la colonna della feture target (operazioni necessarie per lo splitting della k-fold)
Yval = milkQuality["Grade"].to_numpy()

gnbScore = {'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1_list': [],
            }

rfScore = {'accuracy_list': [],
           'precision_list': [],
           'recall_list': [],
           'f1_list': [],
           }

kncScore = {'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1_list': [],
            }

lrScore = {'accuracy_list': [],
           'precision_list': [],
           'recall_list': [],
           'f1_list': [],
           }

dtScore = {'accuracy_list': [],
           'precision_list': [],
           'recall_list': [],
           'f1_list': [],
           }

gnb = GaussianNB()
rfc = RandomForestClassifier()
knc = KNeighborsClassifier()
lr = LogisticRegression(max_iter=3000)
dt = DecisionTreeClassifier()

np.seterr(invalid='ignore')

# ciclo della 5-fold cross validation (che ogni volta va a splittare gli indici delle feature di input in modo diverso)
for trainIndex, testIndex in kfold.split(Xval, Yval):
    # contiene tutte le fold compresa quella di testing
    trainingSet = Xval[trainIndex]
    testSet = Xval[testIndex]

    # Dati di train

    # ritrasforma in dataframe sostituendo agli indici i nomi delle colonne del nostro dataset
    dataTrain = pd.DataFrame(trainingSet, columns=milkQuality.columns)
    # toglie la fold della feature target e...
    X_train = dataTrain.drop("Grade", axis=1)
    # ... la salva in una variabile a parte
    y_train = dataTrain.Grade

    # Dati di test

    # fa la stessa cosa di sopra ma per le fold di test
    dataTest = pd.DataFrame(testSet, columns=milkQuality.columns)
    X_test = dataTest.drop("Grade", axis=1)
    y_test = dataTest.Grade

    # fit (addestramento) del Naive Bayes

    # con fit fa automaticamente tutto il training e restituisce direttamente un modello già trainato
    gnb.fit(X_train, y_train)
    # una volta che ha trainato, va a predire i risultati della y in un vettore contenente tutti i valori predetti di y
    gnbYPredict = gnb.predict(X_test)

    # fit(addestramento) del RandomForest
    rfc.fit(X_train, y_train)
    rfcYPredict = rfc.predict(X_test)

    # fit(addestramento) del KN
    knc.fit(X_train, y_train)
    kncYPredict = knc.predict(X_test)

    # fit(addestramento) del LR
    lr.fit(X_train, y_train)
    lrYPredict = lr.predict(X_test)

    # fit(addestramento) del DTC
    dt.fit(X_train, y_train)
    dtYPredict = dt.predict(X_test)

    # Aggiorno ad ogni ciclo del k-fold le metriche per il classificatore Bayesiano
    gnbScore['accuracy_list'].append(metrics.accuracy_score(y_test, gnbYPredict))
    gnbScore['precision_list'].append(metrics.precision_score(y_test, gnbYPredict, average='macro'))
    gnbScore['recall_list'].append(metrics.recall_score(y_test, gnbYPredict, average='macro'))
    gnbScore['f1_list'].append(metrics.f1_score(y_test, gnbYPredict, average='macro'))

    # Aggiorno ad ogni ciclo del k-fold le metriche per il classificatore RF
    rfScore['accuracy_list'].append(metrics.accuracy_score(y_test, rfcYPredict))
    rfScore['precision_list'].append(metrics.precision_score(y_test, rfcYPredict, average='macro'))
    rfScore['recall_list'].append(metrics.recall_score(y_test, rfcYPredict, average='macro'))
    rfScore['f1_list'].append(metrics.f1_score(y_test, rfcYPredict, average='macro'))

    # Aggiorno ad ogni ciclo del k-fold le metriche per il classificatore KN
    kncScore['accuracy_list'].append(metrics.accuracy_score(y_test, kncYPredict))
    kncScore['precision_list'].append(metrics.precision_score(y_test, kncYPredict, average='macro'))
    kncScore['recall_list'].append(metrics.recall_score(y_test, kncYPredict, average='macro'))
    kncScore['f1_list'].append(metrics.f1_score(y_test, kncYPredict, average='macro'))

    # Aggiorno ad ogni ciclo del k-fold le metriche per il classificatore LR
    lrScore['accuracy_list'].append(metrics.accuracy_score(y_test, lrYPredict))
    lrScore['precision_list'].append(metrics.precision_score(y_test, lrYPredict, average='macro'))
    lrScore['recall_list'].append(metrics.recall_score(y_test, lrYPredict, average='macro'))
    lrScore['f1_list'].append(metrics.f1_score(y_test, lrYPredict, average='macro'))

    # Aggiorno ad ogni ciclo del k-fold le metriche per il classificatore DTC
    dtScore['accuracy_list'].append(metrics.accuracy_score(y_test, dtYPredict))
    dtScore['precision_list'].append(metrics.precision_score(y_test, dtYPredict, average='macro'))
    dtScore['recall_list'].append(metrics.recall_score(y_test, dtYPredict, average='macro'))
    dtScore['f1_list'].append(metrics.f1_score(y_test, dtYPredict, average='macro'))

print('\n'"Media delle metriche del classificatore Bayesiano:")
print("Media Accuracy: %f" % (np.mean(gnbScore['accuracy_list'])))
print("Media Precision: %f" % (np.mean(gnbScore['precision_list'])))
print("Media Recall: %f" % (np.mean(gnbScore['recall_list'])))
print("Media f1: %f" % (np.mean(gnbScore['f1_list'])))

print("\nMedia delle metriche del RandomForest")
print("Media Accuracy: %f" % (np.mean(rfScore['accuracy_list'])))
print("Media Precision: %f" % (np.mean(rfScore['precision_list'])))
print("Media Recall: %f" % (np.mean(rfScore['recall_list'])))
print("Media f1: %f" % (np.mean(rfScore['f1_list'])))

print("\nMedia delle metriche del KN")
print("Media Accuracy: %f" % (np.mean(kncScore['accuracy_list'])))
print("Media Precision: %f" % (np.mean(kncScore['precision_list'])))
print("Media Recall: %f" % (np.mean(kncScore['recall_list'])))
print("Media f1: %f" % (np.mean(kncScore['f1_list'])))

print("\nMedia delle metriche del LR")
print("Media Accuracy: %f" % (np.mean(lrScore['accuracy_list'])))
print("Media Precision: %f" % (np.mean(lrScore['precision_list'])))
print("Media Recall: %f" % (np.mean(lrScore['recall_list'])))
print("Media f1: %f" % (np.mean(lrScore['f1_list'])))

print("\nMedia delle metriche del DTC")
print("Media Accuracy: %f" % (np.mean(dtScore['accuracy_list'])))
print("Media Precision: %f" % (np.mean(dtScore['precision_list'])))
print("Media Recall: %f" % (np.mean(dtScore['recall_list'])))
print("Media f1: %f" % (np.mean(dtScore['f1_list'])))

# creazione classificatore e rete

X_train = milkQuality.drop("Grade", axis=1)
y_train = milkQuality.Grade

# Creazione e fitting del Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# creazione struttura rete
bic = BicScore(milkQuality)
hc_bic = HillClimbSearch(milkQuality)
bic_model = hc_bic.estimate(scoring_method=bic)
# Creazione della rete bayesiana e fit
bNet = BayesianModel(bic_model.edges())
bNet.fit(milkQuality, estimator=MaximumLikelihoodEstimator)

# test funzionamento classificatore Bayesiano semplice

userInfo = {
    "pH": [6],
    "Temprature": [40],
    "Taste": [0],
    "Odor": [0],
    "Fat": [1],
    "Turbidity": [1],
    "Colour": [250],
}
user = pd.DataFrame(userInfo)

# calcola le probabilità che il latte sia buono, discreto o pessimo
print('\n\n\n'"Classificatore Bayesiano - Predizione qualità del latte (alta / bassa / media):"'\n')
print(gnb.predict_proba(user))
# calcolo probabilità con rete bayesiana

print('\n\n\n'"Rete Bayesiana - Esempi vari di predizioni:")
info = VariableElimination(bNet)

quality = info.query(variables=['Grade'],
                     evidence={'pH': 6, 'Temprature': 40, 'Taste': 0, 'Odor': 0, 'Fat': 1, 'Turbidity': 1,
                               'Colour': 250})
print('\n')
print(quality)
print('\n')

quality = info.query(variables=['Turbidity'], evidence={'Taste': 1, 'Odor': 1, 'Colour': 245})
print('\n')
print(quality)
print('\n')

quality = info.query(variables=['Turbidity', 'Taste'], evidence={'pH': 5, 'Odor': 1, 'Temprature': 40})
print('\n')
print(quality)
print('\n')

quality = info.query(variables=['Turbidity', 'Taste', 'Odor'], evidence={'pH': 3, 'Fat': 0})
print('\n')
print(quality)
print('\n')
