# biblioteki
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import sklearn.neighbors
from sklearn.metrics import roc_curve, auc, roc_auc_score, plot_roc_curve
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgbfir
import sklearn.metrics

#wgrywamy dane
data2_1=pd.read_csv('C:/Users/...')
#opcje,  żeby wyswietlało się więcej obesrwacji
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_info_columns', 10000)
#powierzchowne obejrzenie danych 
data2_1.describe()
data2_1.info()

#dziwna zmienna
data2_1=data2_1.drop(["Unnamed: 0"], axis=1)

#Lebel Encoding
le = preprocessing.LabelEncoder()
le.fit(data2_1['Veh13'])
data2_1['Veh13']=le.transform(data2_1['Veh13'])

le1 = preprocessing.LabelEncoder()
le1.fit(data2_1['Veh4'])
data2_1['Veh4']=le1.transform(data2_1['Veh4'])

le2 = preprocessing.LabelEncoder()
le2.fit(data2_1['Veh7'])
data2_1['Veh7']=le2.transform(data2_1['Veh7'])

le3 = preprocessing.LabelEncoder()
le3.fit(data2_1['Veh10'])
data2_1['Veh10']=le3.transform(data2_1['Veh10'])

le4 = preprocessing.LabelEncoder()
le4.fit(data2_1['Veh11'])
data2_1['Veh11']=le4.transform(data2_1['Veh11'])

le5 = preprocessing.LabelEncoder()
le5.fit(data2_1['Veh17'])
data2_1['Veh17']=le5.transform(data2_1['Veh17'])

le6 = preprocessing.LabelEncoder()
le6.fit(data2_1['Veh19'])
data2_1['Veh19']=le6.transform(data2_1['Veh19'])

le7 = preprocessing.LabelEncoder()
le7.fit(data2_1['Veh20'])
data2_1['Veh20']=le7.transform(data2_1['Veh20'])

le8 = preprocessing.LabelEncoder()
le8.fit(data2_1['Reg1'])
data2_1['Reg1']=le8.transform(data2_1['Reg1'])

le9 = preprocessing.LabelEncoder()
le9.fit(data2_1['Reg10'])
data2_1['Reg10']=le9.transform(data2_1['Reg10'])

le10 = preprocessing.LabelEncoder()
le10.fit(data2_1['Reg12'])
data2_1['Reg12']=le10.transform(data2_1['Reg12'])

le11 = preprocessing.LabelEncoder()
le11.fit(data2_1['Dif2'])
data2_1['Dif2']=le11.transform(data2_1['Dif2'])

le12 = preprocessing.LabelEncoder()
le12.fit(data2_1['Per12'])
data2_1['Per12']=le12.transform(data2_1['Per12'])

le13 = preprocessing.LabelEncoder()
le13.fit(data2_1['Reg74'])
data2_1['Reg74']=le13.transform(data2_1['Reg74'])

le14 = preprocessing.LabelEncoder()
le14.fit(data2_1['Reg75'])
data2_1['Reg75']=le14.transform(data2_1['Reg75'])

le15 = preprocessing.LabelEncoder()
le15.fit(data2_1['Reg76'])
data2_1['Reg76']=le15.transform(data2_1['Reg76'])

le16 = preprocessing.LabelEncoder()
le16.fit(data2_1['Reg79'])
data2_1['Reg79']=le16.transform(data2_1['Reg79'])

le17 = preprocessing.LabelEncoder()
le17.fit(data2_1['Reg80'])
data2_1['Reg80']=le17.transform(data2_1['Reg80'])

le18 = preprocessing.LabelEncoder()
le18.fit(data2_1['Reg81'])
data2_1['Reg81']=le18.transform(data2_1['Reg81'])

le19 = preprocessing.LabelEncoder()
le19.fit(data2_1['Hist_Per51'])
data2_1['Hist_Per51']=le19.transform(data2_1['Hist_Per51'])

le20 = preprocessing.LabelEncoder()
le20.fit(data2_1['Hist_Per52'])
data2_1['Hist_Per52']=le20.transform(data2_1['Hist_Per52'])

le21 = preprocessing.LabelEncoder()
le21.fit(data2_1['Hist_Veh6'])
data2_1['Hist_Veh6']=le21.transform(data2_1['Hist_Veh6'])

le22 = preprocessing.LabelEncoder()
le22.fit(data2_1['Hist_Veh7'])
data2_1['Hist_Veh7']=le22.transform(data2_1['Hist_Veh7'])

le23 = preprocessing.LabelEncoder()
le23.fit(data2_1['Hist_VehPer46'])
data2_1['Hist_VehPer46']=le23.transform(data2_1['Hist_VehPer46'])

le24 = preprocessing.LabelEncoder()
le24.fit(data2_1['Hist_VehPer47'])
data2_1['Hist_VehPer47']=le24.transform(data2_1['Hist_VehPer47'])

le25 = preprocessing.LabelEncoder()
le25.fit(data2_1['Veh23'])
data2_1['Veh23']=le25.transform(data2_1['Veh23'])

le26 = preprocessing.LabelEncoder()
le26.fit(data2_1['Veh26'])
data2_1['Veh26']=le26.transform(data2_1['Veh26'])

#rozkład targetu
#dzieki temu mamy lepsze tło na wykresach
sns.set(color_codes=True)

chart=sns.countplot(x=data2_1.target, palette="PuBuGn_d", data=data2_1, order=data2_1.target.value_counts().index)
fig=chart.get_figure()
fig.savefig("C:/Users/...", bbox_inches = "tight")
  
data2_1.target.value_counts()

    
    
    
#przygotowanie zbiorów podstawowych (bez wyboru zmiennych)
#podział na train i test
columny_50=['Per2',	'Veh24',	'Veh3',	'Hist_VehPer47',	'Hist_Per103',	'Reg78',	'Reg77',	'Reg83',	'Hist_VehPer24',	'Per8',	'Dif1',	'Hist_VehPer7',	'Hist_Per6',	'Reg7',	'Hist_Per52',	'Reg58',	'Reg41',	'Per7',	'Hist_Veh29',	'Reg9',	'Hist_Veh22',	'Hist_Veh7',	'Reg28',	'Reg82',	'Reg11',	'Veh16',	'Veh5',	'Reg8',	'Reg13',	'Veh6',	'Reg62',	'Reg36',	'Per12',	'Reg40',	'Veh1',	'Reg15',	'Hist_Per100',	'Veh20',	'Veh8',	'Veh21',	'Hist_Veh3',	'Reg5',	'Veh9',	'Hist_Per55',	'Veh27',	'Reg75',	'Reg81',	'Reg38',	'Veh22',	'Reg3','target']
data2_1_2=data2_1[columny_50]

data2_1_train, data2_1_test=train_test_split(data2_1_2, test_size=0.3, random_state=1, stratify=data2_1_2.target)
#podział na X i y
X_train=data2_1_train[[col for col in data2_1_train.columns if col!="target"]]
y_train=data2_1_train.target

X_test=data2_1_test[[col for col in data2_1_test.columns if col!="target"]]
y_test=data2_1_test.target

#Drzewa decyzyjne i Lasy losowe bez ustawien
model1=DecisionTreeClassifier()
model2=RandomForestClassifier()
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

y_pred_ucz1 = model1.predict(X_train)
y_pred_test1 = model1.predict(X_test)

y_pred_ucz2 = model2.predict(X_train)
y_pred_test2 = model2.predict(X_test)

plot_roc_curve(model1, X_train, y_train)
plot_roc_curve(model1, X_test, y_test)
plot_roc_curve(model2, X_train, y_train)
plot_roc_curve(model2, X_test, y_test)


sklearn.metrics.confusion_matrix(y_train,y_pred_ucz1)
sklearn.metrics.confusion_matrix(y_test,y_pred_test1)
sklearn.metrics.confusion_matrix(y_train,y_pred_ucz2)
sklearn.metrics.confusion_matrix(y_test,y_pred_test2)


def fit_classifier(alg, X_train, X_test, y_train, y_test):
    alg.fit(X_train, y_train)
    y_pred_ucz = alg.predict(X_train)
    y_pred_test = alg.predict(X_test)
    return {
        "ACC_ucz": sklearn.metrics.accuracy_score(y_pred_ucz, y_train),
        "ACC_test": sklearn.metrics.accuracy_score(y_pred_test, y_test),
        "P_ucz":   sklearn.metrics.precision_score(y_pred_ucz, y_train),
        "P_test":   sklearn.metrics.precision_score(y_pred_test, y_test),
        "R_ucz":   sklearn.metrics.recall_score(y_pred_ucz, y_train),
        "R_test":   sklearn.metrics.recall_score(y_pred_test, y_test),
        "F1_ucz":  sklearn.metrics.f1_score(y_pred_ucz, y_train),
        "F1_test":  sklearn.metrics.f1_score(y_pred_test, y_test)
    }
    
results = pd.DataFrame({'DTree': fit_classifier(DecisionTreeClassifier(), X_train, X_test, y_train, y_test)}).T 
results = results.append(pd.DataFrame({'RForest': fit_classifier(RandomForestClassifier(), X_train, X_test, y_train, y_test)}).T )

results   


#Drzewa decyzyjne:
tab_train = list()
tab_test = list()

for i in range(3,30):
    model = DecisionTreeClassifier(max_depth=i) #tworzenie modelu
    print(model)
    model.fit(X_train, y_train) #trenowanie modelu
    Y_train_class = model.predict(X_train) #estymacja zmiennej celu dla zbioru testowego
    Y_test_class = model.predict(X_test)
    tab_train.append(sklearn.metrics.roc_auc_score(Y_train_class, y_train))
    tab_test.append(sklearn.metrics.roc_auc_score(Y_test_class, y_test))


plt.figure(figsize=(10,6))
plt.plot(tab_train)
plt.plot(tab_test)
plt.show()

#Best
model_DT_FIN=DecisionTreeClassifier(max_depth=10)
model_DT_FIN.fit(X_train, y_train)
y_pred_ucz_DT_FIN = model_DT_FIN.predict(X_train)
y_pred_test_DT_FIN = model_DT_FIN.predict(X_test)

plot_roc_curve(model_DT_FIN, X_train, y_train)
plot_roc_curve(model_DT_FIN, X_test, y_test)

sklearn.metrics.confusion_matrix(y_train,y_pred_ucz_DT_FIN)
sklearn.metrics.confusion_matrix(y_test,y_pred_test_DT_FIN)

#"ACC_ucz": 
sklearn.metrics.accuracy_score(y_pred_ucz_DT_FIN, y_train),
#"ACC_test": 
sklearn.metrics.accuracy_score(y_pred_test_DT_FIN, y_test),
#"P_ucz":   
sklearn.metrics.precision_score(y_pred_ucz_DT_FIN, y_train),
#"P_test":   
sklearn.metrics.precision_score(y_pred_test_DT_FIN, y_test),
#"R_ucz":   
sklearn.metrics.recall_score(y_pred_ucz_DT_FIN, y_train),
#"R_test":   
sklearn.metrics.recall_score(y_pred_test_DT_FIN, y_test),
#"F1_ucz":  
sklearn.metrics.f1_score(y_pred_ucz_DT_FIN, y_train),
#"F1_test":  
sklearn.metrics.f1_score(y_pred_test_DT_FIN, y_test)



#Las losowy:
tab_train = list()
tab_test = list()

for i in range(15,25):
    model = RandomForestClassifier(max_depth=i) #tworzenie modelu
    print(model)
    model.fit(X_train, y_train) #trenowanie modelu
    Y_train_class = model.predict(X_train) #estymacja zmiennej celu dla zbioru testowego
    Y_test_class = model.predict(X_test)
    tab_train.append(sklearn.metrics.roc_auc_score(Y_train_class, y_train))
    tab_test.append(sklearn.metrics.roc_auc_score(Y_test_class, y_test))


plt.figure(figsize=(10,6))
plt.plot(tab_train)
plt.plot(tab_test)
plt.show()

#Grid Search 
param_RF = {'n_estimators':[100,200,300,400,500],
            'max_depth':[8,12,16,20,24,28, None],
            'min_samples_split':[2,5],
            'min_samples_leaf':[1,5],
            'max_features': ['auto', 'sqrt', 'log2']
}


gsearch_RF = GridSearchCV(estimator = model2,
                        param_grid = param_RF, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3)


gsearch_RF2=RandomizedSearchCV(estimator=model2, 
                   param_distributions=param_RF, 
                   n_iter=10, 
                   scoring='roc_auc', 
                   n_jobs=None, 
                   iid=False, 
                   cv=3
)

gsearch_RF2.fit(X_train, y_train)
gsearch_RF2.best_params_
gsearch_RF2.best_score_


model_RF_FIN=RandomForestClassifier(n_estimators=400,
                                    min_samples_split=2,
                                    min_samples_leaf=5,
                                    max_features='log2',
                                    max_depth=None)
model_RF_FIN.fit(X_train, y_train)
y_pred_ucz_RF_FIN = model_RF_FIN.predict(X_train)
y_pred_test_RF_FIN = model_RF_FIN.predict(X_test)

plot_roc_curve(model_RF_FIN, X_train, y_train)
plot_roc_curve(model_RF_FIN, X_test, y_test)

sklearn.metrics.confusion_matrix(y_train,y_pred_ucz_RF_FIN)
sklearn.metrics.confusion_matrix(y_test,y_pred_test_RF_FIN)

#"ACC_ucz": 
sklearn.metrics.accuracy_score(y_pred_ucz_RF_FIN, y_train),
#"ACC_test": 
sklearn.metrics.accuracy_score(y_pred_test_RF_FIN, y_test),
#"P_ucz":   
sklearn.metrics.precision_score(y_pred_ucz_RF_FIN, y_train),
#"P_test":   
sklearn.metrics.precision_score(y_pred_test_RF_FIN, y_test),
#"R_ucz":   
sklearn.metrics.recall_score(y_pred_ucz_RF_FIN, y_train),
#"R_test":   
sklearn.metrics.recall_score(y_pred_test_RF_FIN, y_test),
#"F1_ucz":  
sklearn.metrics.f1_score(y_pred_ucz_RF_FIN, y_train),
#"F1_test":  
sklearn.metrics.f1_score(y_pred_test_RF_FIN, y_test)



#XGBoosting
xgb0 = XGBClassifier()
xgb0.fit(X_train, y_train)
y_pred_ucz_XGB_0 = xgb0.predict(X_train)
y_pred_test_XGB_0 = xgb0.predict(X_test)

plot_roc_curve(xgb0, X_train, y_train)
plot_roc_curve(xgb0, X_test, y_test)

sklearn.metrics.confusion_matrix(y_train,y_pred_ucz_XGB_0)
sklearn.metrics.confusion_matrix(y_test,y_pred_test_XGB_0)

#"ACC_ucz": 
sklearn.metrics.accuracy_score(y_pred_ucz_XGB_0, y_train),
#"ACC_test": 
sklearn.metrics.accuracy_score(y_pred_test_XGB_0, y_test),
#"P_ucz":   
sklearn.metrics.precision_score(y_pred_ucz_XGB_0, y_train),
#"P_test":   
sklearn.metrics.precision_score(y_pred_test_XGB_0, y_test),
#"R_ucz":   
sklearn.metrics.recall_score(y_pred_ucz_XGB_0, y_train),
#"R_test":   
sklearn.metrics.recall_score(y_pred_test_XGB_0, y_test),
#"F1_ucz":  
sklearn.metrics.f1_score(y_pred_ucz_XGB_0, y_train),
#"F1_test":  
sklearn.metrics.f1_score(y_pred_test_XGB_0, y_test)


#stopniowy Grid Searh dla XGBoost
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)

def wynik_modelu(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='auc', eval_set=[(X_test, y_test)])
    y_test_pred=model.predict_proba(X_test)[:, 1]
    y_train_pred=model.predict_proba(X_train)[:, 1]
    print('ROC AUC TRAIN: %f' % sklearn.metrics.roc_auc_score(y_train, y_train_pred))
    print('ROC AUC TEST: %f' % sklearn.metrics.roc_auc_score(y_test, y_test_pred))
    

wynik_modelu(xgb1, X_train, y_train, X_test, y_test)

xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=556,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)


param_test1 = {'max_depth':[2,3,6,8],
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(estimator = xgb2,
                        param_grid = param_test1, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3)



gsearch1.fit(X_train, y_train)

gsearch1.best_score_
gsearch1.best_params_
gsearch1.score
gsearch1.scorer_
gsearch1.cv_results_


param_test2 = {'max_depth':[4,5,6,7],
 'min_child_weight': [4,5,6]
}


gsearch2 = GridSearchCV(estimator = xgb2,
                        param_grid = param_test2, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3)

gsearch2.fit(X_train, y_train)

gsearch2.best_score_
gsearch2.best_params_
gsearch2.score
gsearch2.scorer_
gsearch2.cv_results_

param_test3 = {'max_depth':[2,3,4,5],
 'min_child_weight': [2,3,4,5]
}


gsearch3 = GridSearchCV(estimator = xgb2,
                        param_grid = param_test3, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3, 
                        verbose=10)

gsearch3.fit(X_train, y_train)

gsearch3.best_score_
gsearch3.best_params_
gsearch3.score
gsearch3.scorer_
gsearch3.cv_results_

xgb3 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=4,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)

wynik_modelu(xgb3, X_train, y_train, X_test, y_test)

xgb4 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=770,
        max_depth=4,
        min_child_weight=4,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)

param_test4 = {'gamma':[0,0.1,0.2,0.3,0.4,0.5]}

gsearch4 = GridSearchCV(estimator = xgb4,
                        param_grid = param_test4, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3, 
                        verbose=10)

gsearch4.fit(X_train, y_train)

gsearch4.best_score_
gsearch4.best_params_
gsearch4.score
gsearch4.scorer_
gsearch4.cv_results_

param_test5 = {'gamma':[0.15,0.2,0.25]}

gsearch5 = GridSearchCV(estimator = xgb4,
                        param_grid = param_test5, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3, 
                        verbose=10)

gsearch5.fit(X_train, y_train)

gsearch5.best_score_
gsearch5.best_params_
gsearch5.score
gsearch5.scorer_
gsearch5.cv_results_


xgb5 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)

wynik_modelu(xgb5, X_train, y_train, X_test, y_test)


xgb6 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=562,
        max_depth=4,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)

param_test6 = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]}

gsearch6 = GridSearchCV(estimator = xgb6,
                        param_grid = param_test6, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3, 
                        verbose=10)

gsearch6.fit(X_train, y_train)

gsearch6.best_score_
gsearch6.best_params_
gsearch6.score
gsearch6.scorer_
gsearch6.cv_results_


param_test7 = {
        'subsample':[0.75,0.8,0.85],
        'colsample_bytree':[0.75,0.8,0.85]}

gsearch7 = GridSearchCV(estimator = xgb6,
                        param_grid = param_test7, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3, 
                        verbose=10)

gsearch7.fit(X_train, y_train)

gsearch7.best_score_
gsearch7.best_params_
gsearch7.score
gsearch7.scorer_
gsearch7.cv_results_


param_test8 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch8 = GridSearchCV(estimator = xgb6,
                        param_grid = param_test8, 
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False, 
                        cv=3, 
                        verbose=10)

gsearch8.fit(X_train, y_train)

gsearch8.best_score_
gsearch8.best_params_

xgb7 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,#562
        max_depth=4,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)

wynik_modelu(xgb7, X_train, y_train, X_test, y_test)
#ROC AUC TRAIN: 0.796202
#ROC AUC TEST: 0.759881

xgb8 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=5000,#ALL
        max_depth=4,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)
wynik_modelu(xgb8, X_train, y_train, X_test, y_test)

#ROC AUC TRAIN: 0.793948
#ROC AUC TEST: 0.760656


xgb9 = XGBClassifier(
        learning_rate =0.05,
        n_estimators=5000,#1322
        max_depth=4,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)
wynik_modelu(xgb9, X_train, y_train, X_test, y_test)
#ROC AUC TRAIN: 0.803264
#ROC AUC TEST: 0.761147

xgb10 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=10000,#7320
        max_depth=4,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)
wynik_modelu(xgb10, X_train, y_train, X_test, y_test)
#ROC AUC TRAIN: 0.808403
#ROC AUC TEST: 0.762246


#najlepszy modedel xgb8


data2_1_train_ALL, data2_1_test_ALL=train_test_split(data2_1, test_size=0.3, random_state=1, stratify=data2_1.target)

X_train_ALL=data2_1_train_ALL[[col for col in data2_1_train_ALL.columns if col!="target"]]
y_train_ALL=data2_1_train_ALL.target

X_test_ALL=data2_1_test_ALL[[col for col in data2_1_test_ALL.columns if col!="target"]]
y_test_ALL=data2_1_test_ALL.target

xgb8.fit(X_train_ALL, y_train_ALL)

#ważnosć cech
feat_labels = X_train_ALL.columns
importances = xgb8.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train_ALL.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


plt.title('Istotność cech')
plt.bar(range(X_train_ALL.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train_ALL.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train_ALL.shape[1]])
plt.tight_layout()
#plt.savefig('rysunki/04_09.png', dpi=300)
plt.show()



plt.bar(range(len(xgb8.feature_importances_)), xgb8.feature_importances_)
plt.show()




y_test_pred=xgb8.predict_proba(X_test_ALL)[:, 1]
y_train_pred=xgb8.predict_proba(X_train_ALL)[:, 1]
print('ROC AUC TRAIN: %f' % sklearn.metrics.roc_auc_score(y_train_ALL, y_train_pred))#ROC AUC TRAIN: 0.803086
print('ROC AUC TEST: %f' % sklearn.metrics.roc_auc_score(y_test_ALL, y_test_pred))#ROC AUC TEST: 0.764920
xgbfir.saveXgbFI(xgb8, feature_names=X_train_ALL.columns, OutputXlsxFile = 'C:/Users/...') 


columny_100=['Per2',	'Veh24',	'Hist_VehPer47',	'Veh3',	'Hist_Per6',	'Hist_Veh7',	'Hist_VehPer7',	'Per7',	'Reg78',	'Reg41',	'Per8',	'Hist_VehPer24',	'Hist_Veh3',	'Hist_Per52',	'Hist_Per100',	'Hist_VehPer41',	'Veh20',	'Hist_Veh29',	'Hist_Veh22',	'Hist_VehPer81',	'Hist_Per44',	'Reg58',	'Hist_VehPer46',	'Hist_VehPer52',	'Hist_Veh4',	'Hist_VehPer82',	'Hist_VehPer74',	'Dif3',	'Hist_Per63',	'Hist_Per109',	'Per12',	'Hist_Per111',	'Reg81',	'Hist_Veh8',	'Dif1',	'Hist_Per118',	'Hist_VehPer71',	'Hist_VehPer54',	'Reg47',	'Hist_Per103',	'Reg83',	'Dif2',	'Hist_VehPer60',	'Reg77',	'Veh5',	'Reg61',	'Hist_VehPer43',	'Hist_Per51',	'Hist_Per48',	'Reg39',	'Hist_Per69',	'Reg48',	'Reg15',	'Veh18',	'Veh23',	'Hist_Per35',	'Hist_VehPer25',	'Veh17',	'Reg34',	'Reg6',	'Reg82',	'Veh25',	'Hist_Per97',	'Hist_Per28',	'Reg38',	'Reg7',	'Veh22',	'Hist_Per106',	'Reg26',	'Veh8',	'Hist_Per50',	'Reg44',	'Reg57',	'Hist_VehPer59',	'Reg63',	'Reg29',	'Reg72',	'Reg13',	'Veh21',	'Hist_Per46',	'Reg20',	'Veh6',	'Reg22',	'Reg70',	'Hist_Per127',	'Reg18',	'Reg3',	'Reg68',	'Hist_Veh24',	'Hist_Per57',	'Reg24',	'Reg11',	'Reg8',	'Reg71',	'Hist_Per131',	'Reg32',	'Hist_Per41',	'Reg53',	'Reg42',	'Reg28']
columny_50=['Per2',	'Veh24',	'Hist_VehPer47',	'Veh3',	'Hist_Per6',	'Hist_Veh7',	'Hist_VehPer7',	'Per7',	'Reg78',	'Reg41',	'Per8',	'Hist_VehPer24',	'Hist_Veh3',	'Hist_Per52',	'Hist_Per100',	'Hist_VehPer41',	'Veh20',	'Hist_Veh29',	'Hist_Veh22',	'Hist_VehPer81',	'Hist_Per44',	'Reg58',	'Hist_VehPer46',	'Hist_VehPer52',	'Hist_Veh4',	'Hist_VehPer82',	'Hist_VehPer74',	'Dif3',	'Hist_Per63',	'Hist_Per109',	'Per12',	'Hist_Per111',	'Reg81',	'Hist_Veh8',	'Dif1',	'Hist_Per118',	'Hist_VehPer71',	'Hist_VehPer54',	'Reg47',	'Hist_Per103',	'Reg83',	'Dif2',	'Hist_VehPer60',	'Reg77',	'Veh5',	'Reg61',	'Hist_VehPer43',	'Hist_Per51']


X_train_100=X_train_ALL[columny_100]
X_test_100=X_test_ALL[columny_100]

X_train_50=X_train_ALL[columny_50]
X_test_50=X_test_ALL[columny_50]

wynik_all=wynik_modelu(xgb8, X_train_ALL, y_train_ALL, X_test_ALL, y_test_ALL)
#ROC AUC TRAIN: 0.803064
#ROC AUC TEST: 0.764922
wynik_100=wynik_modelu(xgb8, X_train_100, y_train_ALL, X_test_100, y_test_ALL)
#ROC AUC TRAIN: 0.796273
#ROC AUC TEST: 0.762577
wynik_50=wynik_modelu(xgb8, X_train_50, y_train_ALL, X_test_50, y_test_ALL)
#ROC AUC TRAIN: 0.786744
#ROC AUC TEST: 0.759549



#Cross walidacja
data2_2=pd.read_csv('C:/Users/...')



#Lebel Encoding
le_cw = preprocessing.LabelEncoder()
le_cw.fit(data2_2['Veh13'])
data2_2['Veh13']=le_cw.transform(data2_2['Veh13'])

le_cw1 = preprocessing.LabelEncoder()
le_cw1.fit(data2_2['Veh4'])
data2_2['Veh4']=le_cw1.transform(data2_2['Veh4'])

le_cw2 = preprocessing.LabelEncoder()
le_cw2.fit(data2_2['Veh7'])
data2_2['Veh7']=le_cw2.transform(data2_2['Veh7'])

le_cw3 = preprocessing.LabelEncoder()
le_cw3.fit(data2_2['Veh10'])
data2_2['Veh10']=le_cw3.transform(data2_2['Veh10'])

le_cw4 = preprocessing.LabelEncoder()
le_cw4.fit(data2_2['Veh11'])
data2_2['Veh11']=le_cw4.transform(data2_2['Veh11'])

le_cw5 = preprocessing.LabelEncoder()
le_cw5.fit(data2_2['Veh17'])
data2_2['Veh17']=le_cw5.transform(data2_2['Veh17'])

le_cw6 = preprocessing.LabelEncoder()
le_cw6.fit(data2_2['Veh19'])
data2_2['Veh19']=le_cw6.transform(data2_2['Veh19'])

le_cw7 = preprocessing.LabelEncoder()
le_cw7.fit(data2_2['Veh20'])
data2_2['Veh20']=le_cw7.transform(data2_2['Veh20'])

le_cw8 = preprocessing.LabelEncoder()
le_cw8.fit(data2_2['Reg1'])
data2_2['Reg1']=le_cw8.transform(data2_2['Reg1'])

le_cw9 = preprocessing.LabelEncoder()
le_cw9.fit(data2_2['Reg10'])
data2_2['Reg10']=le_cw9.transform(data2_2['Reg10'])

le_cw10 = preprocessing.LabelEncoder()
le_cw10.fit(data2_2['Reg12'])
data2_2['Reg12']=le_cw10.transform(data2_2['Reg12'])

le_cw11 = preprocessing.LabelEncoder()
le_cw11.fit(data2_2['Dif2'])
data2_2['Dif2']=le_cw11.transform(data2_2['Dif2'])

le_cw12 = preprocessing.LabelEncoder()
le_cw12.fit(data2_2['Per12'])
data2_2['Per12']=le_cw12.transform(data2_2['Per12'])

le_cw13 = preprocessing.LabelEncoder()
le_cw13.fit(data2_2['Reg74'])
data2_2['Reg74']=le_cw13.transform(data2_2['Reg74'])

le_cw14 = preprocessing.LabelEncoder()
le_cw14.fit(data2_2['Reg75'])
data2_2['Reg75']=le_cw14.transform(data2_2['Reg75'])

le_cw15 = preprocessing.LabelEncoder()
le_cw15.fit(data2_2['Reg76'])
data2_2['Reg76']=le_cw15.transform(data2_2['Reg76'])

le_cw16 = preprocessing.LabelEncoder()
le_cw16.fit(data2_2['Reg79'])
data2_2['Reg79']=le_cw16.transform(data2_2['Reg79'])

le_cw17 = preprocessing.LabelEncoder()
le_cw17.fit(data2_2['Reg80'])
data2_2['Reg80']=le_cw17.transform(data2_2['Reg80'])

le_cw18 = preprocessing.LabelEncoder()
le_cw18.fit(data2_2['Reg81'])
data2_2['Reg81']=le_cw18.transform(data2_2['Reg81'])

le_cw19 = preprocessing.LabelEncoder()
le_cw19.fit(data2_2['Hist_Per51'])
data2_2['Hist_Per51']=le_cw19.transform(data2_2['Hist_Per51'])

le_cw20 = preprocessing.LabelEncoder()
le_cw20.fit(data2_2['Hist_Per52'])
data2_2['Hist_Per52']=le_cw20.transform(data2_2['Hist_Per52'])

le_cw21 = preprocessing.LabelEncoder()
le_cw21.fit(data2_2['Hist_Veh6'])
data2_2['Hist_Veh6']=le_cw21.transform(data2_2['Hist_Veh6'])

le_cw22 = preprocessing.LabelEncoder()
le_cw22.fit(data2_2['Hist_Veh7'])
data2_2['Hist_Veh7']=le_cw22.transform(data2_2['Hist_Veh7'])

le_cw23 = preprocessing.LabelEncoder()
le_cw23.fit(data2_2['Hist_VehPer46'])
data2_2['Hist_VehPer46']=le_cw23.transform(data2_2['Hist_VehPer46'])

le_cw24 = preprocessing.LabelEncoder()
le_cw24.fit(data2_2['Hist_VehPer47'])
data2_2['Hist_VehPer47']=le_cw24.transform(data2_2['Hist_VehPer47'])

le_cw25 = preprocessing.LabelEncoder()
le_cw25.fit(data2_2['Veh23'])
data2_2['Veh23']=le_cw25.transform(data2_2['Veh23'])

le_cw26 = preprocessing.LabelEncoder()
le_cw26.fit(data2_2['Veh26'])
data2_2['Veh26']=le_cw26.transform(data2_2['Veh26'])


data2_2_train, data2_2_test=train_test_split(data2_2, test_size=0.3, random_state=1, stratify=data2_2.target)

X_train_2=data2_2_train[[col for col in data2_2_train.columns if col!="target"]]
y_train_2=data2_2_train.target
X_test_2=data2_2_test[[col for col in data2_2_test.columns if col!="target"]]
y_test_2=data2_2_test.target

X_train_50_2=X_train_2[columny_50]
X_test_50_2=X_test_2[columny_50]


wynik_50_2=wynik_modelu(xgb8, X_train_50_2, y_train_2, X_test_50_2, y_test_2)
#ROC AUC TRAIN: 0.786857
#ROC AUC TEST: 0.760616



data2_3=pd.read_csv('C:/Users...')



#Lebel Encoding
le_cw = preprocessing.LabelEncoder()
le_cw.fit(data2_3['Veh13'])
data2_3['Veh13']=le_cw.transform(data2_3['Veh13'])

le_cw1 = preprocessing.LabelEncoder()
le_cw1.fit(data2_3['Veh4'])
data2_3['Veh4']=le_cw1.transform(data2_3['Veh4'])

le_cw2 = preprocessing.LabelEncoder()
le_cw2.fit(data2_3['Veh7'])
data2_3['Veh7']=le_cw2.transform(data2_3['Veh7'])

le_cw3 = preprocessing.LabelEncoder()
le_cw3.fit(data2_3['Veh10'])
data2_3['Veh10']=le_cw3.transform(data2_3['Veh10'])

le_cw4 = preprocessing.LabelEncoder()
le_cw4.fit(data2_3['Veh11'])
data2_3['Veh11']=le_cw4.transform(data2_3['Veh11'])

le_cw5 = preprocessing.LabelEncoder()
le_cw5.fit(data2_3['Veh17'])
data2_3['Veh17']=le_cw5.transform(data2_3['Veh17'])

le_cw6 = preprocessing.LabelEncoder()
le_cw6.fit(data2_3['Veh19'])
data2_3['Veh19']=le_cw6.transform(data2_3['Veh19'])

le_cw7 = preprocessing.LabelEncoder()
le_cw7.fit(data2_3['Veh20'])
data2_3['Veh20']=le_cw7.transform(data2_3['Veh20'])

le_cw8 = preprocessing.LabelEncoder()
le_cw8.fit(data2_3['Reg1'])
data2_3['Reg1']=le_cw8.transform(data2_3['Reg1'])

le_cw9 = preprocessing.LabelEncoder()
le_cw9.fit(data2_3['Reg10'])
data2_3['Reg10']=le_cw9.transform(data2_3['Reg10'])

le_cw10 = preprocessing.LabelEncoder()
le_cw10.fit(data2_3['Reg12'])
data2_3['Reg12']=le_cw10.transform(data2_3['Reg12'])

le_cw11 = preprocessing.LabelEncoder()
le_cw11.fit(data2_3['Dif2'])
data2_3['Dif2']=le_cw11.transform(data2_3['Dif2'])

le_cw12 = preprocessing.LabelEncoder()
le_cw12.fit(data2_3['Per12'])
data2_3['Per12']=le_cw12.transform(data2_3['Per12'])

le_cw13 = preprocessing.LabelEncoder()
le_cw13.fit(data2_3['Reg74'])
data2_3['Reg74']=le_cw13.transform(data2_3['Reg74'])

le_cw14 = preprocessing.LabelEncoder()
le_cw14.fit(data2_3['Reg75'])
data2_3['Reg75']=le_cw14.transform(data2_3['Reg75'])

le_cw15 = preprocessing.LabelEncoder()
le_cw15.fit(data2_3['Reg76'])
data2_3['Reg76']=le_cw15.transform(data2_3['Reg76'])

le_cw16 = preprocessing.LabelEncoder()
le_cw16.fit(data2_3['Reg79'])
data2_3['Reg79']=le_cw16.transform(data2_3['Reg79'])

le_cw17 = preprocessing.LabelEncoder()
le_cw17.fit(data2_3['Reg80'])
data2_3['Reg80']=le_cw17.transform(data2_3['Reg80'])

le_cw18 = preprocessing.LabelEncoder()
le_cw18.fit(data2_3['Reg81'])
data2_3['Reg81']=le_cw18.transform(data2_3['Reg81'])

le_cw19 = preprocessing.LabelEncoder()
le_cw19.fit(data2_3['Hist_Per51'])
data2_3['Hist_Per51']=le_cw19.transform(data2_3['Hist_Per51'])

le_cw20 = preprocessing.LabelEncoder()
le_cw20.fit(data2_3['Hist_Per52'])
data2_3['Hist_Per52']=le_cw20.transform(data2_3['Hist_Per52'])

le_cw21 = preprocessing.LabelEncoder()
le_cw21.fit(data2_3['Hist_Veh6'])
data2_3['Hist_Veh6']=le_cw21.transform(data2_3['Hist_Veh6'])

le_cw22 = preprocessing.LabelEncoder()
le_cw22.fit(data2_3['Hist_Veh7'])
data2_3['Hist_Veh7']=le_cw22.transform(data2_3['Hist_Veh7'])

le_cw23 = preprocessing.LabelEncoder()
le_cw23.fit(data2_3['Hist_VehPer46'])
data2_3['Hist_VehPer46']=le_cw23.transform(data2_3['Hist_VehPer46'])

le_cw24 = preprocessing.LabelEncoder()
le_cw24.fit(data2_3['Hist_VehPer47'])
data2_3['Hist_VehPer47']=le_cw24.transform(data2_3['Hist_VehPer47'])

le_cw25 = preprocessing.LabelEncoder()
le_cw25.fit(data2_3['Veh23'])
data2_3['Veh23']=le_cw25.transform(data2_3['Veh23'])

le_cw26 = preprocessing.LabelEncoder()
le_cw26.fit(data2_3['Veh26'])
data2_3['Veh26']=le_cw26.transform(data2_3['Veh26'])


data2_3_train, data2_3_test=train_test_split(data2_3, test_size=0.3, random_state=1, stratify=data2_3.target)

X_train_3=data2_3_train[[col for col in data2_3_train.columns if col!="target"]]
y_train_3=data2_3_train.target
X_test_3=data2_3_test[[col for col in data2_3_test.columns if col!="target"]]
y_test_3=data2_3_test.target

X_train_50_3=X_train_3[columny_50]
X_test_50_3=X_test_3[columny_50]


wynik_50_3=wynik_modelu(xgb8, X_train_50_3, y_train_3, X_test_50_3, y_test_3)
#ROC AUC TRAIN: 0.786650
#ROC AUC TEST: 0.762592


data2_4=pd.read_csv('C:/Users/...')



#Lebel Encoding
le_cw = preprocessing.LabelEncoder()
le_cw.fit(data2_4['Veh13'])
data2_4['Veh13']=le_cw.transform(data2_4['Veh13'])

le_cw1 = preprocessing.LabelEncoder()
le_cw1.fit(data2_4['Veh4'])
data2_4['Veh4']=le_cw1.transform(data2_4['Veh4'])

le_cw2 = preprocessing.LabelEncoder()
le_cw2.fit(data2_4['Veh7'])
data2_4['Veh7']=le_cw2.transform(data2_4['Veh7'])

le_cw3 = preprocessing.LabelEncoder()
le_cw3.fit(data2_4['Veh10'])
data2_4['Veh10']=le_cw3.transform(data2_4['Veh10'])

le_cw4 = preprocessing.LabelEncoder()
le_cw4.fit(data2_4['Veh11'])
data2_4['Veh11']=le_cw4.transform(data2_4['Veh11'])

le_cw5 = preprocessing.LabelEncoder()
le_cw5.fit(data2_4['Veh17'])
data2_4['Veh17']=le_cw5.transform(data2_4['Veh17'])

le_cw6 = preprocessing.LabelEncoder()
le_cw6.fit(data2_4['Veh19'])
data2_4['Veh19']=le_cw6.transform(data2_4['Veh19'])

le_cw7 = preprocessing.LabelEncoder()
le_cw7.fit(data2_4['Veh20'])
data2_4['Veh20']=le_cw7.transform(data2_4['Veh20'])

le_cw8 = preprocessing.LabelEncoder()
le_cw8.fit(data2_4['Reg1'])
data2_4['Reg1']=le_cw8.transform(data2_4['Reg1'])

le_cw9 = preprocessing.LabelEncoder()
le_cw9.fit(data2_4['Reg10'])
data2_4['Reg10']=le_cw9.transform(data2_4['Reg10'])

le_cw10 = preprocessing.LabelEncoder()
le_cw10.fit(data2_4['Reg12'])
data2_4['Reg12']=le_cw10.transform(data2_4['Reg12'])

le_cw11 = preprocessing.LabelEncoder()
le_cw11.fit(data2_4['Dif2'])
data2_4['Dif2']=le_cw11.transform(data2_4['Dif2'])

le_cw12 = preprocessing.LabelEncoder()
le_cw12.fit(data2_4['Per12'])
data2_4['Per12']=le_cw12.transform(data2_4['Per12'])

le_cw13 = preprocessing.LabelEncoder()
le_cw13.fit(data2_4['Reg74'])
data2_4['Reg74']=le_cw13.transform(data2_4['Reg74'])

le_cw14 = preprocessing.LabelEncoder()
le_cw14.fit(data2_4['Reg75'])
data2_4['Reg75']=le_cw14.transform(data2_4['Reg75'])

le_cw15 = preprocessing.LabelEncoder()
le_cw15.fit(data2_4['Reg76'])
data2_4['Reg76']=le_cw15.transform(data2_4['Reg76'])

le_cw16 = preprocessing.LabelEncoder()
le_cw16.fit(data2_4['Reg79'])
data2_4['Reg79']=le_cw16.transform(data2_4['Reg79'])

le_cw17 = preprocessing.LabelEncoder()
le_cw17.fit(data2_4['Reg80'])
data2_4['Reg80']=le_cw17.transform(data2_4['Reg80'])

le_cw18 = preprocessing.LabelEncoder()
le_cw18.fit(data2_4['Reg81'])
data2_4['Reg81']=le_cw18.transform(data2_4['Reg81'])

le_cw19 = preprocessing.LabelEncoder()
le_cw19.fit(data2_4['Hist_Per51'])
data2_4['Hist_Per51']=le_cw19.transform(data2_4['Hist_Per51'])

le_cw20 = preprocessing.LabelEncoder()
le_cw20.fit(data2_4['Hist_Per52'])
data2_4['Hist_Per52']=le_cw20.transform(data2_4['Hist_Per52'])

le_cw21 = preprocessing.LabelEncoder()
le_cw21.fit(data2_4['Hist_Veh6'])
data2_4['Hist_Veh6']=le_cw21.transform(data2_4['Hist_Veh6'])

le_cw22 = preprocessing.LabelEncoder()
le_cw22.fit(data2_4['Hist_Veh7'])
data2_4['Hist_Veh7']=le_cw22.transform(data2_4['Hist_Veh7'])

le_cw23 = preprocessing.LabelEncoder()
le_cw23.fit(data2_4['Hist_VehPer46'])
data2_4['Hist_VehPer46']=le_cw23.transform(data2_4['Hist_VehPer46'])

le_cw24 = preprocessing.LabelEncoder()
le_cw24.fit(data2_4['Hist_VehPer47'])
data2_4['Hist_VehPer47']=le_cw24.transform(data2_4['Hist_VehPer47'])

le_cw25 = preprocessing.LabelEncoder()
le_cw25.fit(data2_4['Veh23'])
data2_4['Veh23']=le_cw25.transform(data2_4['Veh23'])

le_cw26 = preprocessing.LabelEncoder()
le_cw26.fit(data2_4['Veh26'])
data2_4['Veh26']=le_cw26.transform(data2_4['Veh26'])



data2_4_train, data2_4_test=train_test_split(data2_4, test_size=0.3, random_state=1, stratify=data2_4.target)

X_train_4=data2_4_train[[col for col in data2_4_train.columns if col!="target"]]
y_train_4=data2_4_train.target
X_test_4=data2_4_test[[col for col in data2_4_test.columns if col!="target"]]
y_test_4=data2_4_test.target

X_train_50_4=X_train_4[columny_50]
X_test_50_4=X_test_4[columny_50]


wynik_50_4=wynik_modelu(xgb8, X_train_50_4, y_train_4, X_test_50_4, y_test_4)
#ROC AUC TRAIN: 0.787367
#ROC AUC TEST: 0.758764



#WYBOR NAJLEPSZEGO PROGU KLASYFIKATORA
#macierz bez progu
y_pred_ucz_XGB_8 = xgb8.predict(X_train_50)
y_pred_test_XGB_8 = xgb8.predict(X_test_50)

sklearn.metrics.confusion_matrix(y_train_ALL,y_pred_ucz_XGB_8)
sklearn.metrics.confusion_matrix(y_test_ALL,y_pred_test_XGB_8)



y_pred_0=xgb8.predict_proba(X_test_50)
xgb8.score(X_train_50,y_train_ALL)#0.8709852045575084
xgb8.score(X_test_50,y_test_ALL)#0.8696253307160536
y_pred_0 = y_pred_0[:, 1]

# roc curves
fpr, tpr, thresholds = roc_curve(y_test_ALL, y_pred_0)
# g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
plt.plot(fpr, tpr, linestyle='--', label='XGBoost')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()

#wynik
roc_auc=auc(fpr, tpr)

y_pred_nT=[]
for i in range(len(y_pred_0)):
    if y_pred_0[i]>0.137239:
        y_pred_nT.append(1)
    else:
        y_pred_nT.append(0)

sklearn.metrics.confusion_matrix(y_test_ALL,y_pred_nT)
sklearn.metrics.accuracy_score(y_test_ALL,y_pred_nT)
sklearn.metrics.precision_score(y_test_ALL,y_pred_nT)
sklearn.metrics.recall_score(y_test_ALL,y_pred_nT)
sklearn.metrics.f1_score(y_test_ALL,y_pred_nT)


# optimal threshold for precision-recall curve 
from sklearn.metrics import precision_recall_curve

#roc curves
precision, recall, thresholds2 = precision_recall_curve(y_test_ALL, y_pred_0)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds2[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot(recall, precision, linestyle='--', label='XGBoost')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()

y_pred_nT2=[]
for i in range(len(y_pred_0)):
    if y_pred_0[i]>0.192270:
        y_pred_nT2.append(1)
    else:
        y_pred_nT2.append(0)

sklearn.metrics.confusion_matrix(y_test_ALL,y_pred_nT2)
sklearn.metrics.accuracy_score(y_test_ALL,y_pred_nT2)
sklearn.metrics.precision_score(y_test_ALL,y_pred_nT2)
sklearn.metrics.recall_score(y_test_ALL,y_pred_nT2)
sklearn.metrics.f1_score(y_test_ALL,y_pred_nT2)


sklearn.metrics.confusion_matrix(y_test_ALL,y_pred_0_1)

y_pred_0_1=xgb8.predict(X_test_50)

