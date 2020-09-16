# biblioteki
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import seaborn as sns


#opcje,  żeby wyswietlało się więcej obesrwacji
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_info_columns', 10000)
#obejrzenie danych 
data1.describe()
data1.info()
#zmiana typu zmiennych
convert_dict= {'Reg1': object,
               'Reg10': object,
               'Reg12': object,
               'Dif2': object,
               'Per12': object,
               'Reg80': object,
               'Veh17': object,
               'Veh20': object
               }
data1 = data1.astype(convert_dict)
#ponowne sprawdzenie typu zmiennych
data1.info()

#sprawdzenie null
sum(data1.isnull().sum())
data1.isnull().sum().sort_values(ascending=False).head(12)
#wyrzucenie NULL 
data1=data1.dropna(axis=0)

#IsolationForest
#stworzenie zbioru zmiennych numerycznych bez targetu
data_num= data1.select_dtypes(exclude=['object'])
data_num_woTarget=data_num[[col for col in data_num.columns if col!="target"]]

#zdefiniowanie algorytmu
clf = IsolationForest(n_estimators=500, max_samples='auto', random_state=1234)
clf.fit(data_num_woTarget)
data_num_woTarget_clf=clf.predict(data_num_woTarget)

#dodanie kolumny z oznaczeniami -1/1
data1['iForest']=data_num_woTarget_clf

clf.__getitem__
#outlier metoda IQR, tutaj będziemy mogli jeszcze raz zrobić wykres
data_num= data1.select_dtypes(exclude=['object'])

#wyfiltrowanie tylko zmiennych nie uznanych z outliers
data2=data1[data1['iForest']==1].drop(['iForest'], axis=1)
data2.index=range(len(data2))#nowe indeksowanie

#zmienne kategoryczne 
def ZM_kat_poziomy(zmienna, eksp_min):
    VC=data2[zmienna].value_counts()
    kolumny=[]
    eksp=[]
    for i in range(len(VC)):
        if VC.iloc[i] > len(data2[zmienna])*eksp_min:
            kolumny.append(VC.index[i])
            eksp.append(VC.iloc[i])
    a=len(kolumny)+1
    b=1-sum(eksp)/len(data2[zmienna])
    print("Stworzymy %i klas." % a), print("%f zbioru zrczucimay do klasy 'INNA'" % b)

def ZM_kat_poziomy_OK(zmienna, eksp_min):
    VC=data2[zmienna].value_counts()
    kolumny=[]
    eksp=[]
    for i in range(len(VC)):
        if VC.iloc[i] > len(data2[zmienna])*eksp_min:
            kolumny.append(VC.index[i])
            eksp.append(VC.iloc[i])
    kolumny_INNA=[x for x in VC.index if x not in kolumny]
    data2[zmienna]=data2[zmienna].replace(kolumny_INNA, "INNA")
    
#Veh13
ZM_kat_poziomy('Veh13', 0.02)
ZM_kat_poziomy_OK('Veh13', 0.02)
#Veh11
ZM_kat_poziomy('Veh11', 0.01)
ZM_kat_poziomy_OK('Veh11', 0.01)
#Veh10
ZM_kat_poziomy('Veh10', 0.005)
ZM_kat_poziomy_OK('Veh10', 0.005)
#Veh4
ZM_kat_poziomy('Veh4', 0.05)
ZM_kat_poziomy_OK('Veh4', 0.05)
#Hist_VehPer47
ZM_kat_poziomy('Hist_VehPer47', 0.05)
ZM_kat_poziomy_OK('Hist_VehPer47', 0.05)
#Hist_VehPer46
ZM_kat_poziomy('Hist_VehPer46', 0.01)
ZM_kat_poziomy_OK('Hist_VehPer46', 0.01)
#Veh19
ZM_kat_poziomy('Veh19', 0.05)
ZM_kat_poziomy_OK('Veh19', 0.05)
#Hist_Veh7
ZM_kat_poziomy('Hist_Veh7', 0.05)
ZM_kat_poziomy_OK('Hist_Veh7', 0.05)
#Hist_Veh6
ZM_kat_poziomy('Hist_Veh6', 0.01)
ZM_kat_poziomy_OK('Hist_Veh6', 0.01)
#Hist_Per52
ZM_kat_poziomy('Hist_Per52', 0.05)
ZM_kat_poziomy_OK('Hist_Per52', 0.05)
#Hist_Per51
ZM_kat_poziomy('Hist_Per51', 0.01)
ZM_kat_poziomy_OK('Hist_Per51', 0.01)
#Reg76
ZM_kat_poziomy('Reg76', 0.005)
ZM_kat_poziomy_OK('Reg76', 0.005)
#Reg75
ZM_kat_poziomy('Reg75', 0.005)
ZM_kat_poziomy_OK('Reg75', 0.005)
#Reg74
ZM_kat_poziomy('Reg74', 0.01)
ZM_kat_poziomy_OK('Reg74', 0.01)
#Reg12
ZM_kat_poziomy('Reg12', 0.05)
ZM_kat_poziomy_OK('Reg12', 0.05)
#Reg10
ZM_kat_poziomy('Reg10', 0.05)
ZM_kat_poziomy_OK('Reg10', 0.05)
#Reg1
ZM_kat_poziomy('Reg1', 0.05)
ZM_kat_poziomy_OK('Reg1', 0.05)
#poprawa 
data2['Reg12']=data2['Reg12'].replace('INNA', -1)
data2['Reg10']=data2['Reg10'].replace('INNA', -1)  
data2['Reg1']=data2['Reg1'].replace('INNA', -1)  
#sprawdzenie
data_object2= data2.select_dtypes(include=['object'])      
data_object2.describe().to_excel("C:/Users/...") 

# Label Encoder 
le = preprocessing.LabelEncoder()
le.fit(data2['Veh13'])
data2['Veh13']=le.transform(data2['Veh13'])

le1 = preprocessing.LabelEncoder()
le1.fit(data2['Veh4'])
data2['Veh4']=le1.transform(data2['Veh4'])

le2 = preprocessing.LabelEncoder()
le2.fit(data2['Veh7'])
data2['Veh7']=le2.transform(data2['Veh7'])

le3 = preprocessing.LabelEncoder()
le3.fit(data2['Veh10'])
data2['Veh10']=le3.transform(data2['Veh10'])

le4 = preprocessing.LabelEncoder()
le4.fit(data2['Veh11'])
data2['Veh11']=le4.transform(data2['Veh11'])

le5 = preprocessing.LabelEncoder()
le5.fit(data2['Veh17'])
data2['Veh17']=le5.transform(data2['Veh17'])

le6 = preprocessing.LabelEncoder()
le6.fit(data2['Veh19'])
data2['Veh19']=le6.transform(data2['Veh19'])

le7 = preprocessing.LabelEncoder()
le7.fit(data2['Veh20'])
data2['Veh20']=le7.transform(data2['Veh20'])

le8 = preprocessing.LabelEncoder()
le8.fit(data2['Reg1'])
data2['Reg1']=le8.transform(data2['Reg1'])

le9 = preprocessing.LabelEncoder()
le9.fit(data2['Reg10'])
data2['Reg10']=le9.transform(data2['Reg10'])

le10 = preprocessing.LabelEncoder()
le10.fit(data2['Reg12'])
data2['Reg12']=le10.transform(data2['Reg12'])

le11 = preprocessing.LabelEncoder()
le11.fit(data2['Dif2'])
data2['Dif2']=le11.transform(data2['Dif2'])

le12 = preprocessing.LabelEncoder()
le12.fit(data2['Per12'])
data2['Per12']=le12.transform(data2['Per12'])

le13 = preprocessing.LabelEncoder()
le13.fit(data2['Reg74'])
data2['Reg74']=le13.transform(data2['Reg74'])

le14 = preprocessing.LabelEncoder()
le14.fit(data2['Reg75'])
data2['Reg75']=le14.transform(data2['Reg75'])

le15 = preprocessing.LabelEncoder()
le15.fit(data2['Reg76'])
data2['Reg76']=le15.transform(data2['Reg76'])

le16 = preprocessing.LabelEncoder()
le16.fit(data2['Reg79'])
data2['Reg79']=le16.transform(data2['Reg79'])

le17 = preprocessing.LabelEncoder()
le17.fit(data2['Reg80'])
data2['Reg80']=le17.transform(data2['Reg80'])

le18 = preprocessing.LabelEncoder()
le18.fit(data2['Reg81'])
data2['Reg81']=le18.transform(data2['Reg81'])

le19 = preprocessing.LabelEncoder()
le19.fit(data2['Hist_Per51'])
data2['Hist_Per51']=le19.transform(data2['Hist_Per51'])

le20 = preprocessing.LabelEncoder()
le20.fit(data2['Hist_Per52'])
data2['Hist_Per52']=le20.transform(data2['Hist_Per52'])

le21 = preprocessing.LabelEncoder()
le21.fit(data2['Hist_Veh6'])
data2['Hist_Veh6']=le21.transform(data2['Hist_Veh6'])

le22 = preprocessing.LabelEncoder()
le22.fit(data2['Hist_Veh7'])
data2['Hist_Veh7']=le22.transform(data2['Hist_Veh7'])

le23 = preprocessing.LabelEncoder()
le23.fit(data2['Hist_VehPer46'])
data2['Hist_VehPer46']=le23.transform(data2['Hist_VehPer46'])

le24 = preprocessing.LabelEncoder()
le24.fit(data2['Hist_VehPer47'])
data2['Hist_VehPer47']=le24.transform(data2['Hist_VehPer47'])

le25 = preprocessing.LabelEncoder()
le25.fit(data2['Veh23'])
data2['Veh23']=le25.transform(data2['Veh23'])

le26 = preprocessing.LabelEncoder()
le26.fit(data2['Veh26'])
data2['Veh26']=le26.transform(data2['Veh26'])

#Podział zbioru 
#Losujemy 5 losowych próbek po 20% 
random_sample1 = data2.iloc[np.random.randint(0,len(data2),int(len(data2) / 5))]
random_sample2 = data2.iloc[np.random.randint(0,len(data2),int(len(data2) / 5))]
random_sample3 = data2.iloc[np.random.randint(0,len(data2),int(len(data2) / 5))]
random_sample4 = data2.iloc[np.random.randint(0,len(data2),int(len(data2) / 5))]
random_sample5 = data2.iloc[np.random.randint(0,len(data2),int(len(data2) / 5))]

X1=random_sample1[[col for col in data2.columns if col!="target"]]
y1=random_sample1.target
X2=random_sample2[[col for col in data2.columns if col!="target"]]
y2=random_sample2.target
X3=random_sample3[[col for col in data2.columns if col!="target"]]
y3=random_sample3.target
X4=random_sample4[[col for col in data2.columns if col!="target"]]
y4=random_sample4.target
X5=random_sample5[[col for col in data2.columns if col!="target"]]
y5=random_sample5.target

X_train1, X_test1, y_train1, y_test1=train_test_split(X1,y1, test_size=0.3, random_state=1, stratify=y1)
X_train2, X_test2, y_train2, y_test2=train_test_split(X2,y2, test_size=0.3, random_state=1, stratify=y2)
X_train3, X_test3, y_train3, y_test3=train_test_split(X3,y3, test_size=0.3, random_state=1, stratify=y3)
X_train4, X_test4, y_train4, y_test4=train_test_split(X4,y4, test_size=0.3, random_state=1, stratify=y4)
X_train5, X_test5, y_train5, y_test5=train_test_split(X5,y5, test_size=0.3, random_state=1, stratify=y5)

forest1 = RandomForestClassifier()
forest2 = RandomForestClassifier()
forest3 = RandomForestClassifier()
forest4 = RandomForestClassifier()
forest5 = RandomForestClassifier()

forest1.fit(X_train1, y_train1)
forest2.fit(X_train2, y_train2)
forest3.fit(X_train3, y_train3)
forest4.fit(X_train4, y_train4)
forest5.fit(X_train5, y_train5)

importances1=forest1.feature_importances_
importances2=forest2.feature_importances_
importances3=forest3.feature_importances_
importances4=forest4.feature_importances_
importances5=forest5.feature_importances_

labels1=X_train1.columns
labels2=X_train2.columns
labels3=X_train3.columns
labels4=X_train4.columns
labels5=X_train5.columns

importances_fin1=pd.DataFrame()
importances_fin1['kolumna']=labels1
importances_fin1['istotnosc']=importances1

importances_fin2=pd.DataFrame()
importances_fin2['kolumna']=labels2
importances_fin2['istotnosc']=importances2

importances_fin3=pd.DataFrame()
importances_fin3['kolumna']=labels3
importances_fin3['istotnosc']=importances3

importances_fin4=pd.DataFrame()
importances_fin4['kolumna']=labels4
importances_fin4['istotnosc']=importances4

importances_fin5=pd.DataFrame()
importances_fin5['kolumna']=labels5
importances_fin5['istotnosc']=importances5

importances_fin21=importances_fin1.sort_values(by=['istotnosc'], ascending=False)
importances_fin22=importances_fin2.sort_values(by=['istotnosc'], ascending=False)
importances_fin23=importances_fin3.sort_values(by=['istotnosc'], ascending=False)
importances_fin24=importances_fin4.sort_values(by=['istotnosc'], ascending=False)
importances_fin25=importances_fin5.sort_values(by=['istotnosc'], ascending=False)

importances_fin21.index=range(len(importances_fin21))
importances_fin22.index=range(len(importances_fin22))
importances_fin23.index=range(len(importances_fin23))
importances_fin24.index=range(len(importances_fin24))
importances_fin25.index=range(len(importances_fin25))

z1=[]
z2=[]
z3=[]
z4=[]
z5=[]
for i in range(10):
    z1.append(importances_fin21.kolumna[i])
    z2.append(importances_fin22.kolumna[i])
    z3.append(importances_fin23.kolumna[i])
    z4.append(importances_fin24.kolumna[i])
    z5.append(importances_fin25.kolumna[i])

por=pd.DataFrame()
por['z1']=z1
por['z2']=z2
por['z3']=z3
por['z4']=z4
por['z5']=z5

forest1.score(X_test1, y_test1)
forest2.score(X_test2, y_test2)
forest3.score(X_test3, y_test3)
forest4.score(X_test4, y_test4)
forest5.score(X_test5, y_test5)

# LOSOWANIE WARSTWOWE
#stowrzenie warstwy per wybrane zmienne
data2['warstwa']=data2['target'].astype(str)+ "_" +(round(data2['Per2']/1000)).astype(str)+ "_" +(round(data2['Veh3']/1000)).astype(str)+ "_" +(round(data2['Veh5']/100)).astype(str)+ "_" +(round(data2['Veh8']/500)).astype(str)+ "_" +(round(data2['Veh9']/500)).astype(str)+ "_" +(round(data2['Veh27']/500)).astype(str)+ "_" +(round(data2['Veh1']/10)).astype(str)

#sprawdzenie liczebnosci warstw
data2_count = data2.groupby('warstwa')["target"].count()

#zrzucenie warstw które mają tylko jedną obserwacje to jednej warstwy
a2=[]
for i in range(len(data2_count)):
    if data2_count[i]<5:
        a2.append(data2_count.index[i])

a3=[]
for i in data2.index:
    if data2['warstwa'][i] in a2:
        a3.append("inna")
    else:
        a3.append(data2['warstwa'][i])

data2['warstwa2']=a3    
data2.groupby('warstwa2')["target"].count()

#zmianna zmiennych kategoeycznym na wczesniejsze oznaczenia
data2['Veh13']=le.inverse_transform(data2['Veh13'])
data2['Veh4']=le1.inverse_transform(data2['Veh4'])
data2['Veh7']=le2.inverse_transform(data2['Veh7'])
data2['Veh10']=le3.inverse_transform(data2['Veh10'])
data2['Veh11']=le4.inverse_transform(data2['Veh11'])
data2['Veh17']=le5.inverse_transform(data2['Veh17'])
data2['Veh19']=le6.inverse_transform(data2['Veh19'])
data2['Veh20']=le7.inverse_transform(data2['Veh20'])
data2['Reg1']=le8.inverse_transform(data2['Reg1'])
data2['Reg10']=le9.inverse_transform(data2['Reg10'])
data2['Reg12']=le10.inverse_transform(data2['Reg12'])
data2['Dif2']=le11.inverse_transform(data2['Dif2'])
data2['Per12']=le12.inverse_transform(data2['Per12'])
data2['Reg74']=le13.inverse_transform(data2['Reg74'])
data2['Reg75']=le14.inverse_transform(data2['Reg75'])
data2['Reg76']=le15.inverse_transform(data2['Reg76'])
data2['Reg79']=le16.inverse_transform(data2['Reg79'])
data2['Reg80']=le17.inverse_transform(data2['Reg80'])
data2['Reg81']=le18.inverse_transform(data2['Reg81'])
data2['Hist_Per51']=le19.inverse_transform(data2['Hist_Per51'])
data2['Hist_Per52']=le20.inverse_transform(data2['Hist_Per52'])
data2['Hist_Veh6']=le21.inverse_transform(data2['Hist_Veh6'])
data2['Hist_Veh7']=le22.inverse_transform(data2['Hist_Veh7'])
data2['Hist_VehPer46']=le23.inverse_transform(data2['Hist_VehPer46'])
data2['Hist_VehPer47']=le24.inverse_transform(data2['Hist_VehPer47'])
data2['Veh23']=le25.inverse_transform(data2['Veh23'])
data2['Veh26']=le26.inverse_transform(data2['Veh26'])

#podział  w stosunku do warstwy
data2_50_1, data2_50_2 = train_test_split(data2, test_size=0.5, random_state=1234, stratify=data2[['warstwa2']])
data2_1, data2_2 = train_test_split(data2_50_1, test_size=0.5, random_state=1234, stratify=data2_50_1[['warstwa2']])
data2_3, data2_4 = train_test_split(data2_50_2, test_size=0.5, random_state=1234, stratify=data2_50_2[['warstwa2']])

# wyrzucenie kolumn "warstwa"
data2_1=data2_1.drop(["warstwa","warstwa2"], axis=1)
data2_2=data2_2.drop(["warstwa","warstwa2"], axis=1)
data2_3=data2_3.drop(["warstwa","warstwa2"], axis=1)
data2_4=data2_4.drop(["warstwa","warstwa2"], axis=1)


#  wykres istotnosci zmiennych 
feat_labels = X_train1.columns
forest1 = RandomForestClassifier()

forest1.fit(X_train1, y_train1)
importances = forest1.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train1.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

import matplotlib.pyplot as plt

plt.title('Istotność cech')
plt.bar(range(X_train1.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train1.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train1.shape[1]])
plt.tight_layout()
#plt.savefig('rysunki/04_09.png', dpi=300)
plt.show()

#2
sns.set(color_codes=True)
m=30

plt.title('Istotność cech')
plt.bar(range(m), 
        importances[indices[0:m]],
        align='center')

plt.xticks(range(m), 
           feat_labels[indices[0:m]], rotation=90)
plt.xlim([-1, m])
plt.tight_layout()
plt.savefig('C:/Users/...', dpi=300)
plt.show()


