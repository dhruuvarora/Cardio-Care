#loading dataset
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#EDA
from collections import Counter
# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
# data splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# data modeling
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC
#warning
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('/content/drive/MyDrive/content/heart.csv')
data.head()

data.shape

data.info()

data.isnull().sum()

data.describe()

fig = data.target.value_counts().plot(kind = 'bar', color=["lightblue", 'lightgreen'])
fig.set_xticklabels(labels=['Has heart disease', "Doesn't have heart disease"], rotation=0);
plt.title("Heart Disease values")
plt.ylabel("Number of people");

labels = 'Male', 'Female'
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(data.sex.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

fig = sns.countplot(x = 'target', data = data, hue = 'sex')
fig.set_xticklabels(labels=["Doesn't have heart disease", 'Has heart disease'], rotation=0)
plt.legend(['Female', 'Male'])
plt.title("Heart Disease Frequency for Sex");

fig = data.cp.value_counts().plot(kind = 'bar', color = ['salmon', 'lightskyblue', 'springgreen', 'khaki'])
fig.set_xticklabels(labels=['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'], rotation=0)

plt.title('Chest pain type vs count');

fig = sns.countplot(x = 'cp', data = data, hue = 'target')
fig.set_xticklabels(labels=['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'], rotation=0)
plt.legend(['No disease', 'disease']);

plt.figure(figsize=(10,6))

#plotting the values for people who have heart disease
plt.scatter(data.age[data.target==1], 
            data.thalach[data.target==1], 
            c="tomato")

#plotting the values for people who doesn't have heart disease
plt.scatter(data.age[data.target==0], 
            data.thalach[data.target==0], 
            c="lightgreen")

# Addind info
plt.title("Heart Disease w.r.t Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");


df=data
# Creating another figure
plt.figure(figsize=(10,6))

#plotting the values for people who have heart disease
plt.scatter(df.age[df.target==1], 
            df.chol[df.target==1], 
            c="salmon") # define it as a scatter figure

#plotting the values for people who doesn't have heart disease
plt.scatter(df.age[df.target==0], 
            df.chol[df.target==0], 
            c="lightblue") # axis always come as (x, y)

# Add some helpful info
plt.title("Heart Disease w.r.t Age and Serum Cholestoral")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Serum cholestoral");


plt.figure(figsize = (15,15))
sns.heatmap(df.corr(), vmin = -1, vmax = +1, annot = True, cmap = 'coolwarm')
plt.show()


y = data["target"]
X = data.drop('target',axis=1)

cols=X.columns
for col in cols:
   X[col]=minmax_scale(X[col])
X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)

Counter(y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m1 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("confussion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(y_test,lr_predict))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
print("confussion matrix")
print(nb_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')
print(classification_report(y_test,nbpred))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m4 = 'Extreme Gradient Boost'
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_predicted = xgb.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("confussion matrix")
print(xgb_conf_matrix)
print("\n")
print("Accuracy of Extreme Gradient Boost:",xgb_acc_score*100,'\n')
print(classification_report(y_test,xgb_predicted))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report(y_test,knn_predicted))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
m7 = 'Support Vector Classifier'
svc =  SVC()
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state =10)
estimators = [('rf', RandomForestClassifier(max_depth=5)),('svr' ,LinearSVC())]
scv = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
scv.fit(X_train,y_train)
scv_predicted = scv.predict(X_test)
scv_conf_matrix = confusion_matrix(y_test, scv_predicted)
scv_acc_score = accuracy_score(y_test, scv_predicted)
print("confussion matrix")
print(scv_conf_matrix)
print("\n")
print("Accuracy of StackingClassifier:",scv_acc_score*100,'\n')
print(classification_report(y_test,scv_predicted))

lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,lr_predict)
nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,nbpred)
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rf_predicted)                                                             
xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predicted)
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dt_predicted)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svc_predicted)
scv_false_positive_rate,scv_true_positive_rate,scv_threshold = roc_curve(y_test,scv_predicted)


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
plt.plot(scv_false_positive_rate,scv_true_positive_rate,label='StackingClassifier')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()

model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','Extreme Gradient Boost',
                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine','StackingClassifier'], 'Accuracy': [lr_acc_score*100,
                    nb_acc_score*100,rf_acc_score*100,xgb_acc_score*100,knn_acc_score*100,dt_acc_score*100,svc_acc_score*100,scv_acc_score*100]})
model_ev.sort_values("Accuracy", axis = 0, ascending = True,
                 inplace = True, na_position ='last')

model_ev

colors = ['red','green','blue','gold','silver','yellow','orange','purple']
plt.figure(figsize=(20,10))
plt.title("barplot Represent Accuracy of different models")
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.bar(model_ev['Model'],model_ev['Accuracy'],color = colors)
plt.show()

TOP4=[]

acc1=[]


kf=KFold(n_splits=3)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  
  dt.fit(X_train,y_train)
  y_pred = dt.predict(X_test)

  acc1.append(accuracy_score(y_test,y_pred))
  print(classification_report(y_test,y_pred))
print("acc:",sum(acc1)*100/len(acc1))
TOP4.append(sum(acc1)*100/len(acc1))    

acc2=[]


kf=KFold(n_splits=3)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  
  rf.fit(X_train,y_train)
  y_pred = rf.predict(X_test)

  acc2.append(accuracy_score(y_test,y_pred))
  print(classification_report(y_test,y_pred))
print("acc:",sum(acc2)*100/len(acc2))
TOP4.append(sum(acc2)*100/len(acc2))    

acc=[]


kf=KFold(n_splits=3)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  
  scv.fit(X_train,y_train)
  y_pred = scv.predict(X_test)

  acc.append(accuracy_score(y_test,y_pred))
  print(classification_report(y_test,y_pred))
print("acc:",sum(acc)*100/len(acc))
TOP4.append(sum(acc)*100/len(acc))  

acc3=[]


kf=KFold(n_splits=3)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  
  xgb.fit(X_train,y_train)
  y_pred = xgb.predict(X_test)

  acc3.append(accuracy_score(y_test,y_pred))
  print(classification_report(y_test,y_pred))
print("acc:",sum(acc3)*100/len(acc3))
TOP4.append(sum(acc3)*100/len(acc3)) 

model_ev = pd.DataFrame({'Model': ['Decision Tree','Random Forest','StackingClassifier','Extreme Gradient Boost'], 'Accuracy': [TOP4[0],TOP4[1],TOP4[2],TOP4[3]]})
model_ev.sort_values("Accuracy", axis = 0, ascending = True,
                 inplace = True, na_position ='last')
model_ev

colors = ['red','green','blue','yellow']
plt.figure(figsize=(10,10))
plt.title("barplot Represent Accuracy of different models")
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.bar(["DECISION TREE","RANDOM FOREST","STACKING CLASSIFIER","XTREME GRADIENT BOOSTING"],TOP4,color = colors)
plt.show()
