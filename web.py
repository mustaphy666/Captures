import streamlit as st 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("streamlit example")
st.write('''
# Explore different classifier 
**which is the best?**      
''')
data=st.sidebar.selectbox("Select Dataset",('Iris','Breast_cancer','Wine'))
model=st.sidebar.selectbox('select classifier',('KNN','Svm','Random Forest'))
def play(data):
    if data == 'Iris':
        dat=datasets.load_iris()
    elif data== 'Breast_cancer':
        dat=datasets.load_breast_cancer()
    else:
        dat=datasets.load_wine()
    x=dat.data
    y=dat.target
    return x,y
x,y= play(data)
st.write('Shape of dataset', x.shape)
st.write('Number of classes', len(np.unique(y)))
def param(clf):
    p=dict()
    if clf =='KNN':
        K=st.sidebar.slider('k',1,15)
        p['K']=K
    elif clf =='Svm':
        C=st.sidebar.slider('c',0.01,10.0)
        p['C']=C
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        n_estimators=st.sidebar.slider('n',1,100)
        p['max_depth']=max_depth
        p['n_estimators']=n_estimators
        
    return p
let=param(model)
def wow(clf,let):
    if clf =='KNN':
       clf=KNeighborsClassifier(n_neighbors=let['K'])
    elif clf =='Svm':
       clf=SVC(C=let['C'])
    else:
        clf=RandomForestClassifier(max_depth=let['max_depth'],
         n_estimators=let['n_estimators'],random_state=42)
    return clf
clf=wow(model,let)
x_train,x_test,y_tarin,y_test=train_test_split(x,y,test_size=0.2,
                                            random_state=42)
clf.fit(x_train,y_tarin)
pred=clf.predict(x_test)
acc=accuracy_score(y_test,pred)
st.write(f'Model={model}')
st.write(f'Accuracy= {acc}')
pca=PCA(2)
pro=pca.fit_transform(x)
x1=pro[:,0]
x2=pro[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.colorbar()
st.pyplot(fig)
