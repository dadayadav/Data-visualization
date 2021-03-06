import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
%matplotlib inline
data = pd.read_csv(r'F:\Winter pack\Introduction to Game Development\pucho\U.S._Chronic_Disease_Indicators.csv', low_memory=False)
data.head()
print(len(data))
print(data.shape)
data = data.dropna(axis=1)
data.head()
value_yearStart =  data['YearStart'].unique().tolist()
value_yearStart.sort()
value_yearEnd = data['YearEnd'].unique().tolist()
value_yearEnd.sort()
print(value_yearStart,value_yearEnd)
uniq_LocationAbbr = data['LocationAbbr'].unique().tolist()
len(uniq_LocationAbbr)
uniq_LocationDesc = data['LocationDesc'].unique().tolist()
len(uniq_LocationDesc)
data_source = data['DataSource'].unique().tolist()
len(data_source)
topic = data['Topic'].unique().tolist()
len(topic)
dataValueType = data['DataValueType'].unique().tolist()
len(dataValueType)
LocationID = data['LocationID'].unique().tolist()
len(LocationID)
sns.set(style="whitegrid", color_codes=True)
sns.set_palette("deep",desat = 0.6)
sns.set_context(rc={"figure.figsize":(8,4)})
plt.figure(figsize=(10,10))
sns.countplot(y="LocationAbbr", data=data, color="c")
plt.show()
data['YearStart'].value_counts().plot(kind='bar')
plt.show()
data['YearEnd'].value_counts().plot(kind='bar')
plt.show()
data['LocationAbbr'].value_counts().plot()
plt.show()
data['DataSource'].value_counts().plot(kind='bar')
plt.show()
data['Topic'].value_counts().plot(kind='bar')
plt.show()
data['DataValueType'].value_counts().plot(kind='bar')
plt.show()
data['StratificationCategory1'].value_counts().plot(kind='bar')
plt.show()
import plotly
plotly.tools.set_credentials_file(username='dadayadav', api_key='hghqq2ST4H5jQI09Apqa')
import networkx as nx
import plotly.plotly as py
import plotly.graph_objs as go
trace = go.Heatmap(z=data.YearStart,x= data['LocationAbbr'],y=data['Topic'])
d=[trace]
py.iplot(d, filename='basic-heatmap')
# plot Yearstart data with label bar
data_sorted = data.sort_values(by = 'YearStart', ascending = False)
ax = sns.countplot(x="YearStart", data=data)
total = float(len(data))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,height, '{:1.0f}'.format(height),ha="center",rotation=90) 
plt.show()
# remove the insignificant data 
dataclear = data[data['YearStart']>= 2009]
for_alcohol = data[data.Topic == 'Alcohol']  
for_alcohol['Question'].value_counts().plot(kind='bar')
plt.show()
for_alcohol['DataValueType'].value_counts().plot(kind='bar')
plt.show()
# Now we visualize that in alcohol category which is most afffected Male or Female
male = 0
female = 0
for i in range(len(data['Stratification1'])):
    if data['Topic'][i] == 'Alcohol':
        if data['Stratification1'][i] == 'Male':
            male += 1
        if data['Stratification1'][i] == 'Female':
            female += 1    
print(male)
print(female)
# from graph we can also visualize that number of male and female is equal affected due to alcohol
for_alcohol['Stratification1'].value_counts().plot(kind='bar')
plt.show()

# PCA
from sklearn.preprocessing import LabelEncoder
from subprocess import check_output
data1 = pd.read_csv(r'F:\Winter pack\Introduction to Game Development\pucho\U.S._Chronic_Disease_Indicators.csv', low_memory=False)
data1 = data1.dropna(axis=1)
for i in data1.columns:
    if data1[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(data1[i].values))
        data1[i] = lbl.transform(list(data1[i].values))
data1.head()

def plotting(X_reduced,name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)
    titel="First three directions of "+name 
    ax.set_title(titel)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.cluster import KMeans,Birch
import statsmodels.formula.api as sm
from scipy import linalg
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
import matplotlib.pyplot as plt

n_col=12
X = data1
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)
Y=data['LocationDesc']
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)
names = [
         'FastICA',
         'KMeans',    
        ]
classifiers = [
    
    FastICA(n_components=n_col),
    KMeans(n_clusters=24),
    NMF(n_components=n_col),    
]
correction= [1,0,0,0,0,0,0,0,0]
temp=zip(names,classifiers,correction)
print(temp)
for name, clf,correct in temp:
    Xr=clf.fit_transform(X,Y)
    plotting(Xr,name)
    res = sm.OLS(Y,Xr).fit()
    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y))

import seaborn as sns; sns.set(color_codes=True)
# Analyse with the Topic of alcohol only
For_alcohol = disease[disease.Topic == 0] 
ax = sns.lmplot(x="LocationAbbr", y="YearStart", data=For_alcohol)

for_alcohol = data[data.Topic == 'Alcohol'] 
for_alcohol['Question'].value_counts().plot(kind='bar')
plt.show()

for_alcohol['DataValueType'].value_counts().plot(kind='bar')
plt.show()
# Now we visualize that in alcohol category which is most afffected Male or Female
male = 0
female = 0
for i in range(len(data['Stratification1'])):
    if data['Topic'][i] == 'Alcohol':
        if data['Stratification1'][i] == 'Male':
            male += 1
        if data['Stratification1'][i] == 'Female':
            female += 1    
print(male)
print(female)
# from graph we can also visualize that number of male and female is equal affected due to alcohol
for_alcohol['Stratification1'].value_counts().plot(kind='bar')
plt.show()
# extract useful data feature other feature are just id of these data
useful_data = data.iloc[:, [0,1,2,4,5,6,7,8,9]]
# we visualize the dimention reduction method on data feature
x = useful_data.iloc[:,2:]
# we convert each variable into some numerical value
# Encoder to the data variable
from sklearn.preprocessing import LabelEncoder
for i in x.columns:
    if x[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(x[i].values))
        x[i] = lbl.transform(list(x[i].values)) 
# create indication matrix
# label the train data into numeric form by using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6])
x = ohe.fit_transform(x).toarray()
# make burt matrix
tx = x.transpose()
mx = np.matrix(tx)
my = np.matrix(x)
x = mx * my
# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # first see how many value dominating then assign the component
x_train = pca.fit_transform(x) 
explained_variance = pca.explained_variance_ratio_
explained_variance
a = []
b = []
for i in range(len(x_train)):
    x,y = x_train[i]
    a.append(x)
    b.append(y)

fig = plt.figure(figsize= (5,5))
ax = fig.add_subplot(111)
ax.scatter(a, b, c=c)
plt.show()
# dimension analysis by taking less column
y = data.iloc[:,2:5]
for i in y.columns:
    if y[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(y[i].values))
        y[i] = lbl.transform(list(y[i].values)) 
ohe = OneHotEncoder(categorical_features = [0,1,2])
y = ohe.fit_transform(y).toarray()
# make burt matrix
tx = y.transpose()
mx = np.matrix(tx)
my = np.matrix(y)
y = mx * my
# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y = sc.fit_transform(y)
pca = PCA(n_components = 2) # first see how many value dominating then assign the component
y_train = pca.fit_transform(y)
a = []
b = []
for i in range(len(y_train)):
    x,y = y_train[i]
    a.append(x)
    b.append(y)

fig = plt.figure(figsize= (5,5))
ax = fig.add_subplot(111)
ax.scatter(a, b)
plt.show()
x = useful_data.iloc[:,2:]

from sklearn.preprocessing import LabelEncoder
for i in x.columns:
    if x[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(x[i].values))
        x[i] = lbl.transform(list(x[i].values)) 
sns.lmplot(x='Topic', y='Question', hue='DataSource', data=x, fit_reg=False)
data['DataSource'].value_counts().plot(kind='bar')
plt.show()
# Filter DataSource 

data_source = data[data['DataSource']== 'BRFSS']
data_source.head()
# year include when data source is BRFSS
data_source['YearStart'].value_counts().plot(kind='bar')
plt.show()

data_source['Stratification1'].value_counts().plot(kind='bar')
plt.show()
# Apply Factor analysis on whole dataset
from sklearn.decomposition import FactorAnalysis
x = useful_data.iloc[:,2:]
feature_names = ['locationAbbr', 'DataSource', 'Topic', 'Question', 'DataValueType', 'StratificationCategory1', 'Stratification1']
for i in x.columns:
    if x[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(x[i].values))
        x[i] = lbl.transform(list(x[i].values)) 
factor = FactorAnalysis(n_components=None).fit(x)
X = pd.DataFrame(factor.components_,columns = feature_names)
# Visualize the relation between variables
 Y = data.iloc[:,[2,4,5,6,7,8,9]]
 from sklearn.preprocessing import LabelEncoder
for i in Y.columns:
    if Y[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(Y[i].values))
        Y[i] = lbl.transform(list(Y[i].values)) 
sns.pairplot(Y)
plt.show()
