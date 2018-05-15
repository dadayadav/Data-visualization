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
