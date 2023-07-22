#!/usr/bin/env python
# coding: utf-8

# # <center>  Tugas Akhir Data Mining - Prediksi Pendapatan Anggaran Iklan

# ### Deskripsi :
# Dataset yang saya ambil dari kaggle yaitu Advertising Budget and Sales. Dalam datset menjelaskan tentang anggaran dari 3 jenis periklanan yaitu TV, Radio, Newspaper yang dikonversikan dalam ribu dolan(1000$).
# 
# Sedangkan untuk "Sales" adalah sebagai label yang diatur dalam juta dolar(1M$).

# In[ ]:





# # <center>1. Data Exploration

# In[1]:


# Import Library
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.formula import api
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


# Import Dataset
df = pd.read_csv('Advertising Budget and Sales.csv', index_col=0, names=['TV','Radio','Newspaper','Sales'], skiprows=1)

target = 'Sales'
features = [i for i in df.columns if i not in [target]]

original_df = df.copy(deep=True)


# In[3]:


# Load Dataset
df = pd.DataFrame(df)
df


# In[4]:


# check tipe data seluruh kolom attribut
df.info()


# In[5]:


df.describe()


# In[6]:


#Check apakah ada nilai unik pada setiap kolom

df.nunique().sort_values()


# In[7]:


# Check apakah ada tipe data kategorikal didalam attribut
nu = df[features].nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0; #numerik & kategori attribut

for i in range(df[features].shape[1]):
    if nu.values[i]<=16:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('\n\033[1mHasil :\033[0m Dataset mempunyai {} numerical & {} categorical attribut.'.format(len(nf),len(cf)))


# In[8]:


print('\033[1mPlot Check Outliers'.center(90))

n=3

plt.figure(figsize=[15,3*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()


# **Hasil :** Dapat dilihat pada Newspaper terdapat Outlier yang harus dihilangkan

# In[9]:


# Check hubungan antar attribut dan label
print('\033[1mPlot Relasi dalam Dataset'.center(90))

g = sns.pairplot(df)
g.map_upper(sns.kdeplot, levels=4, color="blue")
plt.show()


# In[ ]:





# # <center> 2. Data Preprocessing

# In[10]:


# Inisialisasi df3
df1 = df.copy()
df3 = df1.copy()
df1 = df3.copy()

# Penghapusan outlier menggunakan statsmodel
features1 = nf

# quantile digunakan untuk membagi sama rata pada dataset
for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
display(df1.head())
print('\n\033[1mHasil:\033[0m\nSebelum Outlier dihilangkan, Dataset mempunyai {} sample.'.format(df3.shape[0]))
print('Setelah Outlier dihilangkan, Dataset mempunyai {} sample.'.format(df1.shape[0]))


# In[ ]:





# # <center> 3. Data Manipulation

# In[11]:


#Splitting the data intro training & testing sets

m=[]
for i in df.columns.values:
    m.append(i.replace(' ','_'))
    
df.columns = m
X = df.drop([target],axis=1)
Y = df[target]
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
X_Train.reset_index(drop=True,inplace=True)

print('Data Asli  ---> ',X.shape,Y.shape,
      '\nData Training  ---> ',X_Train.shape,Y_Train.shape,
      '\nData Testing   ---> ', X_Test.shape,'', Y_Test.shape)


# In[12]:


#Feature Scaling (Standardisasi)

std = StandardScaler()

print('\033[1mStandardisasi pada Data Training'.center(80))
X_Train_std = std.fit_transform(X_Train)
X_Train_std = pd.DataFrame(X_Train_std, columns=X.columns)
display(X_Train_std.describe())

print('\n','\033[1mStandardisasi pada Data Testing'.center(80))
X_Test_std = std.transform(X_Test)
X_Test_std = pd.DataFrame(X_Test_std, columns=X.columns)
display(X_Test_std.describe())


# In[13]:


# Check 5 baris atas dan bawah pada Data Training
#X_Train_std.head()
#X_Train_std.tail()


# In[14]:


# Check 5 baris atas dan bawah pada Data Testing
#X_Test_std.head()
#X_Test_std.tail()


# In[ ]:





# # <center> 4. Extraction

# In[15]:


#Check korelasi matrix

print('\033[1mCorrelation Matrix'.center(65))
plt.figure(figsize=[8,5])
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center=0) #cmap='BuGn'
plt.show()


# In[ ]:





# # <center> 5. Modelling

# In[16]:


# Definisi evaluate
rc=np.random.choice(X_Train_std.columns,2)
def Evaluate(n, pred1,pred2):
    # Rencana prediksi yang diprediksi di samping titik data aktual
    plt.figure(figsize=[15,6])
    for e,i in enumerate(rc):
        plt.subplot(2,3,e+1)
        plt.scatter(y=Y_Train, x=X_Train_std[i], label='Actual')
        plt.scatter(y=pred1, x=X_Train_std[i], label='Prediction')
        plt.legend()
    plt.show()


# In[17]:


# Model Multiple Linear Regression

MLR = LinearRegression().fit(X_Train_std,Y_Train)
pred1 = MLR.predict(X_Train_std)
pred2 = MLR.predict(X_Train_std)

print('{}{}\033[1m Evaluasi Model Multiple Linear Regression \033[0m{}{}\n'.format('<'*3,'-'*20,'-'*20,'>'*3))
print('Koefisien Model Regresi ditemukan ',MLR.coef_)
print('Intercept Model Regresi ditemukan ',MLR.intercept_)

Evaluate(0, pred1, pred2)


# In[18]:


# Membuat model
models = LinearRegression()
models.fit(X_Train, Y_Train)


# In[19]:


# Membuat prediksi target
Y_Pred = models.predict(X_Test)

print(f'R2-Score :  {r2_score(Y_Test,Y_Pred)}')


# In[20]:


import pickle
import streamlit as st

filename = 'deployment.sav'
pickle.dump(models, open (filename, 'wb'))


# In[21]:


#Testing Model Regresi Linear

XY_Train = pd.concat([X_Train_std,Y_Train.reset_index(drop=True)],axis=1)
a = XY_Train.columns.values

results = api.ols(formula='{} ~ {}'.format(target,' + '.join(i for i in X_Train.columns)), data=XY_Train).fit()
print(results.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# # <center> 6. Kesimpulan

#     Dataset berisi 200 data yang kemudian ada 2 outlier yang harus dihilangkan sehingga menjadi 198 data.
# 
#     Berdasarkan dari hasil diatas dengan mendapatkan r2-score 0.91, R-Square 0.89, dan dengan standard error yang bervariasi yaitu (TV = 0.141, Radio = 0.153, Newspaper = 0.153).Dari r2-score dan R-Squared yang diperoleh nilai akurasi yang tinggi, maka dapat disimpulkan bahwa seluruh attribut mempunyai pengaruh yang cukup signifikan dalam menentukan hasil dari penjualan.

# In[ ]:




