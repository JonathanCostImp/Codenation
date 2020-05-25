#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import math

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, Binarizer, KBinsDiscretizer, MinMaxScaler, StandardScaler, PolynomialFeatures)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.datasets import load_digits, fetch_20newsgroups


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline
from IPython.core.pylabtools import figsize
figsize(12, 8)
sns.set()


# In[3]:


df = pd.read_csv("countries.csv")


# In[4]:


new_column_names = ["Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio", "Net_migration", "Infant_mortality", "GDP", "Literacy", 
                    "Phones_per_1000", "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture", "Industry", "Service"]
df.columns = new_column_names


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.


# In[6]:


def parse_str_float(str):
    try:
        return float(str.replace(",", "."))
    except:
        return str

df = df.applymap(parse_str_float)


# In[7]:


cols = ['Country', 'Region']

def te (df, colunas): #te: tira espaço
    for col in colunas:
        df[col] = df[col].str.strip()

te (df, cols)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[8]:


def q1():
    return np.sort(df['Region'].unique()).tolist()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[9]:


def q2():
    est = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'quantile')
    est.fit(df[['Pop_density']])
    popdensitycont = est.transform(df[["Pop_density"]])
    p90 = math.trunc((90 / 100) * len(popdensitycont))
    
    return len(popdensitycont) - p90


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


def q3():
    ohe = OneHotEncoder()
    oheft = ohe.fit_transform(df[["Region", "Climate"]].astype('str'))
    return int(oheft.shape[1])


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[11]:


test_country = ['Test Country', 'NEAR EAST', -0.19032480757326514, -0.3232636124824411, -0.04421734470810142, -0.27528113360605316, 0.13255850810281325, 
                -0.8054845935643491, 1.0119784924248225, 0.6189182532646624, 1.0074863283776458, 0.20239896852403538, -0.043678728558593366, 
                -0.13929748680369286, 1.3163604645710438, -0.3699637766938669, -0.6149300604558857, -0.854369594993175, 0.263445277972641, 0.5712416961268142]


# In[12]:


ct = pd.DataFrame([test_country], columns = df.columns)


# In[13]:


cols2 = df.select_dtypes(['int64', 'float64']).columns
pl = Pipeline(steps = [('imp', SimpleImputer(strategy = 'median')), ('scaler', StandardScaler())])
tf = ColumnTransformer(transformers = [('number', pl, cols2)], n_jobs = -1)
tf.fit(df)


# In[14]:


def q4():
    res = tf.transform(ct)[0][cols2.get_loc('Arable')]
    return round(float(res), 3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[15]:


def q5():
    q1, q3 = df['Net_migration'].quantile([.25,.75])
    iqr = q3 - q1

    lbaixo = q1-1.5*iqr
    lalto = q3+1.5*iqr

    outliers_abaixo = int(df[df['Net_migration'] < lbaixo].shape[0])
    outliers_acima = int(df[df['Net_migration'] > lalto].shape[0])
    remove = bool((outliers_abaixo+outliers_acima)/df['Net_migration'].shape[0] < 0.1)
    
    return (outliers_abaixo,outliers_acima,remove)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[16]:


categorias = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup_data = fetch_20newsgroups(subset='train', categories=categorias, shuffle=True, random_state=12)


# In[17]:


def q6():
    vec = CountVectorizer(analyzer = 'word').fit(newsgroup_data.data)
    bag = vec.transform(newsgroup_data.data)
    return int(bag[:, vec.vocabulary_['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[18]:


def q7():
    tf_vec = TfidfVectorizer().fit(newsgroup_data.data)
    tf_bag = tf_vec.transform(newsgroup_data.data)
    return float(tf_bag[:, tf_vec.vocabulary_['phone']].sum().round(3))


# In[ ]:




