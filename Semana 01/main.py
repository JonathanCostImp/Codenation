#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")
df = black_friday


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    q1 = df.shape
    return q1
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
    q2 = int(len(df.query('Age == "26-35" & Gender == "F"')))
    return q2
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[5]:


def q3():
    q3 = df['User_ID'].nunique()
    return q3
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[6]:


def q4():
    q4 = df.dtypes.value_counts().size
    return q4
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[7]:


def q5():
    aux = pd.DataFrame({'colunas': df.columns, 'tipos': df.dtypes, 'missing': df.isna().sum()})
    aux['missing_percentual'] = aux['missing'] / df.shape[0]
    q5 = float(aux['missing_percentual'].max())
    return q5
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[8]:


def q6():
    q6 = int(df['Product_Category_3'].isnull().sum())
    return q6
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[9]:


def q7():
    q7 = int(df['Product_Category_3'].mode().sum())
    return q7
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[10]:


def q8():
    minpurchase = df['Purchase'].min()
    maxpurchase = df['Purchase'].max()
    normpurchase = (df.Purchase - minpurchase) / (maxpurchase - minpurchase)
    q8 = float(normpurchase.mean())
    return q8
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[11]:


def q9():
    meanpurchase = df['Purchase'].mean()
    stdpurchase = df['Purchase'].std()
    padpurchase = pd.DataFrame()
    padpurchase['Purchase'] = (df.Purchase - meanpurchase) / stdpurchase
    q9 = len(padpurchase.query('Purchase >= -1 & Purchase <= 1'))
    return q9
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[31]:


def q10():
    q10 = df.Product_Category_2.isna().shape[0] == df.Product_Category_3.isna().shape[0]
    return q10
    pass


# In[ ]:




