#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline
#from IPython.core.pylabtools import figsize
#figsize(12, 8)
#sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
df = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000), "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# In[4]:


#df.hist(normed = True, bins = 50)


# ## Inicie sua análise a partir da parte 1 a partir daqui

# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[ ]:


def q1():
    q = [0.25, 0.50, 0.75]
    q1_norm, q2_norm, q3_norm = df['normal'].quantile(q)
    q1_binom, q2_binom, q3_binom = df['binomial'].quantile(q)
    q1 = (round(q1_norm - q1_binom, 3), round(q2_norm - q2_binom, 3), round(q3_norm - q3_binom, 3))
    return q1


# Para refletir:
# 
# * Você esperava valores dessa magnitude? É esperado valor de baixa magnitude, se plotarmos um gráfico de distribuição, veremos que existe pequena dispersão (baixo desvio padrão) em relação à média.
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[ ]:


def q2():
    ecdf = ECDF(df.normal)
    µ = df.normal.mean()
    s = df.normal.std()
    q2 = float((round(ecdf(µ + s) - ecdf(µ - s), 3)))
    return q2


# In[ ]:


ecdf = ECDF(df.normal)
µ = df.normal.mean()
s = df.normal.std()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico? Sim.
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# In[ ]:


(round(ecdf(µ + 2*s) - ecdf(µ - 2*s), 3))


# In[ ]:


(round(ecdf(µ + 3*s) - ecdf(µ - 3*s), 3))


# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[ ]:


def q3():
    m_binom = df['binomial'].mean()
    m_norm = df['normal'].mean()
    v_binom = df['binomial'].var()
    v_norm = df['normal'].var()
    q3 = ((round(m_binom - m_norm, 3)), (round(v_binom - v_norm, 3)))
    return q3


# Para refletir:
# 
# * Você esperava valores dessa magnitude? `Devido serem distribuições totalmente diferentes, os valores não foram esperados dessa maneira.`
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`? `Há um deslocamento do "pico" da distribuição.`

# ## Parte 2

# ### _Setup_ da parte 2

# In[ ]:


stars = pd.read_csv("HTRU_2.csv")

stars.rename({old_name: new_name
              for (old_name, new_name) in zip(stars.columns, ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", 
                      "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])}, axis=1, inplace= True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


stars2 = pd.DataFrame(stars.query('target == False'))


# In[ ]:


µmp = stars2['mean_profile'].mean()
smp = stars2['mean_profile'].std()
false_pulsar_mean_profile_standardized = (stars2['mean_profile'] - µmp) / smp


# In[ ]:


def q4():
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    q4 = (round(ecdf(norm.ppf(0.80)), 3), round(ecdf(norm.ppf(0.90)), 3), round(ecdf(norm.ppf(0.95)), 3))
    return q4


# Para refletir:
# 
# * Os valores encontrados fazem sentido? `Sim, uma vez que a distribuição possui similaridade normal.`
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`? `Pode dizer existe uma distribuição normal e com confiabilidade.`

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


def q5():
    quartil = np.percentile(false_pulsar_mean_profile_standardized, [25, 50, 75])
    q5 = (round(quartil[0]-norm.ppf(0.25), 3),round(quartil[1]-norm.ppf(0.50), 3),round(quartil[2]-norm.ppf(0.75), 3))
    return q5


# Para refletir:
# 
# * Os valores encontrados fazem sentido? `Sim, pois os valores de false_pulsar_mean_profile_standardized também possuem média 0 e variância igual a 1, ou seja, ambas as distribuições estão nas faixas de padronização.`
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`? `Pode dizer que é uma distribuição padronização com baixa variância e média igual a zero.`
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.

# In[ ]:




