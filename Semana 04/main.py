#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import pylab


# In[16]:


#%matplotlib inline

from IPython.core.pylabtools import figsize

figsize(12, 8)

sns.set()


# In[17]:


df = pd.read_csv("athletes.csv")


# In[18]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[19]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[22]:


def q1():
    sampleheight = get_sample(df, 'height', n=3000)
    shs = sct.shapiro(sampleheight)
    return bool(shs[1] > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que? `Sim, pois, pelo teste, não podemos afimar que a amostra não saiu de uma distriuição normal, ou seja, o teste dá como conclusão que ela pode ter saído ou não.`
# * Plote o qq-plot para essa variável e a analise. `Pelo QQ-plot, percebemos que os valores estão dentro dos quartis-quantis esperados, ou seja, mais um indício que a amostra advém de uma distribuição normal.`
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).`Não, pois o menor valor de alfa é de 0.001 e o p-valor é muito menor que isso`

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[23]:


def q2():
    sampleheight = df['height'].sample(n = 3000)
    shjb = sct.jarque_bera(sampleheight)
    return bool(shjb[1] > 0.05)


# __Para refletir__:
# 
# * Esse resultado faz sentido? `Uma vez que os valores são próximos de zero, sim!`

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[24]:


def q3():
    sampleweight = df['weight'].sample(n = 3000)
    k2, p = sct.normaltest(sampleweight, nan_policy='omit')
    return bool(p > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[25]:


def q4():
    logweight = np.log(df['weight'])
    samplelogweight = logweight.sample(n = 3000)
    k2, p = sct.normaltest(logweight)
    return bool(p > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[26]:


def q5():
    bra = df[df['nationality']=='BRA']['height'].dropna()
    usa = df[df['nationality']=='USA']['height'].dropna()
    can = df[df['nationality']=='CAN']['height'].dropna()
    
    ttest = sct.ttest_ind(bra, usa)
    
    return bool(ttest [1] > 0.025)


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[27]:


def q6():
    bra = df[df['nationality']=='BRA']['height'].dropna()
    usa = df[df['nationality']=='USA']['height'].dropna()
    can = df[df['nationality']=='CAN']['height'].dropna()
    
    ttest = sct.ttest_ind(bra, can)
    
    return bool(ttest [1] > 0.025)


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[28]:


def q7():
    usa = df.query('nationality in "USA"')['height']
    can = df.query('nationality in "CAN"')['height']
    
    ttest = sct.ttest_ind(usa, can, equal_var = False, nan_policy = 'omit')

    return float(round(ttest.pvalue, 8))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
