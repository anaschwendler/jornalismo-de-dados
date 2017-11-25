
# coding: utf-8

# # Processamento Natural de Linguagem [PLN]
# 
# - O que é?
# - Por que é importante?
# - Ferramentas disponíveis
# - Scrapping de dados
# - Como análisar usando PLN e exemplos atuais
# - Referências

# ## O que é?
# 
# A área de Inteligência Artificial que busca técnicas de como uma máquina pode compreender a linguagem humana.
# Para isso, são necessárias ferramentas que permitam análise da linguagem como ela é. Por exemplo: análise sintática, semântica, léxica..

# ## Por que é importante?
# 
# Com PLN somos capazes de análisar grandes quantidades de dados quase que instantâneamente, usando ferramentas que auxiliam em várias tarefas que seriam impossíveis de serem completadas apenas com trabalho de análise manual.
# 
# Existem algumas tarefas básicas que podem ser realizadas com PLN:
# 
# **1 - Normalização de strings:**
# 
# Para facilitar a análise de grandes bases de dados, podemos transformar letras maiúsculas em minúsculas, remover acentos, caracteres especiais e talvez conteúdo que não interesse na análise que buscamos.
# 
# Exemplo: método de normalização de strings
# 
# **2 - Remoção de stopwords**
# 
# Se buscamos saber quais são as palavras mais utilizadas em um texto, de forma que apareça apenas o conteúdo que nos interessa, precisamos remover palavras muito frequentes da nossa língua, como por exemplo: "a", "de", "que". Para isso usamos métodos de automatização com PLN.
# 
# Exemplo: método de remoção do NLTK, wordclouds
# 
# **3 - Correção ortográfica**
# 
# Existem formas de corrigir erros de digitação com PLN, para isso podemos usar diversas estratégias, como por exemplo: extrair os radicais das palavras, fazer comparações com outras palavras da língua. Se uma palavra não é parecida o suficiente com outras da língua em questão, pode ser que esteja mal escrita.
# 
# Exemplo: Stemming e lematização

# ## Ferramentas disponíveis
# 
# Como podemos análisar textos, bases de dados na prática? Algumas ferramentas que experimento:
# 
# **1 - Python e Anaconda**
# 
# Python é uma linguagem de programação que tem sido amplamente utilizada para estudos de inteligência artificial, por permitir que tenhamos bons resultados, usando um código mais simples, mas significativo. Não faz tudo sozinho, mas também nos dá flexibilidade para programar.
# 
# Python já possui inúmeros módulos, que facilitam a construção de códigos que nos auxiliem no uso de PLN. Abaixo vou explicar a fundo alguns que conheço e faço uso diaramente.
# 
# Anaconda[5] é uma distribuição de python e R (outra linguagem para estudos de dados, focado em estatística) para processamento de bases de dados grandes. Com ela, já vem incluído vários módulos úteis para nosso desenvolvimento.
# 
# **2 - Jupyter Notebook[6]**
# 
# Aplicação que permite a criação e compartilhamento de documentos que contém tanto código quanto descrições narrativas de análises com bases grandes de dados.
# 
# **3 - NLTK[1]: Natural Language Toolkit**
# 
# Um dos principais módulos em python para construir códigos que trabalham com dados de linguagem humana. 
# 
# Tem uma versão em português: http://www.nltk.org/howto/portuguese_en.html
# 
# **4 - Pandas[7]**
# 
# Módulo que simplifica a manipulação de bases grandes de dados.
# 
# **5 - Numpy[8]**
# 
# Módulo para computação científica em python. É usada para cálculo estatistíco, junto com Pandas.

# ## Scrapping de dados
# 
# Python possui também ferraments que ajudam a criar bases de dados, sendo assim, podemos buscar o conteúdo de várias páginas da internet e compilá-los usando ferramentas que facilitam essa busca.
# 
# **1 - Beautiful Soup[9]**
# 
# Módulo para pegar dados de HTML e XML na web, com uso simples, consegue buscar dados de várias fontes diferentes.
# 
# 
# **2 - Scrapy[10]**
# 
# Outro módulo para extração de arquivos da web.

# ## Como análisar usando PLN
# 
# Base de dados utilizada: [CSTCorpus](http://nilc.icmc.usp.br/CSTNews/login/about)
# 
# **1 - Normalização de Strings**
# 
# **2 - Remoção de Stopwords**
# 
# **3 - Wordcloud**
# 
# **4 - Stemming e Lematização**

# In[ ]:


import nltk


# In[ ]:


# Leitura do Sample

f = open('../data/sample.txt')
dataset = f.read()

print(dataset)


# ### Tokenização em sentenças, palavras e expressão regulares
# 
# Token é um pedaço de um todo, então:
# 
# uma palavra é um token em uma sentença;
# uma sentença é um token em um paragrafo.

# #### Tokenização de sentenças

# In[ ]:


from nltk.tokenize import sent_tokenize

sentence_tokenized = sent_tokenize(dataset)
print(sentence_tokenized)


# In[ ]:


len(sentence_tokenized)


# ### Normalização de texto

# In[ ]:


import unicodedata 

def normalize_string(string):
    if isinstance(string, str):
        nfkd_form = unicodedata.normalize('NFKD', string.lower())
        return nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    
print(normalize_string(sentence_tokenized[0]))


# #### Tokenização em Português

# In[ ]:


import nltk.data
portuguese_tokenizer = nltk.data.load('tokenizers/punkt/PY3/portuguese.pickle')
portuguese_tokenizer.tokenize(dataset)


# #### Tokenização em palavras 
# 
# Para a primeira sentença do texto.

# In[ ]:


from nltk.tokenize import word_tokenize

first_sentence_word_tokenized = word_tokenize(sentence_tokenized[0])
print(first_sentence_word_tokenized)


# #### Tokenização com expressão regular
# 
# Para pegar apenas palavras em um texto.

# In[ ]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[\w']+")

first_sentence_word_tokenized_without_punctuation = tokenizer.tokenize(sentence_tokenized[0])
print(first_sentence_word_tokenized_without_punctuation)


# ### Filtrando stopwords!
# 
# Stopwords são palavras que geralmente não contribuem para o significado de uma sentença.

# In[ ]:


from nltk.corpus import stopwords

portuguese_stops = set(stopwords.words('portuguese'))
words = first_sentence_word_tokenized_without_punctuation

words_without_stop = [word for word in words if word not in portuguese_stops]
print(words_without_stop)


# ### Stemming
# 
# Stemming é a técnica que remove os afixos das palavras, deixando apenas seu radical, existe uma versão em Português que é `RSLPStemmer`

# In[ ]:


from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
stem_acidente = stemmer.stem(words_without_stop[1]) #acidente
print(stem_acidente)


# ## Referências
# * [1]: http://www.nltk.org/
# * [2]: https://medium.com/botsbrasil/o-que-%C3%A9-o-processamento-de-linguagem-natural-49ece9371cff* 
# * [3]: https://github.com/generonumero/tse_candidatos_2016/blob/master/GN_receitas_candidatos_2016.ipynb
# * [4]: https://github.com/rafapetter/suspeitando/blob/master/analise/licitacoes.ipynb
# * [5]: https://anaconda.org/
# * [6]: http://jupyter.org/
# * [7]: https://pandas.pydata.org/
# * [8]: http://www.numpy.org/
# * [9]: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# * [10]: https://scrapy.org/
# 
# 
# Livro: https://github.com/karanmilan/Automatic-Answer-Evaluation/blob/master/Python%203%20Text%20Processing%20with%20NLTK%203%20Cookbook.pdf
# 
# WordCloud: https://github.com/rafapetter/suspeitando/blob/master/analise/licitacoes.ipynb
