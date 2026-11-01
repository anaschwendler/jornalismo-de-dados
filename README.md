# Jornalismo de Dados

## Processamento de Linguagem Natural: aplicações em jornalismo e pesquisa ;)

### Objetivos

* Extrair insights para projetos de pesquisa jornalística e acadêmica
* Mostrar exemplos do que já foi feito
* Metodologia básica por trás do uso de Processamento de linguagem natural

### Planejamento

* Introdução a PLN (o que é PLN)
* Por que é importante, como ajuda no dia-a-dia
* Como criar estudos (jupyter notebook)
* Scrapping de dados
* Como análisar usando PLN
* Resultados até agora

### Status

Material criado em 2017 para uma palestra e parado desde então. Em 2026
começou a ser atualizado, um passo pequeno de cada vez:

**Fase 1: fazer rodar de novo**

- [x] Plano de atualização neste README
- [x] Remover as exportações `.html` e `.py` geradas a partir do notebook
- [x] Projeto com uv e `pyproject.toml`, dependências atuais, sem as não usadas
- [x] Tokenizador em português no NLTK atual (`punkt_tab`) e download automático dos recursos
- [ ] Notebook executado do início ao fim, com as saídas salvas
- [ ] Erros de digitação e título ("Processamento de Linguagem Natural")
- [ ] Links quebrados ou que mudaram (INEP, Scrapinghub/Zyte, CSTNews)

**Fase 2: terminar o que ficou pela metade**

- [ ] Dividir em notebooks numerados (`01-introducao`, `02-limpeza`, ...)
- [ ] Escolher uma base de dados aberta e maior, com script de download
- [ ] Frequência de palavras com gráfico
- [ ] Nuvem de palavras
- [ ] Lematização com spaCy, comparada ao stemming
- [ ] Exemplo de raspagem de dados com requests e Beautiful Soup
- [ ] Seção "Resultados" preenchida

**Fase 3: PLN em 2026**

- [ ] Reconhecimento de entidades (pessoas, órgãos, lugares)
- [ ] Embeddings e busca por similaridade
- [ ] Classificação de textos (TF-IDF e scikit-learn)
- [ ] LLMs para extrair e resumir, com verificação dos resultados
- [ ] Estudo de caso jornalístico de ponta a ponta

**Fase 4: organização e divulgação**

- [ ] Código reutilizável em um módulo `src/`
- [ ] Notebooks executados no GitHub Actions
- [ ] Site com o material (Quarto ou Jupyter Book no GitHub Pages)
- [ ] README reescrito e licença definida

### Tecnologias
* [Python](https://www.python.org/)
* [Anaconda](https://www.anaconda.com/download/)
* [Jupyter notebook](http://jupyter.org/install.html)
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [Scrapy](https://docs.scrapy.org/en/latest/)
* [Pandas](http://pandas.pydata.org/pandas-docs/stable/)
* [Numpy](https://docs.scipy.org/doc/numpy-dev/dev/)
* [Scikit-learn](http://scikit-learn.org/stable/documentation.html)
* [Seaborn](https://seaborn.pydata.org/tutorial.html)

### Instalando módulos necessários

O projeto usa o [uv](https://docs.astral.sh/uv/getting-started/installation/)
para instalar o Python e as bibliotecas. Com o uv instalado:

```console
$ uv sync
$ uv run jupyter lab
```

### Bases de dados

[Em construção]

CSTCorpus: http://nilc.icmc.usp.br/CSTNews/login/about

http://download.inep.gov.br/microdados/microdados_censo_superior_2014.zip
http://portal.inep.gov.br/artigo/-/asset_publisher/B4AQV9zFY7Bv/content/id/1336683

