# Jornalismo de Dados

## Processamento de Linguagem Natural: aplicações em jornalismo e pesquisa ;)

### Objetivos

* Extrair insights para projetos de pesquisa jornalística e acadêmica
* Mostrar exemplos do que já foi feito
* Metodologia básica por trás do uso de Processamento de linguagem natural

### Planejamento

* Introdução a PLN (o que é PLN)
* Por que é importante, como ajuda no dia a dia
* Como criar estudos (jupyter notebook)
* Raspagem de dados
* Como analisar usando PLN
* Resultados até agora

### Como rodar

O projeto usa o [uv](https://docs.astral.sh/uv/), que instala o Python e todas
as bibliotecas necessárias. Você não precisa ter o Python instalado antes.

1. Instale o uv. No macOS ou Linux, no terminal:

   ```console
   $ curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   No Windows, no PowerShell:

   ```console
   > powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Baixe este repositório e entre na pasta:

   ```console
   $ git clone https://github.com/anaschwendler/jornalismo-de-dados.git
   $ cd jornalismo-de-dados
   ```

3. Instale as bibliotecas e abra o Jupyter:

   ```console
   $ uv sync
   $ uv run jupyter lab
   ```

4. O Jupyter abre no navegador. Entre na pasta `research/`, abra o notebook e
   rode as células em ordem (Shift+Enter). Na primeira vez, o notebook baixa
   alguns recursos do NLTK, então é preciso estar conectado à internet.

### Status

Material criado em 2017 para uma palestra e parado desde então. Em 2026
começou a ser atualizado, um passo pequeno de cada vez:

**Fase 1: fazer rodar de novo**

- [x] Plano de atualização neste README
- [x] Remover as exportações `.html` e `.py` geradas a partir do notebook
- [x] Projeto com uv e `pyproject.toml`, dependências atuais, sem as não usadas
- [x] Tokenizador em português no NLTK atual (`punkt_tab`) e download automático dos recursos
- [x] Notebook executado do início ao fim, com as saídas salvas
- [x] Erros de digitação e título ("Processamento de Linguagem Natural")
- [x] Links quebrados ou que mudaram (INEP, Scrapinghub/Zyte, CSTNews)

**Fase 2: terminar o que ficou pela metade**

- [ ] Revisar o texto do notebook (correção ortográfica x stemming, exemplos que faltam, Anaconda)
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
* [Anaconda](https://www.anaconda.com/download)
* [Jupyter notebook](https://jupyter.org/install)
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [Scrapy](https://docs.scrapy.org/en/latest/)
* [Pandas](https://pandas.pydata.org/docs/)
* [Numpy](https://numpy.org/doc/stable/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Seaborn](https://seaborn.pydata.org/tutorial.html)

### Bases de dados

[Em construção]

* [CSTNews](https://sites2.icmc.usp.br/taspardo/sucinto/cstnews.html): 140
  notícias em português, agrupadas em 50 assuntos, do NILC (ICMC-USP). É
  distribuído para uso em pesquisa. O texto de exemplo em `data/sample.txt` é
  uma dessas notícias.
* [Microdados do Censo da Educação Superior](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior),
  do INEP ([arquivo de 2014](https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2014.zip)).

