---

editor_options: 
  markdown: 
    wrap: 72
---

Este repositório contém o **Projeto 1** desenvolvido para a disciplina/módulo de **Perspectiva de Dados**. O objetivo principal é construir um ecossistema completo de Ciência de Dados, passando pela coleta de dados na web, processamento de linguagem natural (NLP), engenharia de atributos (features), modelagem estatística/multimodal e, por fim, a disponibilização de uma interface de usuário funcional.

------------------------------------------------------------------------

## 🚀 Visão Geral do Projeto

O projeto consiste em um **Sistema de Recomendação de Animes** que analisa múltiplas características (como sinopse, gêneros e imagem) para sugerir títulos semelhantes ao usuário.

Diferente de recomendadores simples baseados apenas em uma métrica (como nota), este motor utiliza **Processamento de Linguagem Natural (NLP)** para compreender o contexto textual e semântico.

------------------------------------------------------------------------

## 🛠️ Estrutura do Pipeline de Dados

O projeto está dividido em etapas pelos notebooks e scripts do repositório:

1.  **Coleta de Dados (Web Scraping)** (`Scraping.ipynb`)
    - Extração automatizada de informações detalhadas sobre animes diretamente da web.
    - Gera o arquivo bruto `banco_de_dados_animes.csv`.
2.  **Engenharia de Atributos & NLP** (`BoW_TF-IDF.ipynb`)
    - Limpeza e tratamento dos dados textuais (remoção de stopwords, pontuações, etc.).
    - Aplicação de duas técnicas clássicas de vetorização de texto:
      - **Bag of Words (BoW)** (`animes_bow.csv`)
      - **TF-IDF (Term Frequency-Inverse Document Frequency)** (`animes_tfidf.csv`)
    - Exportação da base de dados higienizada: `banco_de_dados_animes_limpo.csv`.
3.  **Modelagem do Motor Multimodal** (`Recomendar_multimodal.ipynb`)
    - Fusão das informações textuais vetorizadas com metadados estruturados (gêneros, sinópse, imagem).
    - Cálculo de matrizes de similaridade (ex: Similaridade de Cosseno) para identificar itens correlacionados.
    - Salvamento do modelo final compactado em `banco_de_dados_animes_vetorizado.pkl`.
4.  **Interface da Aplicação** (`main.py` & `app_recomendador.html`)
    - Camada de aplicação que carrega a base vetorizada e disponibiliza o recomendador de forma interativa para o usuário final através de uma página web.
