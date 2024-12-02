# Análise de Clusters com K-Means - Reconhecimento de Atividades Humanas

## Descrição do Projeto

Este projeto aplica o algoritmo de **K-Means** ao dataset **"Human Activity Recognition Using Smartphones"**, que contém dados de sensores de acelerômetro e giroscópio coletados durante atividades humanas. O objetivo é agrupar os dados em clusters que representem padrões de atividades semelhantes, utilizando técnicas como **PCA** para visualização e análise dos agrupamentos.

## Instalação

Para rodar o projeto, siga os passos abaixo:

1. **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/kmeans_har.git
    ```

2. **Entre no diretório do projeto:**
    ```bash
    cd kmeans_har
    ```

3. **Crie e ative um ambiente virtual (opcional, mas recomendado):**
    - Para **Linux/macOS**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    - Para **Windows**:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

4. **Instale as dependências necessárias:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

## Como Executar

Após a instalação, siga os passos abaixo para rodar o código:

1. **Baixe o dataset** **"UCI HAR Dataset"** e descompacte-o no diretório `./data/` do projeto.

2. **Execute o script Python**:
    ```bash
    python main.py
    ```

   O código realiza as seguintes etapas:

   - Carregamento e normalização dos dados.
   - Redução de dimensionalidade com **PCA**.
   - Determinação do número ideal de clusters utilizando os métodos do **Cotovelo** e **Silhouette Score**.
   - Criação de gráficos para visualização dos clusters e dos resultados dos métodos.
   - Análise das características de cada cluster e distribuição de atividades.

## Estrutura dos Arquivos

- `/data`: Contém o dataset **"UCI HAR Dataset"**.
- `/docs`: Relatório técnico com os resultados e análises do projeto.
- `main.py`: Código-fonte principal que implementa a análise e agrupamento.
- `README.md`: Este arquivo com as instruções do projeto.

## Tecnologias Utilizadas

- **Python**: Linguagem principal para análise e modelagem.
- **Pandas**: Manipulação e análise de dados.
- **NumPy**: Operações matemáticas e manipulação de arrays.
- **Matplotlib** e **Seaborn**: Bibliotecas para visualização de dados.
- **Scikit-learn**: Implementação do algoritmo K-Means, métricas de avaliação e validação de clusters.

## Autores e Colaboradores

- **Fláira Hanny Bomfim dos Santos**: Desenvolvimento do código e análise de clusters.
- **Lis Loureiro Sousa**: Apoio em visualização de dados e validação dos resultados.
