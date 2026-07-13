# Competição Kaggle PCS5022 Redes Nerais e Aprendizado Profundo
URL: https://www.kaggle.com/competitions/pcs-3838-pcs-5022-2026-parte-2

## Overview
Esta competição foi organizada para as disciplinas PCS3838 - Inteligência Artificial e PCS5022 da Escola Politécnica da USP. Nesta competição, você deve utilizar modelos multimodais para resolver um problema de classificação.

Instrução para o nome do time O nome do seu time (dupla) deve seguir o seguinte formato: NUSPAluno1_NUSPAluno2

Entrega: 9 de agosto de 2026

## Description
Considere dois conjuntos de dados (treino e teste), cujas amostras são compostas por uma imagem e uma pergunta em texto. O conjunto de treino possui 2.000 amostras com as respostas/labels (Y). O conjunto de teste possui 1.000 amostras. Sua tarefa é gerar as respostas (idealmente) corretas para essas amostras e submetê-las nesta competição.

## Sugestões
Devido ao elevado custo computacional dos modelos multimodais, sugerimos iniciar as tentativas com técnicas de test-time computing.

## Avaliação
As submissões serão avaliadas com base na habilidade preditiva do modelo no conjunto de teste. Utilizaremos a acurácia como métrica de avaliação, seguindo:

$Acurácia = \frac{Número de Previsões Corretas / Número Total de Previsões}$

## Formato da Submissão
A submissão deve ser um arquivo CSV com duas colunas: index, answer.

"index" refere-se ao número da pergunta, enquanto a coluna "answer" corresponde à previsão realizado pelo modelo. É importante lembrar que a correção é feita por correspondência exata; portanto, questões de resposta numérica devem conter apenas o número, enquanto questões de decisões devem conter apenas as palavras True ou False.

Abaixo está um exemplo de como o arquivo de submissão deve ser formatado:

`csv
index, answer
1,3
2,True
3,0
`

## Exemplo de Código para Geração de Submissão
Abaixo encontra-se um código Python para geração de uma submissão de exemplo.

`python
import json
import csv

INPUT_PATH_TEXT = "/kaggle_dataset/test_dataset/text/questions.jsonl"
OUTPUT_PATH = "submission_dummy.csv"


def predict(entry: dict):
    question = entry["question"]
    if question.startswith("What is the output"):
        return True
    else:
        return 1


def generate_submission(input_path: str, output_path: str):
    with open(input_path) as f, open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["index", "answer"])
        for line in f:
            entry = json.loads(line)
            writer.writerow([entry["index"], predict(entry)])
    print(f"Written to {output_path}")


if __name__ == "__main__":
    generate_submission(INPUT_PATH_TEXT, OUTPUT_PATH)
`

## Execuções até o momento
Os registros sobre as execuções dos códigos estão em Registros.md