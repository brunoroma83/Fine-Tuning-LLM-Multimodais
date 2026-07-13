# Análise do Projeto: PCS5022 - Kaggle Multimodal

Este documento apresenta uma análise detalhada do estado atual do projeto para a competição Kaggle PCS5022 (USP), incluindo a identificação de bugs críticos, análises quantitativas de acurácia e recomendações práticas para melhorar a pontuação final.

---

## 1. Visão Geral do Projeto e Objetivos

- **Objetivo**: Classificação multimodal (imagem do circuito lógico + pergunta em texto).
- **Métrica de Avaliação**: Acurácia (match exato). As respostas numéricas devem conter apenas o número (ex: `3`, `0`) e as de decisão apenas `True` ou `False`.
- **Dataset**: 2.000 amostras de treino com labels, 1.000 amostras de teste.
- **Modelos Utilizados**: 
  - Inicialmente: Qwen2-VL-2B (treinamento lento: 24h para 4 épocas na máquina local).
  - Atual: Qwen2.5-VL-7B-Instruct-bnb-4bit (com Unsloth, ~15 minutos por época, total de ~1h para 4 épocas).
  - Planejado: Modelos maiores (>20B) via LM Studio / Test-Time Compute.

---

## 2. Análise Quantitativa dos Resultados de Validação

Analisamos o último arquivo de inferência da validação (`resultados_inferencia_2026-07-12_16-31-48.csv`):

- **Acurácia Geral**: **87,00%** (174/200 acertos)
- **Perguntas de Contagem de Portas (Gate Counting)**: **93,07%** de acurácia (94/101 acertos)
  - **AND gates**: 100,00% (31/31)
  - **NOT gates**: 100,00% (14/14)
  - **NAND gates**: 100,00% (22/22)
  - **NOR gates**: 93,33% (28/30)
  - **OR gates**: 87,50% (49/56)
  - **XOR gates**: 85,71% (12/14)
  - **XNOR gates**: 87,50% (14/16)
- **Perguntas de Propagação Lógica (Circuit Output)**: **80,81%** de acurácia (80/99 acertos)

### 💡 Insight Principal:
O modelo é extremamente competente em reconhecer e contar as portas lógicas individualmente (93.07%), mas apresenta uma **queda significativa de ~12%** na propagação de lógica (80.81%). Propagar valores através de uma cadeia de portas em um único passo de inferência direta (Direct Inference) é uma tarefa complexa para modelos de visão. Isso valida a decisão de focar em **Chain of Thought (CoT)** e **Test-Time Compute** especificamente para as perguntas de propagação de circuitos.

---

## 3. Problemas Técnicos e Bugs Identificados

### 🚨 Bug Crítico: Vazamento de Dados (Data Leakage) no Split de Validação
- **Arquivos**: [projeto_pcs5022.ipynb](file:///home/bruno/PCS5022-Kaggle/projeto_pcs5022.ipynb) (Células 20 e 23) e [projeto_pcs5022-chain-of-thought.ipynb](file:///home/bruno/PCS5022-Kaggle/projeto_pcs5022-chain-of-thought.ipynb) (Células 28 e 31).
- **Problema**: O objeto `SFTTrainer` é instanciado na Célula 28/20 recebendo `train_dataset = dataset_treino` (que neste momento possui 2.000 amostras). Logo depois, na Célula 31/23, o dataset é dividido em treino (1800 amostras) e validação (200 amostras). No entanto, o `SFTTrainer` mantém a referência ao dataset original completo. Logo, **o modelo é treinado sobre as 2.000 amostras, incluindo as 200 de validação**. Isso mascara a acurácia real de generalização (que atualmente mostra 87.00% mas provavelmente é menor em dados inéditos).
- **Como corrigir**: Mova a célula de split do dataset (Célula 31/23) para antes da inicialização do `SFTTrainer` (Célula 28/20).


### ⚠️ Inconsistência na Inferência de Teste com Chain of Thought
- **Arquivo**: [projeto_pcs5022-chain-of-thought.ipynb](file:///home/bruno/PCS5022-Kaggle/projeto_pcs5022-chain-of-thought.ipynb)
- **Problema**: Embora a técnica de Chain of Thought (CoT) e Self-Consistency tenha sido testada para 5 amostras na Célula 24, a Célula 56 (geração das previsões do conjunto de teste) continua gerando respostas sem CoT (inferência direta tradicional). Além disso, não há código para extrair o valor final da string gerada pelo CoT para formatação exigida pelo Kaggle.
- **Como corrigir**: Adaptar a Célula 56 para usar a estrutura Few-Shot CoT e implementar um extrator via regex/lógica de string para o CSV final.

---

## 4. Recomendações Estratégicas para o Projeto

1. **Correção Imediata do Split**: Corrigir o vazamento de dados nas duas notebooks para obter métricas de validação realistas.
2. **Implementar Extrator de Resposta CoT**: Criar uma função de pós-processamento simples para extrair o resultado final do texto longo de raciocínio. Exemplo:
   ```python
   import re
   def extrair_resposta(texto, tipo):
       if tipo == "boolean":
           matches = re.findall(r'\b(True|False)\b', texto, re.IGNORECASE)
           return matches[-1].capitalize() if matches else "False"
       else:
           # encontra o último número na resposta
           matches = re.findall(r'\b\d+\b', texto)
           return matches[-1] if matches else "0"
   ```
3. **Integrar Self-Consistency (Votação)**: 
   - Modificar a inferência de teste para gerar 3 respostas com CoT (usando temperatura de 0.7 a 1.0 e `min_p=0.1`).
   - Extrair a resposta de cada uma das 3 gerações e selecionar a resposta mais frequente (voto majoritário).
4. **Data Augmentation Lógico**: Como as imagens de circuitos são sintetizadas, o modelo pode ter dificuldade com a orientação ou qualidade. Se as imagens forem redimensionadas para 512px, certifique-se de que as conexões finas (linhas) e pequenos círculos (inversores de NOT/NAND/NOR) não fiquem borrados. Caso fiquem, aumentar `IMAGE_MAX_SIZE` para 768px ou usar processamento em resolução nativa no Qwen2.5-VL pode ajudar.
5. **LM Studio / Modelos Maiores**: Avaliar um modelo maior (>20B) apenas nas amostras onde o modelo 7B tiver maior incerteza (por exemplo, quando as 3 inferências de Self-Consistency divergirem). Isso economiza tempo de inferência local e foca o processamento pesado apenas nos casos difíceis.
