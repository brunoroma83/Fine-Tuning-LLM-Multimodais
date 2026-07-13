# Registros do trabalho

## Treinar um modelo Qwen2-VL-2B com LoRA
### multimodel.py
1. Primeiro modelo que fiz, treinamento lento. 
2. Treinar 4 épocas demora 24 horas na minha máquina.
3. Não consegui treinar no Google Colab pois estourava a memória, encontrei uma solução com o Unsloth.

### Unsloth: projeto_pcs5022.ipynb
1. Treina o Qwen2.5-VL-7b-Instruct-4bit com LoRA
2. Feito com Unsloth. Treinamento mais rápido.
3. O mesmo treinamento de 4 épocas aqui dura cerca de 1 hora.
4. Fiz redução das dimensões da imagem para acelerar o treinamento e as inferencias
5. Treinamento:
- cria a pasta qwen_lora com as informações do treinamento
- cria a pasta outputs com os checkpoints do treinamento
- se carregar novamente o último modelo lora treinado, ele continuará o treinamento.
- no conjunto de validação (200 amostras do conjunto de treinamento) o modelo está acertando cerca de 87%
- foram treinadas 10 épocas, cada época demora cerca de 15 minutos

### Unsloth: projeto_pcs5022-chain-of-thought.ipynb
1. Não consegui bons resultados com o Chain of Thought, talvez seja por conta do tamanho do modelo (7B), talvez precise de um modelo maior.

## Rodar um modelo grande (>20B) no LM Studio
1. Tentar fazer as inferencias usando:
- Avalição sem modificações: simplesmente enviando as imagens e as perguntas para serem respondidas
- Chain of Thought
- Self-Consistency (gerar 3 chain of thoughts diferentes)
- Test-Time Compute (selecionar a melhor resposta de 3 ou 5)