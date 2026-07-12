import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Treina o Qwen2-VL com LoRA para circuitos.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Quantidade de épocas de treino. Padrão: 1.",
    )
    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_training",
        help="Continua do checkpoint ou treinamento salvo mais recente.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("O número de épocas deve ser maior ou igual a 1.")

    return args


args = parse_args()


from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from datetime import datetime
import torch
from dotenv import load_dotenv
import kagglehub
import os
import json
from datasets import Dataset
from PIL import Image as PILImage, ImageOps
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm

load_dotenv()  # Load environment variables from .env file

OUTPUT_DIR = "./qwen2-vl-circuitos"
FINAL_MODEL_PREFIX = "./qwen2-vl-circuitos-final"
IMAGE_MAX_SIZE = 512
RESIZED_IMAGE_DIR = "./temp/resized_train_images"


def encontrar_treino_mais_recente(output_dir, final_model_prefix):
    candidatos = []

    if os.path.isdir(output_dir):
        for nome in os.listdir(output_dir):
            caminho = os.path.join(output_dir, nome)
            if nome.startswith("checkpoint-") and os.path.isdir(caminho):
                candidatos.append((os.path.getmtime(caminho), "checkpoint", caminho))

    pasta_base = os.path.dirname(final_model_prefix) or "."
    prefixo = os.path.basename(final_model_prefix)
    if os.path.isdir(pasta_base):
        for nome in os.listdir(pasta_base):
            caminho = os.path.join(pasta_base, nome)
            if nome.startswith(prefixo) and os.path.isdir(caminho):
                adapter_config = os.path.join(caminho, "adapter_config.json")
                if os.path.exists(adapter_config):
                    candidatos.append((os.path.getmtime(caminho), "lora_final", caminho))

    if not candidatos:
        return None, None

    _, tipo, caminho = max(candidatos, key=lambda item: item[0])
    return tipo, caminho


def redimensionar_imagem(caminho_imagem, pasta_saida, tamanho_max=IMAGE_MAX_SIZE):
    os.makedirs(pasta_saida, exist_ok=True)

    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]
    caminho_saida = os.path.join(pasta_saida, f"{nome_base}-{tamanho_max}.png")

    if os.path.exists(caminho_saida):
        return caminho_saida

    with PILImage.open(caminho_imagem) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail((tamanho_max, tamanho_max), PILImage.Resampling.LANCZOS)

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        img.save(caminho_saida, format="PNG", optimize=True)

    return caminho_saida


def exigir_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/GPU nao esta disponivel para o PyTorch neste ambiente. "
            "Verifique se o driver NVIDIA esta funcionando, se o Python esta rodando "
            "no ambiente correto e se a GPU esta visivel para o processo."
        )

    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"CUDA do PyTorch: {torch.version.cuda}")


exigir_gpu()


# Download latest version
path = kagglehub.competition_download('pcs-3838-pcs-5022-2026-parte-2')

print("Path to competition files:", path)

###
### Parte 1. Carregar os dados do JSON e imagens no formato do Qwen2-VL
###

# função para carregar os dados do JSON e imagens no formato do Qwen2-VL
def carregar_dados_qwen(jsonl_path, pasta_imagens, pasta_imagens_redimensionadas=RESIZED_IMAGE_DIR):

    lista_formatada = []

    # 1. Abrir o arquivo JSONL e ler linha por linha
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for linha in tqdm(f, desc="Carregando dados"):
            # Ignora linhas vazias que possam existir no final do arquivo
            if not linha.strip():
                continue
                
            # Transforma a linha atual em um dicionário Python
            item = json.loads(linha)
            
            # 2. Extrair os dados (ajuste os nomes das chaves se forem diferentes)
            idx = item['index'] 
            pergunta = item['question']
            resposta = item.get('answer', '') 

            pergunta_formatada = f"<image>\n{pergunta}"
            
            # Construir o caminho da imagem baseado no padrão 'circuit_n.png'
            caminho_imagem = os.path.join(pasta_imagens, f"circuit_{idx}.png")

            # verificar se a imagem existe
            if not os.path.exists(caminho_imagem):
                print(f"Imagem não encontrada: {caminho_imagem}")
                continue

            caminho_imagem = redimensionar_imagem(
                caminho_imagem,
                pasta_imagens_redimensionadas,
            )
            
            # Estrutura de mensagens do Qwen2-VL
            mensagens = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": caminho_imagem},
                        {"type": "text", "text": pergunta_formatada}
                    ]
                }
            ]
            
            # Se for dado de treino, adicionamos a resposta do assistente
            if resposta:
                mensagens.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": resposta}
                    ]
                })
                
            lista_formatada.append({
                "messages": mensagens,
                "id": idx
            })
        
    # 3. Converter para um Dataset do Hugging Face
    dataset = Dataset.from_list(lista_formatada)
    return dataset


# Iniciando o carregamento dos dados do treino
print("Carregando dados do treino...\n")
train_path = os.path.join(path, 'kaggle_dataset/train_dataset')  # Ajuste conforme necessário

train_json_path = os.path.join(train_path, 'text/questions_with_answers.jsonl')  # Ajuste conforme necessário
train_image_path = os.path.join(train_path, 'images/')  # Ajuste conforme necessário
dataset_treino = carregar_dados_qwen(train_json_path, train_image_path)

# analisa uma entrada do dataset
print(dataset_treino[0])  # Mostra a primeira entrada do dataset

print(f"Total de exemplos no dataset de treino: {len(dataset_treino)}\n")

###
### Parte 2. Carregar o modelo e processor Qwen2-VL
###

# 1. Definir o caminho do modelo no Hugging Face
model_id = "Qwen/Qwen2-VL-2B-Instruct"

# 2. Carregar o processador (cuida de texto e imagem simultaneamente)
processor = AutoProcessor.from_pretrained(model_id)

# 3. Carregar o modelo configurado para a sua GPU
# Usamos device_map="auto" para o Transformers alocar a memória automaticamente
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map={"": 0},
    attn_implementation="sdpa"
)

tipo_continuacao = None
caminho_continuacao = None

if args.continue_training:
    tipo_continuacao, caminho_continuacao = encontrar_treino_mais_recente(
        OUTPUT_DIR,
        FINAL_MODEL_PREFIX,
    )

    if tipo_continuacao == "lora_final":
        print(f"Continuando a partir do LoRA salvo mais recente: {caminho_continuacao}")
        model = PeftModel.from_pretrained(
            model,
            caminho_continuacao,
            is_trainable=True,
        )
    elif tipo_continuacao == "checkpoint":
        print(f"Continuando a partir do checkpoint mais recente: {caminho_continuacao}")
    else:
        print("Nenhum checkpoint ou LoRA final encontrado. Iniciando treinamento do zero.")

print("Modelo e Processor carregados com sucesso!")
print(f"Device principal do modelo: {next(model.parameters()).device}")

# Entendendo a Colação de Dados (Data Collator)
# No treino de redes neurais, as imagens e textos têm tamanhos diferentes, mas a GPU precisa recebê-los em blocos padronizados 
# (batches). O papel do Data Collator é pegar uma lista de exemplos do seu dataset, passá-los pelo processor e gerar o batch final.
# Para o Qwen2-VL, a própria documentação sugere uma função de colação simples. 

def limpar_campos_vazios_mensagens(messages):
    mensagens_limpas = []

    for message in messages:
        conteudo_limpo = []
        for item in message["content"]:
            conteudo_limpo.append({
                chave: valor
                for chave, valor in item.items()
                if valor is not None
            })

        mensagens_limpas.append({
            "role": message["role"],
            "content": conteudo_limpo,
        })

    return mensagens_limpas


def data_collator(examples):
    conversations = [
        limpar_campos_vazios_mensagens(example["messages"])
        for example in examples
    ]
    
    # Aplica o template correto para que o processador entenda onde ficam os tokens de imagem
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in conversations
    ]
    
    image_inputs, _ = process_vision_info(conversations)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs

###
### Parte 3. LoRA - redução de parâmetros treináveis
###

# 1. Configuração do LoRA
peft_config = LoraConfig(
    r=16, # Rank: quanto maior, mais parâmetros treináveis (16 ou 32 são bons valores)
    lora_alpha=32, # Escala do aprendizado
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Camadas alvo
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", # O Qwen2-VL se comporta como um modelo de linguagem causal
)

# 2. Aplicar o LoRA ao modelo carregado
#model = get_peft_model(model, peft_config)

# Mostrar quantos parâmetros estamos realmente treinando (geralmente < 2%)
#model.print_trainable_parameters()

# Configurar o Trainer (SFTTrainer)
# Para facilitar sua vida, usaremos o SFTTrainer da biblioteca trl. Ele é especializado em "Supervised Fine-Tuning" e 
# integra perfeitamente com o nosso data_collator.
# Aqui você define os "hiperparâmetros" da rede. 
# Atenção: Como o dataset tem 2000 imagens, o treino pode demorar. Comece com 1 ou 2 épocas para testar.

# 1. Configurações de Hiperparâmetros
training_args = SFTConfig(
    output_dir=OUTPUT_DIR, # Onde salvar o modelo treinado
    #max_steps=1000,              # Número total de passos (ou use num_train_epochs)
    num_train_epochs=args.epochs, # Número de épocas (passadas completas pelo dataset)
    per_device_train_batch_size=2, # Reduza para 1 se der erro de memória (Out of Memory)
    gradient_accumulation_steps=4, # Aumenta o "batch virtual" para 8 (2 * 4)
    learning_rate=1e-4,          # Taxa de aprendizado clássica para LoRA
    logging_steps=10,            # Log a cada 10 passos
    save_steps=100,              # Salva um checkpoint a cada 100 passos
    bf16=True if torch.cuda.is_bf16_supported() else False, # Precisão mista
    fp16=False if torch.cuda.is_bf16_supported() else True,
    remove_unused_columns=False, # Importante para modelos multimodais
    dataset_text_field="messages", # Não usado diretamente pois temos o collator
    dataset_kwargs={"skip_prepare_dataset": True}
)

trainer_kwargs = {
    "model": model,
    "train_dataset": dataset_treino, # O dataset que criamos na Fase 1
    "data_collator": data_collator,  # A função de colação da Fase 2
    "args": training_args,
    "processing_class": processor,
}

if tipo_continuacao != "lora_final":
    trainer_kwargs["peft_config"] = peft_config

# 2. Inicializar o Trainer
trainer = SFTTrainer(
    **trainer_kwargs
)

print("Tudo pronto para começar o treinamento!")

if tipo_continuacao == "checkpoint":
    trainer.train(resume_from_checkpoint=caminho_continuacao)
else:
    trainer.train()

# Após o treino, salve os adaptadores LoRA
trainer.save_model(f"{FINAL_MODEL_PREFIX}-epochs-{args.epochs}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
