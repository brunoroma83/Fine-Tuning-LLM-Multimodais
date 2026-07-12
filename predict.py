import os
import json
import csv
from datetime import datetime
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image as PILImage, ImageOps
load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
PATH = "/home/bruno/.cache/kagglehub/competitions/pcs-3838-pcs-5022-2026-parte-2"
INPUT_PATH_TEXT = os.path.join(PATH, "kaggle_dataset/test_dataset/text/questions.jsonl")  # Arquivo JSONL com perguntas de teste
PASTA_IMAGENS_TEST = os.path.join(PATH, "kaggle_dataset/test_dataset/images")       # Pasta contendo circuit_0.jpg, etc.
PASTA_IMAGENS_TEST_RESIZED = "./resized_test_images"
IMAGE_MAX_SIZE = 512
MODEL_BASE_ID = "Qwen/Qwen2-VL-2B-Instruct"
DIRETORIO_LORA = "./qwen2-vl-circuitos-final-epochs-4-2026-07-03-01-20-00"        # Onde o SFTTrainer salvou o modelo
OUTPUT_PATH = f"submission-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"

# --- 2. CARREGAR MODELO E PROCESSADOR ---
print("Carregando o modelo base e o processador...")
processor = AutoProcessor.from_pretrained(MODEL_BASE_ID)

# Carrega o modelo base original em 16-bits
model_base = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_BASE_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# Injeta os adaptadores LoRA que você treinou no modelo base
print("Acoplando os adaptadores LoRA treinados...")
model = PeftModel.from_pretrained(model_base, DIRETORIO_LORA)
model.eval() # Coloca o modelo em modo de avaliação (desativa dropout)


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


# --- 3. FUNÇÃO DE PREDIÇÃO COM O QWEN2-VL ---
def predict_qwen(entry: dict):
    idx = entry["index"]
    pergunta = entry["question"]
    
    # Monta o caminho da imagem de teste correspondente
    caminho_imagem = os.path.join(PASTA_IMAGENS_TEST, f"circuit_{idx}.png")

    # verfica se a imagem existe
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    caminho_imagem = redimensionar_imagem(
        caminho_imagem,
        PASTA_IMAGENS_TEST_RESIZED,
    )
        
    # Formata a mensagem no padrão que o modelo espera (apenas o bloco do User)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": caminho_imagem},
                {"type": "text", "text": f"<image>\n{pergunta}"}
            ]
        }
    ]
    
    # Prepara o prompt de chat sem o bloco do assistente (o modelo vai completar)
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # O processador transforma a imagem e o texto em tensores para a GPU
    # Como processamos 1 por 1, não precisamos extrair o video_inputs
    #image_input, _, _ = processor.image_processor.preprocess(images=caminho_imagem, return_tensors="pt").values() # Fallback seguro
    
    # Uma forma mais direta e recomendada usando a biblioteca utilitária do Qwen:
    from qwen_vl_utils import process_vision_info
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda") # Garante que os inputs vão para a GPU
    
    # Gera a resposta
    with torch.no_grad(): # Desativa o cálculo de gradientes para economizar memória e ir mais rápido
        generated_ids = model.generate(**inputs, max_new_tokens=50) # Circuitos lógicos geralmente têm respostas curtas
        
        # Recorta os tokens do prompt original para ficar apenas com a resposta gerada
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Converte os números gerados de volta para texto puro
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
    return output_text.strip()

# --- 4. LAÇO PRINCIPAL DA SUBMISSÃO ---
def generate_submission(input_path: str, output_path: str):
    print("Iniciando a geração das respostas de teste...")
    with open(input_path) as f, open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["index", "answer"]) # Cabeçalho exigido pelo Kaggle

        total = sum(1 for _ in f)
        f.seek(0)
        
        for line in tqdm(f, total=total, desc="Gerando submissão", unit="exemplo"):
            entry = json.loads(line)
                
            resposta_modelo = predict_qwen(entry)
            writer.writerow([entry["index"], resposta_modelo])
            
    print(f"\nFeito! Arquivo gravado com sucesso em {output_path}")

if __name__ == "__main__":
    generate_submission(INPUT_PATH_TEXT, OUTPUT_PATH)
