import os
from datetime import datetime
# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
PATH = "/home/bruno/.cache/kagglehub/competitions/pcs-3838-pcs-5022-2026-parte-2"
INPUT_PATH_TEXT = os.path.join(PATH, "kaggle_dataset/test_dataset/text/questions.jsonl")  # Arquivo JSONL com perguntas de teste
PASTA_IMAGENS_TEST = os.path.join(PATH, "kaggle_dataset/test_dataset/images")       # Pasta contendo circuit_0.jpg, etc.
PASTA_IMAGENS_TEST_RESIZED = "./temp/resized_test_images"
IMAGE_MAX_SIZE = 512
#MODEL_BASE_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_BASE_ID = "Qwen/Qwen2-VL-7B-Instruct"
DIRETORIO_LORA = "./temp/qwen2-vl-circuitos-final"        # Onde o SFTTrainer salvou o modelo
OUTPUT_PATH = f"submission-v1-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"

OUTPUT_DIR = "./temp/qwen2-vl-circuitos"
FINAL_MODEL_PREFIX = "./temp/qwen2-vl-circuitos-final"
IMAGE_MAX_SIZE = 512
RESIZED_IMAGE_DIR = "./temp/resized_train_images"