from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
import requests
import threading
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


def start_timer(message="Processando"):
    stop_event = threading.Event()

    def timer():
        start_time = time.perf_counter()
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start_time
            print(f"\r{message}: {elapsed:.1f}s", end="", flush=True)
            time.sleep(0.1)

        elapsed = time.perf_counter() - start_time
        print(f"\r{message}: {elapsed:.1f}s concluido")

    thread = threading.Thread(target=timer, daemon=True)
    thread.start()
    return stop_event, thread

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG")  # Replace with your image URL
image = Image.open(BytesIO(response.content))

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "Descreva esta imagem em detalhes."
            }
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
)

inputs = inputs.to("cuda")  # Move to GPU if available

timer_stop, timer_thread = start_timer("Gerando resposta")
try:
    output_ids = model.generate(**inputs, max_new_tokens=1024)
finally:
    timer_stop.set()
    timer_thread.join()

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]

output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)
