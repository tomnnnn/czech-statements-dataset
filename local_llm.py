import transformers
import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


class Model:
    def __init__(self, model_path):
        self.model_id = model_path

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, token=ACCESS_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=ACCESS_TOKEN,
        )

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def __call__(self, prompts):
        prompts = prompts if isinstance(prompts, list) else [prompts]
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + [
            {"role": "user", "content": p+'\n'} for p in prompts
        ]

        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(chat)

        inputs = self.tokenizer(chat, return_tensors="pt", padding=True)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


model = Model("meta-llama/Llama-3.2-1B-Instruct")
model.set_system_prompt("You are a chatbot assistant helping a user with their request. Keep it short and simple.")
result = model("Tell me a joke\n\n")

for r in result:
    print(r)
