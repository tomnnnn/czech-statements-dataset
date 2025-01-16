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
        sequences = [f"{self.system_prompt}\n{p}" for p in prompts]

        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True)
        print(inputs)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=10,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


print(ACCESS_TOKEN)
model = Model("meta-llama/Llama-3.2-1B")
model.set_system_prompt("")
result = model(["Capital of France is", "The capital of Czech Republic is"])

for r in result:
    print(r)
