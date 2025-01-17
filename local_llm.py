import transformers
import os
import time
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

class Model:
    def __init__(self, model_path, max_tokens=100):
        self.model_id = model_path
        self.pipeline = transformers.pipeline("text-generation", model=model_path, tokenizer=model_path, token=ACCESS_TOKEN)
        self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
        self.max_tokens = max_tokens

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def __call__(self, prompts, batch_size=1):
        prompts = prompts if isinstance(prompts, list) else [prompts]
        messages = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": p}
            ] for p in prompts
        ]
        
        results = self.pipeline(messages, max_new_tokens=self.max_tokens,return_full_text=False, batch_size=batch_size)

        return [r[0]["generated_text"] for r in results]


# model = Model("meta-llama/Llama-3.2-1B-Instruct")
# model.set_system_prompt("You are a chatbot assistant helping a user with their request. Keep it short and simple.")
#
# start = time.time()
# result = model(["Tell me a joke\n\n", "What is the capital of France?\n\n"])
#
# print(result)
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
