import transformers
from transformers import BitsAndBytesConfig
import os
from dotenv import load_dotenv
import time
from accelerate import infer_auto_device_map

load_dotenv()
HF_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

class Model:
    examples = []
    system_prompt = ""

    def __init__(self, model_path, max_tokens=1000):
        self.model_id = model_path

        # bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model and tokenizer
        # self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, token=HF_ACCESS_TOKEN, device_map="auto", torch_dtype="auto")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, token=HF_ACCESS_TOKEN, device_map="auto", torch_dtype="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=HF_ACCESS_TOKEN, padding_side="left")

        # Set padding token to eos token if it is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, token=HF_ACCESS_TOKEN)

        # Additional settings
        self.max_tokens = max_tokens

    def set_system_prompt(self, prompt):
        """Set system prompt, which will be included in every conversation"""
        self.system_prompt = prompt

    def set_examples(self, examples):
        """Set examples for few-shot prompts"""
        self.examples = examples

    def set_generation_prompt(self, prompt):
        """Set generation prompt, which will be used to generate completions"""
        self.generation_prompt = prompt

    def __call__(self, prompts, batch_size=8):
        prompts = prompts if isinstance(prompts, list) else [prompts]

        conversations = [
            [
                {"role": "system", "content": self.system_prompt},
            ]
            +
            [
                {"role": role, "content": example[key]}
                for example in self.examples
                for role,key in [("user", "input"), ("assistant", "output")]
            ]
            +
            [
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]

        conversations = [
            chat + [{"role": "assistant", "content": self.generation_prompt}] for chat in conversations
        ]

        results = self.pipeline(conversations, max_new_tokens=self.max_tokens,return_full_text=False, batch_size=batch_size, truncation=True, do_sample=False)

        return [result[0]["generated_text"] for result in results]


if __name__ == "__main__":
    """Example usage"""
    model = Model("meta-llama/Llama-3.2-1B-Instruct")
    model.set_system_prompt("You are a fact-checker. Your task is to provide accurate prediction of whether a statement is true or false.")
    model.set_examples([
        {
            "input": "Barack Obama, born in Texas, has been a hope for a better future",
            "output": "Let me think. I need to identify parts that can be factually verified. 1) Was Barack Obama born in Texas? Conclusion: Upon inspecting, I found 1 passage that can be factually checked. Barack Obama was born in Hawaii, therefore this statement is false. Whether he has been a hope for a better future cannot be factually checked."
        }
    ])
    model.set_generation_prompt("Let me think. I need to identify parts that can be factually verified.")

    start = time.time()
    result = model("(…) porovnat si, kolik jsme platili za nájmy, potraviny, léky, dopravu, energie v roce 2021 a kolik platíme nyní. (…) zdražilo o třetinu úplně všechno.")
    print(result[0].replace("\\n", "\n"))

    end = time.time()
    print(f"Time taken: {end - start} seconds")
