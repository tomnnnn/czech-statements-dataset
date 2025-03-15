import transformers
from transformers import BitsAndBytesConfig
import os
from dotenv import load_dotenv
import time
from accelerate import infer_auto_device_map
from together import Together
from openai import OpenAI

load_dotenv()

class LanguageModelAPI:
    def __init__(self, model_path, max_tokens=1000):
        # Additional settings
        self.max_tokens = max_tokens
        self.model_path = model_path

    def set_system_prompt(self, prompt):
        """Set system prompt, which will be included in every conversation"""
        self.system_prompt = prompt

    def set_examples(self, examples):
        """Set examples for few-shot prompts"""
        self.examples = examples

    def set_generation_prompt(self, prompt):
        """Set generation prompt, which will be used to generate completions"""
        self.generation_prompt = prompt

    def prepare_input(self, prompts):
        """
        Puts prompts and examples into list of chat format dictionaries
        """
        prompts = prompts if isinstance(prompts, list) else [prompts]

        conversations = [
            [ {"role": "system", "content": self.system_prompt}, ]
            +
            [ {"role": role, "content": example[key]} for example in self.examples for role,key in [("user", "input"), ("assistant", "output")] ]
            +
            [ {"role": "user", "content": prompt} ]
            for prompt in prompts
        ]

        if self.generation_prompt:
            conversations = [
                chat + [{"role": "assistant", "content": self.generation_prompt}] for chat in conversations
            ]

        return conversations

    def _infer(self, conversations, batch_size=8):
        raise NotImplementedError

    def __call__(self, prompts, batch_size=8)->list:
        conversations = self.prepare_input(prompts)
        return self._infer(conversations, batch_size)


class Transformer_Local(LanguageModelAPI):
    def __init__(self, model_path, max_tokens=1000, quant_config=None):
        super().__init__(model_path, max_tokens)

        # Load model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, token=HF_ACCESS_TOKEN, device_map="auto", torch_dtype="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=HF_ACCESS_TOKEN, padding_side="left")

        # Set padding token to eos token if it is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, token=os.getenv("HF_ACCESS_TOKEN"))

    def _infer(self, conversations, batch_size=8):
        results = self.pipeline(conversations, max_new_tokens=self.max_tokens,return_full_text=False, batch_size=batch_size, truncation=True, do_sample=False)
        return [result[0]["generated_text"] for result in results]
