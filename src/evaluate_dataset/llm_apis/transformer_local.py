from llm_api import LanguageModelAPI
import os
import transformers

class Transformer_Local(LanguageModelAPI):
    def __init__(self, model_path, max_tokens=1000, quant_config=None):
        super().__init__(model_path, max_tokens)

        hf_token = os.getenv("HF_ACCESS_TOKEN")

        # Load model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, token=hf_token, device_map="auto", torch_dtype="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=hf_token, padding_side="left")

        # Set padding token to eos token if it is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, token=hf_token)

    def _infer(self, conversations, batch_size=8):
        results = self.pipeline(conversations, max_new_tokens=self.max_tokens,return_full_text=False, batch_size=batch_size, truncation=True, do_sample=False)
        return [result[0]["generated_text"] for result in results]
