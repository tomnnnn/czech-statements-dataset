import os
import logging

logger = logging.getLogger(__name__)

class LanguageModelAPI:
    generation_prompt = ""
    system_prompt = ""
    examples = []

    def __init__(self, model_path, **kwargs):
        if os.getenv("HF_TOKEN", None) is None:
            logger.warning("Hugging Face token not found in environment variables.")

        self.model_path = model_path
        self.chat_format = not kwargs.get("no_chat_format", False)

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

        if self.chat_format:
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
        else:
            examples = "\r\n\r\n".join([f"Výrok: {example['input']}\r\n{example['output']}" for example in self.examples])
            conversations = [f"{self.system_prompt}\r\n\r\n{examples}\r\n\r\nVýrok: {prompt}\r\n{self.generation_prompt or 'Hodnocení'}" for prompt in prompts]

        return conversations

    async def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        raise NotImplementedError

    def _infer_sync(self, conversations, batch_size=8, max_new_tokens=1000):
        raise NotImplementedError

    async def __call__(self, prompts, batch_size=8, max_new_tokens=1000) -> list:
        conversations = self.prepare_input(prompts)
        return await self._infer(conversations, batch_size, max_new_tokens)

    def generate_sync(self, prompts, batch_size=8, max_new_tokens=1000) -> list:
        """
        Generate predictions synchronously.
        """
        conversations = self.prepare_input(prompts)
        return self._infer_sync(conversations, batch_size, max_new_tokens)
