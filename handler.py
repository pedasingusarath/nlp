import json
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class AcademicQuestionHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The model files are packaged inside the .mar archive.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.initialized = True
        logger.info("Academic Question Handler initialized on %s", self.device)

    def preprocess(self, data):
        item = data[0]
        body = item.get("body") or item.get("data") or item
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)

        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        if len(prompt) > 2000:
            raise ValueError("prompt must be 2000 characters or fewer")

        max_new_tokens = int(body.get("max_new_tokens", 60))
        max_new_tokens = max(10, min(max_new_tokens, 120))
        temperature = float(body.get("temperature", 0.8))
        temperature = max(0.1, min(temperature, 1.5))
        full_prompt = (
            "Explain the following academic topic in simple terms. "
            "Give a clear, concise educational explanation.\n"
            f"Topic: {prompt}\nExplanation:"
        )
        return full_prompt, max_new_tokens, temperature

    def inference(self, inputs):
        prompt, max_new_tokens, temperature = inputs
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=400
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        generated = self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return generated

    def postprocess(self, inference_output):
        return [{"generated_text": inference_output}]

_service = AcademicQuestionHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if not data:
        return []
    inputs = _service.preprocess(data)
    result = _service.inference(inputs)
    return _service.postprocess(result)
