from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    WhisperProcessor,
)


fp8_model = AutoModelForCausalLM.from_pretrained("./llama3.1-8b-instruct-nvfp4")
print(fp8_model)
