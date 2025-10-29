from vllm import LLM

# llm_fp8 = LLM(model="<the exported model path>", quantization="modelopt")
llm_fp8 = LLM(model="/storage/lkk/Llama-3.1-8B-Instruct/")
print(llm_fp8.generate(["What's the age of the earth? "]))
