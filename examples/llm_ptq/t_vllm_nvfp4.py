from vllm import LLM

# llm_fp8 = LLM(model="<the exported model path>", quantization="modelopt")
llm_fp8 = LLM(model="./llama3.1-8b-instruct-nvfp4", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
