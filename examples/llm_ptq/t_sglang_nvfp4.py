import sglang as sgl

llm_fp8 = sgl.Engine(model_path="./llama3.1-8b-instruct-nvfp4", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
