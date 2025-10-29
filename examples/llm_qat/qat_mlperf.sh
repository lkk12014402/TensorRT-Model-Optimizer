
export CUDA_VISIBLE_DEVICES=5

./launch.sh --model /storage/lkk/Llama-3.1-8B-Instruct/  \
   --num_epochs 2.0 \
   --lr 1e-5 \
   --do_train True \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --output_dir llama3-qat
