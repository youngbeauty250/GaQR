python train.py \
    --model_name_or_path /Llama-2-7b-chat-hf \
    --data_path /data/hotpotqa_train_data_0103 \
    --bf16 True \
    --output_dir s2_output-Llama-2-7b-chat-hf \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --tf32 False
    # --save_steps 1500 \
