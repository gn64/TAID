python train.py \
    --student_model TinyLlama/TinyLlama_v1.1 \
    --teacher_model meta-llama/Llama-2-7b-chat-hf \
    --data_path data/llama-2 \
    --output_dir logs/llama-2-distillm \
    --batch_size 4 \
    --num_epochs 5 \
    --loss_type srkl \
    --skew_beta 0.1 \
    --sampling_type adaptive \
    --val_check_interval 0.5 \
    --validate_first \
    --accumulate_grad_batches 2