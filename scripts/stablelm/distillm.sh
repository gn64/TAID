python train.py \
    --student_model EleutherAI/pythia-410m \
    --teacher_model stabilityai/stablelm-zephyr-3b \
    --data_path data/stablelm \
    --output_dir logs/stablelm-distillm \
    --batch_size 8 \
    --num_epochs 5 \
    --loss_type sfkl \
    --skew_beta 0.1 \
    --sampling_type adaptive \
    --val_check_interval 0.5 \
    --validate_first 