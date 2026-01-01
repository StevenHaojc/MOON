experiment_name="best_clip64_MLP_originInter_imgtxt_concat_size_descp"

python main/trainer3D_multi.py --model_use Inter --fusion_method concat \
--fold 1 --batch_size 16 --gpus 0 --num_epochs 100 --optimizer_use AdamW --num_classes 4 \
--data_dir ../Data --results_root ./results/Cls4/multiOrgan_${experiment_name}

python main/infer3D_multi.py --model_use Inter --fusion_method concat \
--dataset_type valid --fold 1 --model_path best_model_f1.pth --batch_size 16 \
--gpus 0 --results_root results/Cls4/multiOrgan_${experiment_name} \

python main/infer3D_multi.py --model_use Inter --fusion_method concat \
--dataset_type test --fold 1 --model_path best_model_f1.pth --batch_size 16 \
--gpus 0 --results_root results/Cls4/multiOrgan_${experiment_name} \

python main/postprocess/evaluate_one_fold.py \
--rootpath results/Cls4/multiOrgan_${experiment_name}/Inter/concat/fold1