Training:
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_mono_hk_288.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_add_mono_hk_288.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_rem_mono_hk_288.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_addrem_mono_hk_288.json

Testing:
median scaling:
CUDA_VISIBLE_DEVICES=2 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist --eval --seq all --save_triplet > IID_hk_288.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_add_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist --eval --seq all --save_triplet > IID_add_hk_288.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_rem_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist --eval --seq all --save_triplet > IID_rem_hk_288.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_addrem_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist --eval --seq all --save_triplet > IID_addrem_hk_288.log 2>&1 &

single scale = median of sca
les from logs of all methods:
grep "Scaling ratios | med:" IID*.log | awk -F 'med: ' '{print $2}' | awk '{print $1}' | sort -n | awk 'BEGIN {c=0; sum=0;} {a[c++]=$1; sum+=$1;} END {if (c%2==1) print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2;}'
79406.8 ~ 79407

CUDA_VISIBLE_DEVICES=1 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist/singlescale --disable_median_scaling --pred_depth_scale_factor 79407 --eval --seq all --save_triplet > singlescale_IID_hk_288.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_add_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist/singlescale --disable_median_scaling --pred_depth_scale_factor 79407 --eval --seq all --save_triplet > singlescale_IID_add_hk_288.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_rem_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist/singlescale --disable_median_scaling --pred_depth_scale_factor 79407 --eval --seq all --save_triplet > singlescale_IID_rem_hk_288.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name finetuned_addrem_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist/singlescale --disable_median_scaling --pred_depth_scale_factor 79407 --eval --seq all --save_triplet > singlescale_IID_addrem_hk_288.log 2>&1 &


# test on hyper kvasir
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1 --save_depth  --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1inpainted --save_depth  --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_add_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1 --save_depth  --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_rem_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1 --save_depth  --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_addrem_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1 --save_depth  --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png


CUDA_VISIBLE_DEVICES=3 nohup python test_simple.py --image_path /media/rema/data/C3VD/Undistorted/Dataset --model_name IID_depth_model --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist --eval --seq all --save_triplet > IID.log 2>&1 &
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name IID_depth_model --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist --seq Test1 --save_depth
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name IID_depth_model --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1 --save_depth --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png

generating albedo:
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1 --save_depth --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png --decompose
python test_simple.py --image_path /media/rema/data/DataHKGab/Undistorted --model_name finetuned_mono_hk_288/models/weights_19 --method IID --model_basepath /media/rema/trained_models --output_path /media/rema/outputs/undisttrain/undist_masked --seq Test1inpainted --save_depth --input_mask /media/rema/data/DataHKGab/Undistorted/mask_hk_288.png --decompose

training with different parameters and epoch number = 30
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_mono_hk_288_lr5.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_mono_hk_288_ds3.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_mono_hk_288_alb05.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_mono_hk_288_rep5.json
CUDA_VISIBLE_DEVICES=1 python train.py --config  /media/rema/configs/IID/finetuned_mono_hk_288_rec05.json