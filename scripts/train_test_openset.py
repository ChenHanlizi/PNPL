import os

NEGA_CTX = 2
LOG = 1
distance_weight = 1
negative_weight = 1
nega_nega_weight = 0.05
open_set_method = 'MSP'
open_score = 'OE'
clip_backbone = 'ViT-B/16'
batch_size = 64
dataset  = "UCM_p2"
few_shot = 0

# train stage1
os.system('CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
python osr_nega_prompt.py --dataset {dataset} --batch_size {batch_size} --NEGA_CTX {NEGA_CTX} --LOG {LOG}\
 --distance_weight {distance_weight}  --negative_weight {negative_weight} --open_score {open_score}\
 --clip_backbone {clip_backbone} --nega_nega_weight {nega_nega_weight} --stage {stage} \
 --few_shot {few_shot}'.format(
    dataset = dataset, batch_size = batch_size, NEGA_CTX = NEGA_CTX, distance_weight=distance_weight,
    negative_weight = negative_weight,  open_score = open_score, clip_backbone=clip_backbone,
    nega_nega_weight = nega_nega_weight, LOG = LOG, stage = 1, few_shot = 0))
# train stage2
os.system('CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
python osr_nega_prompt.py --dataset {dataset} --batch_size {batch_size} --NEGA_CTX {NEGA_CTX} --LOG {LOG}\
 --distance_weight {distance_weight}  --negative_weight {negative_weight} --open_score {open_score}\
 --clip_backbone {clip_backbone} --nega_nega_weight {nega_nega_weight} --stage {stage} \
 --few_shot {few_shot}'.format(
    dataset = dataset, batch_size = batch_size, NEGA_CTX = NEGA_CTX, distance_weight=distance_weight,
    negative_weight = negative_weight,  open_score = open_score, clip_backbone=clip_backbone,
    nega_nega_weight = nega_nega_weight, LOG = LOG, stage = 3, few_shot = 0))
# test
os.system('CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
python osr_nega_prompt.py --dataset {dataset} --batch_size {batch_size} --NEGA_CTX {NEGA_CTX} --LOG {LOG}\
 --distance_weight {distance_weight}  --negative_weight {negative_weight} --open_score {open_score}\
 --clip_backbone {clip_backbone} --nega_nega_weight {nega_nega_weight} --stage {stage} \
 --few_shot {few_shot}'.format(
    dataset = dataset, batch_size = batch_size, NEGA_CTX = NEGA_CTX, distance_weight=distance_weight,
    negative_weight = negative_weight,  open_score = open_score, clip_backbone=clip_backbone,
    nega_nega_weight = nega_nega_weight, LOG = LOG, stage = 6, few_shot = 0))