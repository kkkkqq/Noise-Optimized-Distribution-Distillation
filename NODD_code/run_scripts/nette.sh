



### The code for generating vanilla DiT samples
python sample_noise_extracted.py --model DiT-XL/2 --image-size 256 --ckpt PATH2DIT \
    --save-dir DIRECTORY4STORAGE --spec nette --nclass 10 --phase 0\
    --devices-per-model 1 --num-samples 50 \
    --seed 0 --stats-dir DIRECTORY4STATS \
    --no-opt

### The code for sampling template DiT statistics
python sample_stats_extracted.py --model DiT-XL/2 --image-size 256 --ckpt PATH2DIT \
    --save-dir DIRECTORY4STATS --spec nette --nclass 10 --phase 0\
    --devices-per-model 1 --num-samples TEMPLATESIZE\
    --seed 0 \
    --extractor-types S_16\
    --stats-types channel

### The code for generating NODD samples with DiT
### reg-strength is lambda_align
python sample_noise_extracted.py --model DiT-XL/2 --image-size 256 --ckpt PATH2DIT \
    --save-dir DIRECTORY4STORAGE --spec nette --nclass 10 --phase 0\
    --devices-per-model 1 --num-samples 50 \
    --seed 0 --reg-strength 0.001 --stats-dir DIRECTORY4STATS \
    --batch-size 2 --stats-type channel

### The code for sampling template stats with Minimax
### Note that flatten means flattening the 4x32x32 denoised sample into a 4096x1x1 tensor
python sample_stats_extracted.py --model DiT-XL/2 --image-size 256 --ckpt PATH2MINIMAX \
    --save-dir DIRECTORY4STATS --spec nette --nclass 10 --phase 0\
    --devices-per-model 1 --num-samples TEMPLATESIZE\
    --seed 0 \
    --extractor-types flatten\
    --stats-types channel

### The code for generating NODD samples with Minimax DiT
### this is equivalent to setting --extracted-weight to 0 (lambda_F=0) and --x-weight to 1 (lambda_Z=1), but it runs faster as it doesn't go over the 0-weighted extracted distribution alignment
python sample_noise_extracted.py --model DiT-XL/2 --image-size 256 --ckpt PATH2MINIMAX \
    --save-dir DIRECTORY4STORAGE --spec nette --nclass 10 --phase 0\
    --devices-per-model 1 --num-samples 50 \
    --seed 0 --reg-strength 0.01 --stats-dir DIRECTORY4STATS \
    --batch-size 2 --extractor-type flatten --stats-type channel

