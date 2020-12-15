# Segmentation renormalized harmonization 

An anatomically-regularized un-paired image-to-image translation framework built on CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Environment
```shell script
conda create --name srn python=3.7
pip install -r requirements.txt
```

## Training
```shell script
python mains/train.py --name 'seg_renorm_cyclegan'\
      --model 'cycle_gan_2d'\
      --checkpoints_dir '../ckpts' \
      --crop_size 128 \
      --batch_size 4 \
      --lr_g 0.0002 \
      --lr_d 0.0001 \
      --gpu_id 0\
      --ngf 64 \
      --ndf 64 \
      --typeG 'resunet'\
      --netD 'n_layers'\
      --n_layers_D 2 \
      --dim 2\
      --save_epoch_freq 100\
      --save_latest_freq 2  \
      --input_nc 1\
      --output_nc 1\
      --thickness 1\
      --dataset 'ixi'\
      --lambda_identity 0\
      --lambda_cc 0\
      --lambda_tv 0\
      --lambda_A 10\
      --lambda_B 10\
      --gan_mode 'lsgan'\
      --init_type 'normal'\
      --seg_nc 4\
      --niter 20\
      --niter_decay 300 \
      --joint_seg\
      --spade\
```
