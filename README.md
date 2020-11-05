## 1. Installation (Anaconda with python3.6 installation is recommended)

## 2. Download datasets
Please manually download all datasets from [avenue.tar.gz and shanghaitech.tar.gz](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F)
and tar each tar.gz file, and move them in to **data** folder.

You can also download data from BaiduYun(https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ)  i9b3 

## 3. Inference the pretrain model

#### 3.1 Inference with Only-Normal-Data Pretrained model
```
python inference.py  --dataset  avenue    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      0                     \
          --interpolation  --snapshot_dir  ./data/pretrains/avenue/normal/checkpoints/model.ckpt-74000
```

#### 3.2 Inference with Video-Annotated Pretrained model
```
python inference.py  --dataset  avenue    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  tune_video            \
          --gpu      0                     \
          --interpolation  --snapshot_dir  ./data/pretrains/avenue/tune_video/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0/model.ckpt-76000
```


#### 3.3 Inference with Temporal-Annotated Pretrained model
```
python inference.py  --dataset  avenue    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      0                     \
          --interpolation  --snapshot_dir  ./data/pretrains/avenue/temporal/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0/model.ckpt-77000
```

## 4. Training model with different settings from scratch
See more details in 

4.1 [only_normal_data](./doc/only_normal_data.md);

4.2 [video_annotation](./doc/video_annotation.md);

4.3 [temporal_annotation](./doc/temporal_annotation.md).


