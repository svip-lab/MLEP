
# Training the model with only normal data.
In this setting, we only have the normal videos.

```shell script
python train_scripts/train_normal_annotation.py --dataset  avenue    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  normal           \
         --gpu      0                    \
         --iters    80000  --output_dir  ./outputs

```

# Inference and evaluation.
After we train the model, we run the inference and evaluate all the checkpoints.
If there a more than 2 GPUs, you can immediately run the inference scripts after run the training scripts,
because the inference script is always listening the directory of the checkpoints, once there is a new
checkpoint, it will evaluate it immediately. Here we use `gpu 0` for training, and `gpu 1` for testing.

```shell script
python inference.py  --dataset  avenue    \
          --prednet  cyclegan_convlstm    \
          --num_his  4                     \
          --label_level  normal            \
          --gpu      1                     \
          --interpolation  --output_dir  ./outputs
```