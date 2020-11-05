
# Training the model with temporal annotation
In this setting, we have a large number of normal data, and a few of abnormal data with temporal annotation, where we 
know which frame is normal or abnormal. 

We perform `k-fold` cross validation on `avenue` and `shanghaitech` dataset. 
In `avenue` dataset, we **re-annotate** the labels of each frames, and the re-annotated file is [avenue.json]("../data/annotations/avenue.json"). 
We set `k=10`, in avenue dataset and in `shanghaiTech` dataset, we set `k=5`. 
The following script is an example to train the model on avenue dataset with the `kth = 1` folder.
We change `kth` to other folders. 

```shell script
python train_scripts/train_temporal_annotation.py --dataset  avenue    \
         --prednet  cyclegan_convlstm    \
         --batch    2                    \
         --num_his  4                    \
         --label_level  temporal         \
         --k_folds  10                   \
         --kth      1                    \
         --gpu      0                    \
         --iters    80000   --output_dir  ./outputs
```

# Inference and evaluation.
After we train the model, we run the inference and evaluate all the checkpoints.
If there a more than 2 GPUs, you can immediately run the inference scripts after run the training scripts,
because the inference script is always listening the directory of the checkpoints, once there is a new
checkpoint, it will evaluate it immediately. Here we use `gpu 0` for training, and `gpu 1` for testing.

```shell script
python inference.py  --dataset  avenue    \
         --prednet  cyclegan_convlstm     \
         --num_his  4                     \
         --label_level  temporal          \
         --k_folds  10                    \
         --kth      1                     \
         --gpu      1                     \
         --interpolation   --output_dir  ./outputs
```