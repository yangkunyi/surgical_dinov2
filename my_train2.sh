export http_proxy="http://localhost:20171"
export https_proxy="http://localhost:20171"

/bd_byta6000i0/users/surgical_depth/miniforge3/condabin/conda run --live-stream -n dino_train python dinov2/train/train_flat_lr.py --config-file dinov2/configs/train/custom3.yaml --name lr=1e-3_batchsize=32_flat