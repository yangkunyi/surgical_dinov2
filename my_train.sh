# export http_proxy="http://localhost:20171"
# export https_proxy="http://localhost:20171"

/bd_byta6000i0/users/surgicaldinov2/miniforge3/condabin/conda run --live-stream -n DinoBloom python dinov2/train/train.py --config-file dinov2/configs/train/custom2.yaml --name seg_depth