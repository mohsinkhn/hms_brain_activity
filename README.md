# hms_brain_activity

# python -m src.convert_parquet_to_npy
# nohup python train.py -m experiment=clean_effb1 model.net.use_bnorm=true model.net.bnorm=instance paths.root_dir=/home/mohsin/testing/hms_benchmark data.batch_size=64 model.optimizer.lr=0.002 +model.freeze_bnorm=true comment='instance norm' data.fold_id=0 seed=786 &
# nohup python train.py -m experiment=clean_effb1 model.net.use_bnorm=false paths.root_dir=/home/mohsin/testing/hms_benchmark data.batch_size=64 model.optimizer.lr=0.002 data.fold_id=1,2,3,4 +model.freeze_bnorm=true comment='instance norm' &

# python -m src.merge_preds



# Pseudo labels
# nohup python train.py -m experiment=clean_effb1_pseudo model.net.use_bnorm=false paths.root_dir=/home/mohsin/testing/hms_benchmark data.fold_id=0,1,2,3,4 comment='take labels for less than 20' &
