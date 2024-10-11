python install.py --neurips23track sparse --algorithm nmslib
# python run.py --neurips23track sparse    --algorithm nmslib1 --dataset sparse-small
python run.py --neurips23track sparse    --algorithm nmslib_efc256 --dataset sparse-1M
python run.py --neurips23track sparse    --algorithm nmslib_efc256 --dataset sparse-full
python run.py --neurips23track sparse    --algorithm nmslib_efc512 --dataset sparse-full
python run.py --neurips23track sparse    --algorithm nmslib_efc1024 --dataset sparse-full