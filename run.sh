python install.py --neurips23track sparse --algorithm linscan
python run.py --neurips23track sparse    --algorithm linscan --dataset sparse-small
python run.py --neurips23track sparse    --algorithm linscan --dataset sparse-1M
python run.py --neurips23track sparse    --algorithm linscan --dataset sparse-full

python install.py --neurips23track sparse --algorithm nmslib
python run.py --neurips23track sparse    --algorithm nmslib --dataset sparse-small
python run.py --neurips23track sparse    --algorithm nmslib --dataset sparse-1M
python run.py --neurips23track sparse    --algorithm nmslib --dataset sparse-full

python install.py --neurips23track sparse --algorithm shnsw
python run.py --neurips23track sparse    --algorithm shnsw --dataset sparse-small
python run.py --neurips23track sparse    --algorithm shnsw --dataset sparse-1M
python run.py --neurips23track sparse    --algorithm shnsw --dataset sparse-full

python install.py --neurips23track sparse --algorithm pyanns
python run.py --neurips23track sparse    --algorithm pyanns --dataset sparse-small
python run.py --neurips23track sparse    --algorithm pyanns --dataset sparse-1M
python run.py --neurips23track sparse    --algorithm pyanns --dataset sparse-full