# Possible commands list
python main.py --rans --worker 3 --merge --init_gdsc --val_gdsc --sp
python main.py --rans --worker 3 --merge --val_gdsc --sp
python main.py --worker 3 --rans --merge --init_gdsc --p 1.0 --q 0.3

python main.py --dataset CCLE --loocv --worker 1 --fold 1 --p 0.8 --q 0.3

python main.py --worker 1 --fix_gdsc --external CCLE --p 1.2 --fix_w 3 --rans

python main.py --worker 1 --external CCLE --p 0.8 --q 0.3 --fix_w 3 --rans
python main.py --worker 1 --external CCLE --p 0.8 --q 0.3 --fix_w 3
python main.py --worker 1 --external CCLE --p 0.8 --q 0.3 --rans
python main.py --worker 1 --external CCLE --p 0.8 --q 0.3
python main.py --rans --worker 4 --p 1.0 --q 0.7 --dim 64 --curated
