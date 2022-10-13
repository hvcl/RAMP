cd ../

if [ -z "$1" ]; then
    echo "Please give dataset name(=directory name) as the first argument."
    return 0
fi
dataset=$1

rans=1
if [ -z "$2" ]; then
    rans=0
fi

external=0
if [[ $dataset == *"_"* ]]; then
    external=${dataset#*_}
    dataset=${dataset%_*}
    echo "Train embedding vectors of $external with $dataset."
else
    echo "Train embedding vectors of $dataset."
fi

if [ $external = 0 ]; then
    if [ $rans = 0 ]; then
        python src/embed/main.py --dataset $dataset --data_path /Users/jarvis08/Documents/10_Organs/3_HYU/DeepLearningSystemLab/02_HPC/RAMP/data
    else
        python src/embed/main.py --dataset $dataset --data_path /Users/jarvis08/Documents/10_Organs/3_HYU/DeepLearningSystemLab/02_HPC/RAMP/data --rans
    fi
else
    if [ $rans = 0 ]; then
        python src/embed/main.py --dataset $dataset --data_path /Users/jarvis08/Documents/10_Organs/3_HYU/DeepLearningSystemLab/02_HPC/RAMP/data --fold 1 --external $external
    else
        python src/embed/main.py --dataset $dataset --data_path /Users/jarvis08/Documents/10_Organs/3_HYU/DeepLearningSystemLab/02_HPC/RAMP/data --rans --fold 1 --external  $external
    fi
fi
cd scripts