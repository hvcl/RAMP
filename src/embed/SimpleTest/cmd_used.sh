python simple_test_tf.py --dataset GDSC --embed ../../EmbeddingData/GDSC/ORI_p1.0q0.5w10d64
python simple_test_tf.py --dataset GDSC_MERGED_CCLE --embed ../../EmbeddingData/GDSC_EXTERNAL_CCLE/SOL_TRAINED_NW18000
python simple_test_tf.py --dataset GDSC_MERGED_TCGA --embed ../../EmbeddingData/GDSC_EXTERNAL_TCGA/SOL_TRAINED_NW12000
python simple_test_tf.py --dataset CCLE --extra --embed ../../EmbeddingData/CCLE_EXTRA_GDSC/ORI_f5_p1.0q0.5w10d64/
python simple_test_tf.py --patience 20 --hidden 256 --lr 0.0001 --dataset MERGED_CCLE --embed ../../EmbeddingData/MERGED_CCLE/ORI_NS_p0.8q0.5w10d64
python simple_test_tf.py --dataset MERGED --val_gdsc --embed ../../EmbeddingData/MERGED/ORI_NS_gval_p1.0q0.5w10d64
