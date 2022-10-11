import os
import sys
import argparse

pwd = os.getcwd()
os.chdir(pwd)
sys.path.append(pwd)

from Embedding.EmbeddingGenerator import generate_embedding
from Embedding.OperateFunctions import extract_cellline_embedding, set_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[RAMP] Train Network Embedding')
    parser.add_argument('--data_path', default='', required=True, type=str, help="Absolute path of datasets")
    parser.add_argument('--embed_path', default='', type=str, help="Absolute path of pretrained embedding")
    parser.add_argument('--dataset', default='GDSC', type=str, help="Dataset to use between GDSC and TCGA")
    parser.add_argument('--curated', action='store_true', 
            help='Use curated Protein-Protein edges only, else include predicted edges.')
    parser.add_argument('--rans', action='store_true', help='Use Response-aware Negative Sampling.')
    parser.add_argument('--val_gdsc', action='store_true', 
            help='Use GDSC in training & validation, and use CCLE in testing')
    parser.add_argument('--loocv', action='store_true', help='Train without Cell-Drug edges for LOOCV')
    parser.add_argument('--init_gdsc', action='store_true', help='Initialize GDSC cells and train CCLE in GC.')
    parser.add_argument('--fix_gdsc', action='store_true', help='Fix GDSC cells and train CCLE in GC.')
    parser.add_argument('--fix_ext', action='store_true', help='Use Cell line Embedding Initialization.')
    parser.add_argument('--no_pp', action='store_true', help='No Protein-Protein edges in external validation')
    parser.add_argument('--extra', default='', type=str, help="In training, use extra dataset also")
    parser.add_argument('--external', default='', type=str,
            help="External Validation. Training: GDSC, Testing: CCLE or TCGA")
    parser.add_argument('--incremental', default='', type=str,
            help="Based on GDSC, train external list of cells, adding Cell-Protein edges only.")
    parser.add_argument('--fix_w', default=0, type=float, help="Fix Cell-Protein edge's weight values")
    parser.add_argument('--fold_start', default=1, type=int, help="Fold # to start")
    parser.add_argument('--fold_stop', default=0, type=int, help="Fold # to stop")
    parser.add_argument('--fold', default=10, type=int, help="Number of folds for Cross Validation")
    parser.add_argument('--dim', default=64, type=int, help="H-param of Node2Vec")
    parser.add_argument('--p', default=1.0, type=float, help="H-param of Node2Vec")
    parser.add_argument('--q', default=0.5, type=float, help="H-param of Node2Vec")
    parser.add_argument('--window', default=10, type=int, help="H-param of Node2Vec")
    parser.add_argument('--num_walks', default=100, type=int, help="H-param of Node2Vec")
    parser.add_argument('--len_walk', default=50, type=int, help="H-param of Node2Vec")
    parser.add_argument('--workers', default=10, type=int, help="H-param of Node2Vec")
    parser.add_argument('--walkers', default='', type=str, help='Designate nodes to walk')
    parser.add_argument('--only_cell', action='store_true', help='Use Cell lines and Drugs in walks')
    parser.add_argument('--fname_ext', default='', type=str, 
            help='Add extra string to filename of total-embedding')
    args = parser.parse_args()

    args.fold_stop = args.fold_stop if args.fold_stop else args.fold

    dataset_files, save_files = set_path(args)
    generate_embedding(args, dataset_files, save_files['EMBEDDING_FILE'])
    extract_cellline_embedding(
            save_files,
            (args.fold_start, args.fold_stop),
            'a' if args.walkers else 'w',
            args.fname_ext
    )

