import os
import json

import scipy.io
import networkx as nx
from data_path import *


def extract_cellline_testset(CELL_FILE, CELL_INDEX_FILE, valid_idx):
    cellline_dict = dict()
    index = 0

    with open(CELL_FILE, mode='r') as file:
        for cell in file:
            cellline_dict[index] = str(cell.split('\n')[0])
            index += 1

    cell_idx_file = scipy.io.loadmat(CELL_INDEX_FILE)
    cell_idx_list = cell_idx_file['idx'][0]

    divider = len(cellline_dict) // 10
    if valid_idx == 9:
        cell_testset_list = [cellline_dict[cell] for cell in cell_idx_list[divider * valid_idx: divider * (valid_idx + 1) + 1]]
    else:
        cell_testset_list = [cellline_dict[cell] for cell in cell_idx_list[divider * valid_idx: divider * (valid_idx + 1)]]

    return cell_testset_list


def extract_cellline_embedding(SAVE_FILES, folds_trained, cell_embed_mode, ext_str):
    def write_cellline_embedding(read_file, write_file, cellline_list):
        file_reader = open(read_file, 'r')
        file_extract = open(write_file, cell_embed_mode)

        extracted_lines_count = 0
        extracted_dimensions = file_reader.readline().split()[1]
        list_extract_tmp = []
        while True:
            line = file_reader.readline()
            if not line:
                break

            if line is not '':
                line_key = line.split()[0]
                if line_key in cellline_list:
                    list_extract_tmp.append(line)
                    extracted_lines_count += 1

        file_extract.write(str(extracted_lines_count) + ' ' + str(extracted_dimensions) + '\n')
        file_extract.writelines(list_extract_tmp)

        file_reader.close()
        file_extract.close()

    cellline_list = list()
    with open(SAVE_FILES['CELL_TO_EXTRACT'], mode='r') as file:
        for cell in file:
            cell = cell.split('\n')[0]
            cellline_list.append(cell)

    for idx in range(folds_trained[0], folds_trained[1] + 1):
        total_embedding_file = SAVE_FILES['EMBEDDING_FILE'].format(str(idx) + ext_str)
        cellline_embedding_file = SAVE_FILES['CELL_EMBEDDING_FILE'].format(str(idx) + ext_str)
        write_cellline_embedding(total_embedding_file, cellline_embedding_file, cellline_list)


def set_path(args):
    tag = get_save_dir_name(args)
    files = PATH.copy()

    base = args.data_path
    prefix = os.path.join(base, args.dataset)
    for info, f_name in files.items():
        files[info] = os.path.join(prefix, f_name)

    if args.init_gdsc or args.fix_gdsc:
        files['GDSC_EMBEDDING_FILE'] = args.embed_path

    if args.curated:
        files['PROTEIN_PROTEIN_FILE'] = os.path.join(base, PROTEIN['CURATED'])
    else:
        files['PROTEIN_PROTEIN_FILE'] = os.path.join(base, PROTEIN['MERGED'])


    if args.extra:
        extra_files = PATH.copy()
        prefix = os.path.join(base, args.extra)
        for info, f_name in extra_files.items():
            extra_files[info] = os.path.join(prefix, f_name)
        files['extra'] = extra_files

        save_dir = f'../results/embedding/{args.dataset}_EXTRA_{args.extra}/{tag}'
    elif args.external:
        save_dir = f'../results/embedding/GDSC_EXTERNAL_{args.external}/{tag}'
        args.fold = 1

        files['MERGED_CELL_PROTEIN_FILE'] = os.path.join(base, MERGE['CELL_PROTEIN_FILE'].format(args.external))
        if args.val_gdsc:
            files['CELL_DRUG_FILE'] = os.path.join(base, MERGE['CELL_DRUG_VAL_FILE'].format('GDSC'))
        else:
            files['CELL_DRUG_FILE'] = os.path.join(base, MERGE['CELL_DRUG_ALL_FILE'].format('GDSC'))
    else:
        save_dir = f'../results/embedding/{args.dataset}/{tag}'

    save_files = SAVE.copy()
    for info, f_name in save_files.items():
        save_files[info] = os.path.join(save_dir, f_name)
    print(f'Save directory: {save_dir}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.walkers:
        save_files['CELL_TO_EXTRACT'] = args.walkers
    elif args.extra:
        save_files['CELL_TO_EXTRACT'] = os.path.join(base, MERGE['CELL_FILE'].format(args.dataset))
    elif args.external:
        save_files['CELL_TO_EXTRACT'] = os.path.join(base, MERGE['CELL_FILE'].format(args.external))
    else:
        save_files['CELL_TO_EXTRACT'] = files['CELL_FILE']

    files["RESISTANT_FILE"] = os.path.join(base, RESISTANT[args.dataset])

    with open(os.path.join(save_dir, "hparams.json"), 'w') as f:
        tmp = vars(args)
        json.dump(tmp, f, indent=4)
    return files, save_files


def get_save_dir_name(args):
    if args.walkers:
        return f'IL_NW{args.num_walks}'

    if args.loocv:
        tag = f'LOOCV'
    elif args.init_gdsc:
        tag = 'INIT'
    elif args.fix_gdsc:
        tag = 'GFIX'
    elif args.fix_ext:
        tag = 'FIX'
    else:
        tag = 'ORI'

    if args.rans:
        tag += '_NS'

    if args.fix_w:
        tag += '_FW'

    if args.external and args.val_gdsc:
        tag += '_gval'

    if args.curated:
        tag += '_curated'

    if args.fold != 10:
        tag += f'_f{args.fold}'
    return tag + f'_p{args.p}q{args.q}w{args.window}d{args.dim}'

