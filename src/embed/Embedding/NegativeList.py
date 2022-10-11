from Embedding.OperateFunctions import extract_cellline_testset
import scipy.io
import numpy as np
import random


def extract_gdsc_negative_list(CELL_FILE, DRUG_FILE, CELL_DRUG_RESPONSE_FILE, test_cells):
    negative_list = dict()

    cell_idx_dict = dict()
    drug_idx_dict = dict()

    # indexing cell_line & drug
    idx = 0
    with open(CELL_FILE, 'r') as c_file:
        for cell in c_file:
            cell_idx_dict[idx] = cell.split('\n')[0]
            idx += 1

    idx = 0
    with open(DRUG_FILE, 'r') as d_file:
        for drug in d_file:
            drug_idx_dict[idx] = drug.split('\n')[0]
            idx += 1

    # fill negative data
    cell_drug_res_file = scipy.io.loadmat(CELL_DRUG_RESPONSE_FILE)
    cell_drug_res_list = cell_drug_res_file['data']

    for idx, cell_drug in enumerate(cell_drug_res_list):
        cell = cell_idx_dict[idx]
        if cell in test_cells:
            continue

        resistant_drug_idx = np.where(cell_drug == 0)[0]
        negative_list[cell] = [drug_idx_dict[drug_idx] for drug_idx in resistant_drug_idx]

    drug_cell_res_list = np.transpose(cell_drug_res_list)
    for idx, drug_cell in enumerate(drug_cell_res_list):
        drug = drug_idx_dict[idx]

        resistant_cell_idx = np.where(drug_cell == 0)[0]
        negative_list[drug] = [cell_idx_dict[cell_idx] for cell_idx in resistant_cell_idx]
        negative_list[drug] = np.setdiff1d(negative_list[drug], test_cells)

    for idx, data in enumerate(negative_list):
        random.shuffle(negative_list[data])

    return negative_list


#def extract_negative_list(path):
#    from_cell = dict()
#    from_drug = dict()
#    with open(path, 'r') as f:
#        cell_drug = f.readlines()
#        for idx in range(0, len(cell_drug), 2):
#            info = cell_drug[idx].split('\t')
#            cell, drug = info[0], info[1]
#            if from_cell.get(cell) is None:
#                from_cell[cell] = []
#            if from_drug.get(drug) is None:
#                from_drug[drug] = []
#            from_cell[cell].append(drug)
#            from_drug[drug].append(cell)
#
#    negative_list = dict()
#
#    all_drugs = list(from_drug.keys())
#    for cell, drugs in from_cell.items():
#        drugs_connected_to_cell = np.array(drugs)
#        negative_drugs = np.setdiff1d(all_drugs, drugs_connected_to_cell)
#        if len(negative_drugs) < 10:
#            continue
#        negative_list[cell] = negative_drugs
#
#    all_cells = list(from_cell.keys())
#    for drug, cells in from_drug.items():
#        cells_connected_to_drug = np.array(cells)
#        negative_list[drug] = np.setdiff1d(all_cells, cells_connected_to_drug)
#    
#    for _, data in enumerate(negative_list):
#        random.shuffle(negative_list[data])
#    return negative_list


def extract_negative_list(cell_drug_path, resistant_path):
    training_cells = set()
    with open(cell_drug_path, 'r') as f:
        cell_drug = f.readlines()
        for idx in range(0, len(cell_drug), 2):
            cell = cell_drug[idx].split('\t')[0]
            training_cells.add(cell)

    from_cell = dict()
    from_drug = dict()
    with open(resistant_path, 'r') as f:
        resistants = f.readlines()
        for idx in range(len(resistants)):
            info = resistants[idx].split('\t')
            cell, drug = info[0], info[1]
            if cell not in training_cells:
                continue

            if from_cell.get(cell) is None:
                from_cell[cell] = []
            if from_drug.get(drug) is None:
                from_drug[drug] = []
            from_cell[cell].append(drug)
            from_drug[drug].append(cell)

    negative_list = dict()
    negative_list.update(from_cell)
    negative_list.update(from_drug)
    
    for data in negative_list.keys():
        random.shuffle(negative_list[data])
    return negative_list


def generate_negative_list(args, files, k):
    if not args.rans:
        return None

    if args.dataset == 'GDSC':
        if args.external and not args.val_gdsc:
            test_cells = []
        else:
            test_cells = extract_cellline_testset(files['CELL_FILE'], files['CELL_INDEX_FILE'], k - 1)
        #negative_list = extract_gdsc_negative_list(files['CELL_FILE'], files['DRUG_FILE'],
        #                                      files['CELL_DRUG_RESPONSE_FILE'], test_cells)
        negative_list = extract_negative_list(
                files['CELL_DRUG_FILE'].format(args.fold, k),
                files['RESISTANT_FILE']
        )
    else:
        #negative_list = extract_negative_list(files['CELL_DRUG_FILE'].format(args.fold, k))
        negative_list = extract_negative_list(
                files['CELL_DRUG_FILE'].format(args.fold, k),
                files['RESISTANT_FILE']
        )
        cnt = 0
        for _, item in negative_list.items():
            if len(item) < 10:
                cnt += 1
        print(f"# of nodes whose negative samples are not enough : {cnt}")

    if args.extra:
        negative_list.update(extract_negative_list(files['extra']['CELL_DRUG_ALL_FILE']))
    return negative_list

