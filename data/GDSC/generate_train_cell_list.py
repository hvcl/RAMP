import scipy.io

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


for k in range(10):
    with open('./10-fold/Train-Cells-{}.txt'.format(k + 1), 'w') as f:
        test_list = extract_cellline_testset('./cell_list.txt', './index.mat', k)
        for cell in test_list:
            f.write(f'{cell}\n')

