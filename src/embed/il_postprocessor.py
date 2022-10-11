import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='[RAMP] Post-processing tool for outputs of Incremental Learning')
    parser.add_argument('--dataset', default='GDSC', type=str,
            help="Dataset to use between GDSC and TCGA")
    parser.add_argument('--num_shells', default=4, type=int,
            help="Number of shells to use")
    parser.add_argument('--num_walks', default=18000, type=int,
            help="H-param of Node2Vec")

    with open(f'../Dataset/{args.dataset}/cell_list.txt', 'r') as f:
        nodes = f.readlines()

    path = f'../EmbeddingData/GDSC_EXTERNAL_{args.dataset}/IL_NW{args.num_walks}/cell_embedding-1.txt'
    with open(path, 'r') as f:
        original = f.readlines()
    with open(path, 'w') as f:
        f.write(f'{len(nodes)} 64\n')
        for line in original:
            if '1 64' in line:
                continue
            else:
                f.write(line)


