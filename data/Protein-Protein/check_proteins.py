import argparse


def save_to_txt(content: set, path):
    with open(path, 'w') as f:
        for node in content:
            f.write(f'{node}\n')


def load_file(path):
    with open(path, 'r') as f:
        contents = f.readlines()
    return contents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check CCLE Dataset.')
    parser.add_argument('--gen_list', action='store_true', help='Generate list file for Cell lines and Drugs for each')
    
    args = parser.parse_args()
    logger = open('check_proteins.log', 'a')

    merged = load_file('Merged.Protein-Protein.dir.txt')
    len_merged = len(merged)
    proteins = set()
    for idx in range(1, len_merged):
        row = merged[idx]
        info = row.replace('"', '').split()
        proteins.add(info[1])
        proteins.add(info[2])

    logger.write(f'# of Merged P-P Interactions: {len(merged) - 1}\n')
    logger.write(f'# of Proteins: {len(proteins)}\n')

    logger.close()
