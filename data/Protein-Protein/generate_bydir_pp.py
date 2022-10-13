with open('./New.Protein-Protein.Weight.txt', 'r') as f_dir:
    dir = f_dir.readlines()
    with open('./IRefindex_protein_protein_merged.txt', 'w') as f_bydir:
        for row in dir:
            p1, p2, value = row.split('\t')
            f_bydir.write('\t'.join([p1, p2, value]))
            f_bydir.write('\t'.join([p2, p1, value]))

with open('./Curation.Protein-Protein.dir.txt', 'r') as f_dir:
    dir = f_dir.readlines()
    with open('./IRefindex_protein_protein_curated.txt', 'w') as f_bydir:
        for row in dir:
            p1, p2, value = row.split('\t')[:3]
            f_bydir.write('\t'.join([p1, p2, value]) + '\n')
            f_bydir.write('\t'.join([p2, p1, value]) + '\n')

