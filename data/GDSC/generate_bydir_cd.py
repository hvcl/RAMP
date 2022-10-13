
with open('./Cell-Drug.dir.txt', 'r') as f_dir:
    dir_edges = f_dir.readlines()
    with open('./Cell-Drug.bydir.txt', 'w') as f_bydir:
        for row in dir_edges:
            cell, drug, value = row.split('\t')
            f_bydir.write(row)
            f_bydir.write('\t'.join([drug, cell, value]))
            
