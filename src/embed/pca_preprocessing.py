import numpy as np
from sklearn.decomposition import PCA

f_pp = open("../Dataset/IRefindex_protein_protein.txt", 'r')
f_cp = open("../Dataset/IRefindex_cell_protein.txt", 'r')
pp = f_pp.readlines()
cp = f_cp.readlines()
possible_proteins = set()
for l in pp:
    s = l.split()
    possible_proteins.add(s[0])
f_pp.close()
for l in cp:
    s = l.split()
    possible_proteins.add(s[0])
f_cp.close()
print("# of Proteins in the IRefindex files : {}".format(len(list(possible_proteins))))


# Read Protens' initial feature vectors (not preprocessed)
f_before = open("../Dataset/Pathway/GOBP.GeneVector.txt", 'r')

proteins = f_before.readlines()
p_name = []
p_vec = []
for i in range(1, len(proteins)):
    p = proteins[i].split('\t')

    # Check if value == NA
    check = set(p[1:])
    if "NA" in check:
        continue

    if p[0] in possible_proteins:
        p_name.append(p[0])
        str_to_float = list(map(float, p[1:]))
        p_vec.append(str_to_float)

target_dim = 64
print("# of intersect Proteins between the GeneVector file & IRefindex files : {}".format(len(p_name)))
data = np.asarray(p_vec)
pca_model = PCA(n_components=target_dim)
pca_model.fit(data)
rst = pca_model.transform(data)

f_after = open("../Dataset/Pathway/GOBP.GeneVector.PCA.txt", 'w')
f_after.write("{} {}\n".format(len(rst), target_dim))
for i in range(len(rst)):
    p = p_name[i]
    for j in range(target_dim):
        p += " {}".format(rst[i][j])
    f_after.write(p + "\n")

