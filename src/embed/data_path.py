PATH = {
    'CELL_DRUG_RESPONSE_FILE': 'cell_drug.mat',
    'CELL_INDEX_FILE': 'index.mat',

    'CELL_FILE': 'cell_list.txt',
    'DRUG_FILE': 'drug_list.txt',

    'CELL_DRUG_FILE': '{}-fold/Cell-Drug-{}.txt',
    'CELL_DRUG_ALL_FILE': 'Cell-Drug.bydir.txt',

    'CELL_PROTEIN_FILE': 'IRefindex_cell_protein.txt',
}

MERGE = {
    'CELL_FILE': 'GDSC_MERGED_{}/cell_list.txt',
    'TEST_DRUG_FILE': 'GDSC_MERGED_{}/drug_list.txt',
    
    'CELL_DRUG_ALL_FILE': '{}/Cell-Drug.bydir.txt',
    'CELL_DRUG_VAL_FILE': '{}/10-fold/Cell-Drug-1.txt',
    'CELL_PROTEIN_FILE': '{}/IRefindex_cell_protein.txt',
}

PROTEIN = {
    'CURATED': 'Protein-Protein/IRefindex_protein_protein_curated.txt',
    'MERGED': 'Protein-Protein/IRefindex_protein_protein_merged.txt',
}

SAVE = {
    'EMBEDDING_FILE': 'total_embedding-{}.txt',
    'CELL_EMBEDDING_FILE': 'cell_embedding-{}.txt',
}

RESISTANT = {
    "GDSC": "GDSC/GDSC.Resistant.Cell-Drug.txt",
    "TCGA": "TCGA/TCGA.Resistant.Sample-Drug.txt",
}
