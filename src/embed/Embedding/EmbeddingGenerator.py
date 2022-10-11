from node2vec import Node2Vec
import networkx as nx
import numpy as np

from Embedding.NegativeList import generate_negative_list
#from Embedding.Callbacks import get_callback_func


def modify_edge_weight_to_constant(graph, value):
    print(f'Fix Cell-Protein weight values to {value}')
    for u, v, d in graph.edges(data=True):
        d['weight'] = value


def load_graph(path, _dtype, fix_weight=0):
    with open(path, 'rb') as f:
        g = nx.read_edgelist(f, data=(('weight', _dtype),))

    if fix_weight:
        modify_edge_weight_to_constant(g, fix_weight)
    return g


def load_graph_without_weight(path, _dtype):
    with open(path, 'rb') as f:
        g = nx.read_edgelist(f)
    return g


def build_graph_from_files(args, files, k, walkers):
    cell_protein_graph = load_graph(files['CELL_PROTEIN_FILE'], float, args.fix_w)
    protein_protein_graph = load_graph(files['PROTEIN_PROTEIN_FILE'], float)
    if args.no_pp:
        protein_protein_graph = nx.create_empty_copy(protein_protein_graph)

    if args.loocv:
        graphs = [cell_protein_graph, protein_protein_graph]
    else:
        cell_drug_graph = load_graph(files['CELL_DRUG_FILE'].format(args.fold, k), int)
        graphs = [cell_drug_graph, cell_protein_graph, protein_protein_graph]

    if args.extra:
        extra = files['extra']
        extra_cd_graph = load_graph(extra['CELL_DRUG_ALL_FILE'], int)
        extra_cp_graph = load_graph(extra['CELL_PROTEIN_FILE'], float, args.fix_w)
        graphs.extend([extra_cd_graph, extra_cp_graph])
    elif args.external:
        external_cp_graph = load_graph(files['MERGED_CELL_PROTEIN_FILE'], float, args.fix_w)
        if walkers is not None:
            nodes_connected_to_walkers = []
            for walker in walkers:
                connected_to_walker = nx.all_neighbors(external_cp_graph, walker)
                nodes_connected_to_walkers.extend(list(connected_to_walker))
                nodes_connected_to_walkers.append(walker)
            external_cp_graph = external_cp_graph.subgraph(nodes_connected_to_walkers)
        graphs.append(external_cp_graph)
    graph = nx.compose_all(graphs)
    return graph


def use_walks_including_cells(walks, cell_list, args):
    external_cells_walks = []
    half_win = args.window // 2
    new_side = half_win * 2
    new_walk_size = args.window + half_win * 2
    for walk in walks:
        for c in cells:
            indices = np.where(walk == c)
            for idx in indices:
                if idx < new_side:
                    node_walk = walk[:new_walk_size]
                elif idx + new_side > args.len_walk:
                    node_walk = walk[-new_walk_size:]
                else:
                    node_walk = walk[idx - new_side:idx + new_side]
                external_cells_walks.append(node_walk)
    return external_cells_walks


def get_node_list_to_walk(walkers):
    if walkers:
        with open(walkers, 'r') as f:
            nodes = f.readlines()
        return [node.replace('\n', '') for node in nodes]
    return None


def generate_embedding(args, files, save_path):
    for k in range(args.fold_start, args.fold_stop + 1):
        print(f'Train Fold-{k}/{args.fold}..')
        walkers = get_node_list_to_walk(args.walkers)  # walkers: `None` means all nodes
        graph = build_graph_from_files(args, files, k, walkers)
        if args.walkers:
            node2vec = Node2Vec(graph, dimensions=args.dim, walk_length=args.len_walk, walkers=walkers,
                                num_walks=args.num_walks, workers=args.workers, p=args.p, q=args.q)
        else:
            node2vec = Node2Vec(graph, dimensions=args.dim, walk_length=args.len_walk,
                                num_walks=args.num_walks, workers=args.workers, p=args.p, q=args.q)

        if args.walkers:
            f_cell = open(f'./il_nw{args.num_walks}_count_cell_in_walks.txt', 'a')
            count = np.count_nonzero(np.array(node2vec.walks) == walkers[0])
            f_cell.write(f'{walkers[0]}: {count}\n')
            f_cell.close()

        callback_constraint = node2vec.walks if args.walkers else None
        callbacks = []
        #callbacks = get_callback_func(args, files, callback_constraint)

        negative_list = generate_negative_list(args, files, k)
        model = node2vec.fit(window=args.window, min_count=1, batch_words=4, negative=20,
                             callbacks=callbacks, negative_list=negative_list)
        model.wv.save_word2vec_format(save_path.format(str(k) + args.fname_ext))

