import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='[RAMP] Generator of Scripts for Incremental Learning')
    parser.add_argument('--dataset', default='GDSC', type=str,
            help="Dataset to use between GDSC and TCGA")
    parser.add_argument('--num_shells', default=4, type=int,
            help="Number of shells to use")
    parser.add_argument('--num_walks', default=18000, type=int,
            help="H-param of Node2Vec")

    with open(f'../Dataset/{args.dataset}/cell_list.txt', 'r') as f:
        nodes = f.readlines()
    nodes = [node.replace('\n', '') for node in nodes]
    num_nodes_per_shell = len(nodes) // num_shells

    for s in range(args.num_shells):
        start, end = s * num_nodes_per_shell, (s + 1) * num_nodes_per_shell
        if s == args.num_shells - 1:
            end = len(nodes)
        with open(f'./il-{args.dataset}-nw{args.num_walks}-{s}.sh', 'w') as f:
            for idx in range(start, end):
                node = nodes[idx]
                f.write(f'echo "{node}" >| ./designate-{args.dataset}-{s}.txt\n')
                f.write(f'python main.py --fname_ext {s} --worker 1 --num_walks {args.num_walks} 
                        --fix_gdsc --external {args.dataset} 
                        --walkers ./Source/designate-{args.dataset}-{s}.txt\n')

