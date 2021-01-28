import json
import sys
from prettify_data import format_data


def get_lca_pair(hmap, n1, n2):
    seen = set()
    while n1 != -1 or n2 != -1:
        if n1 != -1:
            if n1 in seen:
                return n1
            seen.add(n1)
            n1 = hmap[n1]

        if n2 != -1:
            if n2 in seen:
                return n2
            seen.add(n2)
            n2 = hmap[n2]
    return -1


def get_lca(hmap, nodes):
    while len(nodes) > 1:
        n1 = nodes.pop(0)
        n2 = nodes.pop(0)
        lca = get_lca_pair(hmap, n1, n2)
        # There is one bad corner case in the result of libnlp parsing:
        # {0 : 1, 1 : - 1, 2 : 7, 3 : 7, 4 : 7, 5 : 7, 6 : 7, 7 : - 1, 8 : 7 }
        if lca == -1:
            lca = n1
        nodes.append(lca)
    return nodes[0]


def add_lca(json_data):
    for rel in json_data:
        instances = json_data[rel]
        for ins in instances:
            toks = ins["tokens"]
            ent1_indexs = ins["h"][2][0]
            ent2_indexs = ins["t"][2][0]
            libnlp_heads = ins["libnlp_head"]
            hmap = {i: libnlp_heads[i] for i in range(0, len(libnlp_heads))}
            ins["libnlp_lca"] = get_lca(hmap, list(set(ent1_indexs + ent2_indexs)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_lca.py <input json> <output json>")
        exit()
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    json_data = json.load(open(in_file))
    add_lca(json_data)
    with open(out_file, "wt") as out:
        out.write(format_data(json_data))
