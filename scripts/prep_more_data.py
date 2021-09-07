#!/usr/bin/env python3

###############################################################################
# This file was added to the original FewRel repository as part of the work
# described in the paper "Towards Realistic Few-Shot Relation Extraction",
# published in EMNLP 2021.
#
# It contains utility methods for converting additional datasets for the FewRel
# JSON format. The additional datasets can be found here:
#   TACRED: https://nlp.stanford.edu/projects/tacred/
#   CRE: https://raw.githubusercontent.com/shacharosn/CRE/main/challenge_set.json
#   MAVEN: https://drive.google.com/drive/folders/19Q0lqJE6A98OLnRqQVhbX3e6rG4BVGn8?usp=sharing
# 
# Authors: Sam Brody (sbrody18@bloomberg.net), Sichao Wu (swu389@bloomberg.net),
#          Adrian Benton (abenton10@bloomberg.net)
###############################################################################

import json
import os
import random
from typing import List, Optional

import pandas as pd

# directories where raw data live, manually downloaded/cloned
TACRED_DIR = "/mnt2/data/tacred/data/json/"
CRE_DIR = "/mnt2/data/CRE/"
MAVEN_DIR = "/mnt2/data/MAVEN/"

# root directory to save processed files
ROOT_OUT_DIR = "./data/"

# FewRel input is a JSON object, where keys are relation IDs and values are lists of entity pairs
# and the context (tokenized sentence) where that relation holds:
#
# {
#    'tokens': ['words', 'in', 'the', 'sentence'],
#    'h': [
#        'lowercased_head_word',
#        'entity_id?', 
#        [
#            [indices, of head, words]
#        ]
#    ],
#    't': [
#        'lowercased_tail_word', 
#        'entity_id?', 
#        [
#            [indices, of, tail, words]
#        ]
#    ]
# }
#
# There's also an accompanying pid2name.json file which give names and
# descriptions for relations in the data.  The second field in the head
# and tail entities is unused.


def prep_tacredfmt_data(
    out_dir: str,
    in_dir: str,
    base_in_paths: List[str],
    rel_key: str = "relation",
    fold_pct: Optional[List[float]] = None,
):
    # prepare TACRED-formatted data, but also include information like example ID, entity

    rel_to_id = {}
    ent_to_id = {}
    curr_rid = 1
    curr_eid = 1

    rel_cnts = {"rel": [], "htype": [], "ttype": [], "id": []}

    in_paths = [os.path.join(in_dir, p) for p in base_in_paths]
    out_paths = [os.path.join(out_dir, os.path.basename(p)) for p in in_paths]

    for inp, outp in zip(in_paths, out_paths):
        objs = {}

        if fold_pct is not None:
            random.shuffle(objs)

            new_inps = []

        with open(inp, "rt") as f:
            tacred_objs = json.load(f)
            for o in tacred_objs:
                rel = o[rel_key]
                if rel not in rel_to_id:
                    rel_to_id[rel] = f"P{curr_rid}"
                    curr_rid += 1
                rel_id = rel_to_id[rel]
                if rel_id not in objs:
                    objs[rel_id] = []

                tokens = o["token"]
                hs, he = o["subj_start"], o["subj_end"]
                obs, obe = o["obj_start"], o["obj_end"]

                htoken = " ".join(tokens[hs : (he + 1)]).lower()
                ttoken = " ".join(tokens[obs : (obe + 1)]).lower()

                if htoken not in ent_to_id:
                    ent_to_id[htoken] = f"Q{curr_eid}"
                    curr_eid += 1

                if ttoken not in ent_to_id:
                    ent_to_id[ttoken] = f"Q{curr_eid}"
                    curr_eid += 1

                head = [
                    htoken,
                    ent_to_id[htoken],
                    [list(range(hs, he + 1))],
                    o["subj_type"],
                ]
                tail = [
                    ttoken,
                    ent_to_id[ttoken],
                    [list(range(obs, obe + 1))],
                    o["obj_type"],
                ]

                objs[rel_id].append(
                    {
                        "tokens": tokens,
                        "h": head,
                        "t": tail,
                        "id": o["id"],
                        "stanford_ner": o["stanford_ner"],
                    }
                )

                rel_cnts["rel"].append(rel)
                rel_cnts["htype"].append(head[3])
                rel_cnts["ttype"].append(tail[3])
                rel_cnts["id"].append(o["id"])

        with open(outp, "wt") as out_file:
            out_file.write(json.dumps(objs))

    pid2name = {v: [k, ""] for k, v in rel_to_id.items()}
    with open(os.path.join(out_dir, "pid2name.json"), "wt") as out_file:
        out_file.write(json.dumps(pid2name))

    rel_cnts = pd.DataFrame(rel_cnts)
    rel_cnts.to_csv(
        os.path.join(out_dir, "relation_ent_types.tsv"),
        header=True,
        index=False,
        sep="\t",
    )

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # print stats for relations, entity types
    print(f"N={rel_cnts.shape[0]}\n")

    print("== Relation counts ==")
    print(rel_cnts.groupby("rel")["id"].count())

    print("== Head entities ==")
    print(rel_cnts.groupby("htype")["id"].count())

    print("== Tail entities ==")
    print(rel_cnts.groupby("ttype")["id"].count())

    print("== <relation, head entity, tail entity> triples ==")
    print(rel_cnts.groupby(["rel", "htype", "ttype"])["id"].count())


def prep_tacred(out_dir: str):
    prep_tacredfmt_data(
        out_dir, TACRED_DIR, ["train.json", "dev.json", "test.json"], "relation"
    )


def prep_cre(out_dir: str):
    prep_tacredfmt_data(
        out_dir,
        CRE_DIR,
        [
            "challenge_set.train.json",
            "challenge_set.dev.json",
            "challenge_set.test.json",
        ],
        "gold_relation",
    )


def prep_maven(out_dir: str):
    raise NotImplementedException(
        "MAVEN does not appear to include annotations for head & tail entities"
    )


def main():
    for nm, prep_fn in [
        ("tacred", prep_tacred),
        ("cre", prep_cre),
        # ('maven', prep_maven)
    ]:
        print(f'Preparing "{nm}"')

        out_dir = os.path.join(ROOT_OUT_DIR, nm)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        prep_fn(out_dir)


if __name__ == "__main__":
    if not os.path.exists(ROOT_OUT_DIR):
        os.mkdir(ROOT_OUT_DIR)

    main()
