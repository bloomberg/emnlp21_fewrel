#!/usr/bin/env python3

###############################################################################
# This file was added to the original FewRel repository as part of the work
# described in the paper "Towards Realistic Few-Shot Relation Extraction",
# published in EMNLP 2021.
#
# It contains a utility method to convert the FewRel JSON data into a more
# human readable format by adding line breaks and indentation. The process
# does not modify the actual data.
# 
# Authors: Sam Brody (sbrody18@bloomberg.net), Sichao Wu (swu389@bloomberg.net),
#          Adrian Benton (abenton10@bloomberg.net)
###############################################################################

import json
import os
import sys


def indent(n):
    return "  " * n


def remove_last_comma(s):
    assert s[-2:] == ",\n"
    s = s[:-2] + "\n"
    return s


def format_data(data) -> str:
    """Returns a pretty-formatted string version of the input JSON object."""
    out = "{\n"
    for rel in data:
        out += f"{indent(1)}{json.dumps(rel)}: [\n"
        for i, instance in enumerate(data[rel]):
            out += f"{indent(2)}{{\n"
            for key in instance:
                out += f"{indent(3)}{json.dumps(key)}: {json.dumps(instance[key])},\n"
            out = remove_last_comma(out)
            out += f"{indent(2)}}},\n"
        out = remove_last_comma(out)
        out += "  ],\n"
    out = remove_last_comma(out)
    out += "}\n"
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input json> <output json>")
        exit()
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    data = json.load(open(in_file, "rt"))
    with open(out_file, "wt") as out:
        out.write(format_data(data))
