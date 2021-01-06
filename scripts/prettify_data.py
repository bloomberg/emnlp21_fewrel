import json
import os
import sys

def indent(n):
    return "  " * n

def remove_last_comma(s):
    assert s[-2:] == ",\n"
    s = s[:-2] + "\n"
    return s

def format_data(data)-> str:
    """Returns a pretty-formatted string version of the input JSON object."""
    out = "{\n"
    for rel in data:
        out += "{}{}: [\n".format(indent(1), json.dumps(rel))
        for i, instance in enumerate(data[rel]):
            out += "{}{{\n".format(indent(2))
            for key in instance:
                out += "{}{}: {},\n".format(indent(3), json.dumps(key), json.dumps(instance[key]))
            out = remove_last_comma(out)
            out += "{}}},\n".format(indent(2))                        
        out = remove_last_comma(out)            
        out += "  ],\n"
    out = remove_last_comma(out) 
    out += "}\n"
    return out



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python {} <input json> <output json>".format(sys.argv[0]))
        exit()
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    data = json.load(open(in_file, "rt"))
    with open(out_file, "wt") as out:
        out.write(format_data(data))