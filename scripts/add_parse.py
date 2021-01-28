import libnlp
import json
import os
import sys

from .prettify_data import format_data

_eng = libnlp.core.Engine()


def create_doc(tokens):
    start = 0
    doc = libnlp.core.Document(" ".join(tokens))
    token_list = libnlp.attr.BaseTokenList()
    for t in tokens:
        end = start + len(t.encode("utf-8"))
        token_attr = libnlp.attr.BaseToken(libnlp.core.Span(start, end), t)
        assert doc.contentAt(token_attr) == t
        token_list.insert(token_attr)
        start = end + 1
    doc.put(token_list, "Tokenizer")
    sentence_list = libnlp.attr.SentenceList()
    sentence_list.insert(libnlp.attr.Sentence(libnlp.core.Span(0, end)))
    assert doc.contentAt(sentence_list[0]) == doc.content
    doc.put(sentence_list, "Sentencer")
    
    return doc


def get_parse(instance):
    doc = create_doc(instance["tokens"])    
    dep_nodes = _eng.gen(doc, libnlp.attr.DependencyTreeNodeList)
    assert len(dep_nodes) == len(instance["tokens"])
    deps = []
    for n in dep_nodes:
        deps.append(int(n.head) - 1)
    return deps

def add_parse(data):
    for relation in data:
        print("processing relation:", relation)
        for i, instance in enumerate(data[relation]):
            if i % 100 == 0:
                print(i)
            instance["libnlp_head"] = get_parse(instance)
        print("processed", i, "instances")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python add_parse.py <input json> <output json> <recipe>")
        exit()
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    recipe = sys.argv[3]
    _eng.loadRecipe(recipe)
    data = json.load(open(in_file, "rt"))
    add_parse(data)
    with open(out_file, "wt") as out:
        out.write(format_data(data))