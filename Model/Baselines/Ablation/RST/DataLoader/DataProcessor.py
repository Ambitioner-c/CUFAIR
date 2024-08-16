# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/16 19:57
import codecs
import collections
import os
import re
from abc import ABC
from xml.dom import minidom

from pyexpat import ExpatError

import pandas as pd
from six import iterkeys
from transformers import DataProcessor


class Node:
    def __init__(self, idx, left, right, parent, depth, kind, text, rel_name, rel_kind, secedges):
        """
        Basic class to hold all nodes (EDU, span and multinuc) in structure.py and while importing
        """
        self.idx = idx
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = depth
        self.kind = kind    # edu, multinuc or span node
        self.text = text    # text of an edu node; empty for spans/multinucs
        self.rel_name = rel_name
        self.rel_kind = rel_kind    # rst (a.k.a. satellite), multinuc or span relation
        self.sort_depth = depth
        self.secedges = secedges

    def __str__(self):
        return f'Node(idx={self.idx}, left={self.left}, right={self.right}, parent={self.parent}, depth={self.depth}, ' \
               f'kind={self.kind}, text={self.text}, rel_name={self.rel_name}, rel_kind={self.rel_kind}, ' \
               f'sort_depth={self.sort_depth}, secedges={self.secedges})'


class Segment:
    def __init__(self, idx, text):
        """
        Class used by segment.py to represent EDUs, NOT used by the structurer in structure.py
        """
        self.idx = idx
        self.text = text
        self.tokens = text.split(' ')


def get_depth(orig_node, probe_node, nodes, doc=None, project=None, user=None):
    """
	Calculate graphical nesting depth of a node based on the node list graph.
    Note that RST parentage without span/multinuc does NOT increase depth.
    """
    if probe_node.parent != '0':
        try:
            parent = nodes[probe_node.parent]
        except KeyError:
            raise KeyError("Node ID " + probe_node.id + " has non existing parent " + probe_node.parent + " and user not set in function\n")
        if parent.kind != 'edu' and (probe_node.rel_name == "span" or parent.kind == "multinuc" and probe_node.rel_kind =="multinuc"):
            orig_node.depth += 1
            orig_node.sort_depth += 1
        elif parent.kind == 'edu':
            orig_node.sort_depth += 1
        get_depth(orig_node, parent, nodes, doc=doc, project=project, user=user)


def get_text(node: Node, nodes: {str: Node}):
    if node.kind != 'edu' and node.text == '':
        left = node.left
        right = node.right
        text = ''
        for idx in range(int(left), int(right) + 1):
            text += nodes[str(idx)].text + ' '
        node.text = text.strip()

def get_rel_kind(node: Node):
    if node.rel_name == 'span':
        node.rel_kind = 'span'
    elif '_r' in node.rel_name:
        node.rel_kind = 'satellite'
    elif '_m' in node.rel_name:
        node.rel_kind = 'multinuc'


def get_left_right(node_idx, nodes, min_left, max_right, rel_hash):
    """
    Calculate leftmost and rightmost EDU covered by a NODE object. For EDUs this is the number of the EDU
	itself. For spans and multinucs, the leftmost and rightmost child dominated by the NODE is found recursively.
    """
    if nodes[node_idx].parent != '0' and node_idx != '0':
        parent = nodes[nodes[node_idx].parent]
        if min_left > nodes[node_idx].left or min_left == 0:
            if nodes[node_idx].left != 0:
                min_left = nodes[node_idx].left
        if max_right < nodes[node_idx].right or max_right == 0:
            max_right = nodes[node_idx].right
        if nodes[node_idx].rel_name == "span":
            if parent.left > min_left or parent.left == 0:
                parent.left = min_left
            if parent.right < max_right:
                parent.right = max_right
        elif nodes[node_idx].rel_name in rel_hash:
            if parent.kind == "multinuc" and rel_hash[nodes[node_idx].rel_name] =="multinuc":
                if parent.left > min_left or parent.left == 0:
                    parent.left = min_left
                if parent.right < max_right:
                    parent.right = max_right
        get_left_right(parent.idx, nodes, min_left, max_right, rel_hash)


def read_rst(filename, rel_hash, as_string=False):
    if as_string:
        in_rs4 = filename
    else:
        f = codecs.open(filename, 'r', 'utf-8')
        in_rs4 = f.read()

    # Remove processing instruction
    in_rs4 = re.sub(r'<\?xml[^<>]*?\?>','',in_rs4)
    try:
        xml_doc = minidom.parseString(codecs.encode(in_rs4, 'utf-8'))
    except ExpatError:
        message = 'Invalid .rs4 file'
        return message, None

    nodes = []
    ordered_id = {}
    schemas = []
    default_rst = ""

    # Get relation names and their types, append type suffix to disambiguate
    # relation names that can be both RST and multinuc
    item_list = xml_doc.getElementsByTagName('rel')
    for rel in item_list:
        rel_name = re.sub(r"[:;,]", '', rel.attributes['name'].value)
        if rel.hasAttribute('type'):
            rel_hash[rel_name + '_' + rel.attributes['type'].value[0:1]] = rel.attributes['type'].value
            if rel.attributes['type'].value == 'rst' and default_rst == "":
                default_rst = rel_name + '_' + rel.attributes['type'].value[0:1]
        else:
            schemas.append(rel_name)

    item_list = xml_doc.getElementsByTagName('segment')
    if len(item_list) < 1:
        return 'No segment elements found in .rs4 file.', None

    id_counter = 0
    total_tokens = 0

    # Get hash to reorder EDUs and spans according to the order of appearance in .rs3 file
    for segment in item_list:
        id_counter += 1
        ordered_id[segment.attributes['id'].value] = id_counter
    item_list = xml_doc.getElementsByTagName('group')
    for group in item_list:
        id_counter += 1
        ordered_id[group.attributes['id'].value] = id_counter
    all_node_ids = set(range(1, id_counter + 1))    # All non-zero IDs in documents, which a signal may refer back to
    ordered_id['0'] = 0

    element_types = {}
    node_elements = xml_doc.getElementsByTagName('segment')
    for element in node_elements:
        element_types[element.attributes['id'].value] = 'edu'
    node_elements = xml_doc.getElementsByTagName('group')
    for element in node_elements:
        element_types[element.attributes['id'].value] = element.attributes['type'].value

    # Collect all children of multinuc parents to prioritize which potentially multinuc relation they have
    item_list = xml_doc.getElementsByTagName('segment') + xml_doc.getElementsByTagName('group')
    multinuc_children = collections.defaultdict(lambda: collections.defaultdict(int))
    for element in item_list:
        if element.attributes.length >= 3:
            parent = element.attributes['parent'].value
            rel_name = element.attributes['relname'].value
            # Tolerate schemas by treating as spans
            if rel_name in schemas:
                rel_name = 'span'
            rel_name = re.sub(r'[:;,]', '', rel_name)   # Remove characters used for undo logging, not allowed in rel names
            if parent in element_types:
                if element_types[parent] == 'multinuc' and rel_name + '_m' in rel_hash:
                    multinuc_children[parent][rel_name] += 1

    id_counter = 0
    item_list = xml_doc.getElementsByTagName('segment')
    for segment in item_list:
        id_counter += 1
        if segment.hasAttribute('parent'):
            parent = segment.attributes['parent'].value
        else:
            parent = '0'
        if segment.hasAttribute('relname'):
            rel_name = segment.attributes['relname'].value
        else:
            rel_name = default_rst

        # Tolerate schemas, but no real support yet:
        if rel_name in schemas:
            rel_name = 'span'
            rel_name = re.sub(r'[:;,]', '', rel_name)

        # Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
        if parent in multinuc_children:
            if len(multinuc_children[parent]) > 0:
                key_list = list(iterkeys(multinuc_children[parent]))[:]
                for key in key_list:
                    if multinuc_children[parent][key] < 2:
                        del multinuc_children[parent][key]

        if parent in element_types:
            if element_types[parent] == "multinuc" and rel_name + "_m" in rel_hash and (
                    rel_name in multinuc_children[parent] or len(multinuc_children[parent]) == 0):
                rel_name = rel_name + "_m"
            elif rel_name != 'span':
                rel_name = rel_name + "_r"
        else:
            if not rel_name.endswith('_r') and len(rel_name) > 0:
                rel_name = rel_name + "_r"
        edu_id = segment.attributes['id'].value
        if len(segment.childNodes) > 0: # Check the node is not empty
            contents = segment.childNodes[0].data.strip()
            if len(contents) == 0:
                continue
        else:
            continue

        # Check for invalid XML in segment contents
        if '<' in contents or '>' in contents or '&' in contents:
            contents = contents.replace('>', '&gt;')
            contents = contents.replace('<', '&lt;')
            contents = re.sub(r'&([^ ;]* )', r'&amp;\1', contents)
            contents = re.sub(r'&$', r'&amp;', contents)

        total_tokens += contents.strip().count(' ') + 1
        nodes.append([str(ordered_id[edu_id]), id_counter, id_counter, str(ordered_id[parent]), 0, 'edu', contents, rel_name])

    item_list = xml_doc.getElementsByTagName('group')
    for group in item_list:
        if group.attributes.length == 4:
            parent = group.attributes['parent'].value
        else:
            parent = '0'
        if group.attributes.length == 4:
            rel_name = group.attributes['relname'].value
            # Tolerate schemas by treating as spans
            if rel_name in schemas:
                rel_name = 'span'

            rel_name = re.sub(r"[:;,]","",rel_name)

            # Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
            if parent in multinuc_children:
                if len(multinuc_children[parent]) > 0:
                    key_list = list(iterkeys(multinuc_children[parent]))[:]
                    for key in key_list:
                        if multinuc_children[parent][key] < 2:
                            del multinuc_children[parent][key]

            if parent in element_types:
                if element_types[parent] == "multinuc" and rel_name+"_m" in rel_hash and (
                        rel_name in multinuc_children[parent] or len(multinuc_children[parent]) == 0):
                    rel_name = rel_name + "_m"
                elif rel_name != 'span':
                    rel_name = rel_name + "_r"
            else:
                rel_name = ''
        else:
            rel_name = ''
        group_id = group.attributes['id'].value
        group_type = group.attributes['type'].value
        contents = ''
        nodes.append([str(ordered_id[group_id]), 0, 0, str(ordered_id[parent]), 0, group_type, contents, rel_name])

    # Collect discourse signal annotations if any are available
    item_list = xml_doc.getElementsByTagName('signal')
    signals = []
    for signal in item_list:
        source = signal.attributes['source'].value
        if '-' in source:   # Secedge signal
            src, trg = source.split('-')
            # We assume .rs4 format files are properly ordered, so directly look up IDs from secedge
            if int(src) not in all_node_ids or int(trg) not in all_node_ids:
                raise IOError("Invalid secedge ID for signal: " + str(source) + " (from XML file source="+signal.attributes["source"].value+")\n")
            src = str(ordered_id[src])
            trg = str(ordered_id[trg])
            source = src + '-' + trg
        else:
            # This will crash if signal source refers to a non-existing node:
            source = ordered_id[source]
            if source not in all_node_ids:
                raise IOError("Invalid source node ID for signal: " + str(source) + " (from XML file source=" + signal.attributes["source"].value + ")\n")
        signal_type = signal.attributes['type'].value
        signal_subtype = signal.attributes['subtype'].value
        tokens = signal.attributes['tokens'].value
        if tokens != '':
            # This will crash if tokens contains non-numbers:
            token_list = [int(token) for token in tokens.split(',')]
            max_token = max(token_list)
            if max_token > total_tokens:
                raise IOError("Signal refers to non-existent token: " + str(max_token))
        signals.append([str(source), signal_type, signal_subtype, tokens])

    # Collect signal type inventory declaration if available
    item_list = xml_doc.getElementsByTagName('sig')
    signal_type_dict = {}
    for sig in item_list:
        sig_type = sig.attributes['type'].value
        sig_subtypes = sig.attributes['subtypes'].value.split(';')
        signal_type_dict[sig_type] = sig_subtypes
    if len(signal_type_dict) == 0:
        signal_type_dict = None

    # Collect secondary edges if any are available
    item_list = xml_doc.getElementsByTagName('secedge')
    secedge_dict = collections.defaultdict(set)
    for secedge in item_list:
        source = secedge.attributes['source'].value
        target = secedge.attributes['target'].value
        rel_name = secedge.attributes['relname'].value
        # This will crash if signal source or target refers to a non-existing node:
        source = ordered_id[source]
        target = ordered_id[target]
        if source not in all_node_ids:
            raise IOError("Invalid source node ID for secedge: " + str(source) + " (from XML file source=" + secedge.attributes["source"].value + "(\n")
        if target not in all_node_ids:
            raise IOError("Invalid target node ID for secedge: " + str(target) + " (from XML file source=" + secedge.attributes["target"].value + "(\n")
        secedge_dict[str(source)].add(str(source) + '-' + str(target) + ':' + rel_name + '_r')

    elements = {}
    for row in nodes:
        secedges = ';'.join(secedge_dict[row[0]]) if row[0] in secedge_dict else ''
        elements[row[0]] = Node(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], "", secedges)

    for element in elements:
        if elements[element].kind == 'edu':
            get_left_right(element, elements, 0, 0, rel_hash)

        get_rel_kind(elements[element])

        get_depth(elements[element], elements[element], elements)

        get_text(elements[element], elements)

    return elements, signals, signal_type_dict


def pair_nodes(nodes: {str: Node}):
    n_s_pairs = {}
    s_n_pairs = {}
    n_n_pairs = {}
    for idx in nodes:
        node = nodes[idx]
        rel_kind = node.rel_kind
        if rel_kind == 'satellite':
            satellite = node.text
            nucleus = nodes[node.parent].text
            if node.left < nodes[node.parent].left:
                s_n_pairs[idx] = [satellite, nucleus]
            elif node.right > nodes[node.parent].right:
                n_s_pairs[idx] = [nucleus, satellite]
            else:
                raise Exception('Satellite is not on the left or right of the nucleus')
        elif rel_kind == 'multinuc':
            nucleus = node.text

            parent = node.parent
            if parent not in n_n_pairs:
                n_n_pairs[parent] = [{node.right: nucleus}]
            else:
                n_n_pairs[parent].append({node.right: nucleus})
                n_n_pairs[parent].sort(key=lambda x: list(x.keys())[0])

    return n_s_pairs, s_n_pairs, n_n_pairs


class RSTProcessor(DataProcessor, ABC):
    def __init__(self):
        super(RSTProcessor, self).__init__()

    def get_train_examples(self, data_dir: str):
        return self.create_examples([os.path.join(data_dir, 'Train', x) for x in sorted(os.listdir(data_dir + '/Train'))])


    def get_dev_examples(self, data_dir: str):
        return self.create_examples([os.path.join(data_dir, 'Dev', x) for x in sorted(os.listdir(data_dir + '/Dev'))])

    def get_test_examples(self, data_dir: str):
        return self.create_examples([os.path.join(data_dir, 'Test', x) for x in sorted(os.listdir(data_dir + '/Test'))])

    def get_labels(self):
        pass

    @staticmethod
    def create_examples(filepaths: list) -> pd.DataFrame:
        lefts = []
        rights = []
        labels = []
        for filepath in filepaths:
            elements, _, _ = read_rst(filepath, rel_hash={})
            n_s_pairs, s_n_pairs, n_n_pairs = pair_nodes(elements)

            # N-S
            for idx in n_s_pairs:
                lefts.append(n_s_pairs[idx][0])
                rights.append(n_s_pairs[idx][1])
                labels.append(0)

            # S-N
            for idx in s_n_pairs:
                lefts.append(s_n_pairs[idx][0])
                rights.append(s_n_pairs[idx][1])
                labels.append(1)

            # N-N
            for idx in n_n_pairs:
                for j in range(len(n_n_pairs[idx]) - 1):
                    lefts.append(list(n_n_pairs[idx][j].values())[0])
                    rights.append(list(n_n_pairs[idx][j + 1].values())[0])
                    labels.append(2)

        df = pd.DataFrame({
            'left': lefts,
            'right': rights,
            'label': labels
        })

        return df


def main():
    data_dir = '/home/cuifulai/Projects/CQA/Data/RST/GUM'

    df = RSTProcessor().get_train_examples(data_dir)
    print(df.head(10).to_csv())


if __name__ == '__main__':
    main()
