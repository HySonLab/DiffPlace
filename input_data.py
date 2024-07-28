'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
# import numpy as np
# import sys
# import pickle as pkl
# import networkx as nx
# import scipy.sparse as sp

# def parse_index_file(filename):
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index

# def load_data(dataset):
#     # load the data: x, tx, allx, graph
#     names = ['x', 'tx', 'allx', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#     x, tx, allx, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
#     test_idx_range = np.sort(test_idx_reorder)

#     if dataset == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended

#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#     return adj, features
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import OneHotEncoder
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import networkx as nx
@dataclass
class Attribute:
    key: str
    value: str

@dataclass
class Node:
    name: str
    attr: Dict[str, Attribute] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)

@dataclass
class Macro(Node):
    height: float = 0.0
    width: float = 0.0
    x: float = 0.0
    y: float = 0.0
    orientation: str = ""

@dataclass
class SoftMacro(Macro):
    pass

@dataclass
class StandardCell(Node):
    height: float = 0.0
    width: float = 0.0
    x: float = 0.0
    y: float = 0.0
    weight: float = 1.0

@dataclass
class Port(Node):
    side: str = ""
    x: float = 0.0
    y: float = 0.0
    weight: float = 1.0

@dataclass
class MacroPin(Node):
    macro_name: str = ""
    x_offset: float = 0.0
    y_offset: float = 0.0
    x: Optional[float] = None
    y: Optional[float] = None
    weight: float = 1.0

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = {}
        self.node_list = []

    def add_node(self, node: Node):
        self.nodes[node.name] = node
        if node.name not in self.edges:
            self.edges[node.name] = []
        self.node_list.append(node.name)

    def add_edge(self, from_node: str, to_node: str):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.node_list)

    def __getitem__(self, node_name):
        return self.nodes[node_name]

    def encode(self):
        # Create node mapping
        node_mapping = {node: i for i, node in enumerate(self.nodes.keys())}

        # Create adjacency matrix
        num_nodes = len(self.nodes)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for src, dsts in self.edges.items():
            for dst in dsts:
                adj[node_mapping[src], node_mapping[dst]] = 1
                adj[node_mapping[dst], node_mapping[src]] = 1  # Make it symmetric

        # Create feature matrix (using one-hot encoding for node types)
        node_types = [type(node).__name__ for node in self.nodes.values()]
        encoder = OneHotEncoder(sparse_output=False)
        features = encoder.fit_transform(np.array(node_types).reshape(-1, 1))
        features = torch.FloatTensor(features)

        # Normalize features and adjacency matrix
        features = self.normalize_features(features)
        adj = self.normalize_adj(adj)

        return adj, features

    @staticmethod
    def normalize_features(features):
        rowsum = features.sum(dim=1, keepdim=True)
        rowsum[rowsum == 0] = 1  # Avoid division by zero
        return features / rowsum

    @staticmethod
    def normalize_adj(adj):
        adj = adj + torch.eye(adj.shape[0])
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
def parse_pb_txt(file_path: str) -> Graph:
    graph = Graph()
    current_node = None
    line_number = 0

    def parse_value(value_line):
        if 'placeholder:' in value_line:
            return value_line.split('"')[1] if '"' in value_line else value_line.split(':')[1].strip()
        elif 'f:' in value_line:
            return float(value_line.split(':')[1].strip())
        elif ':' in value_line:
            return value_line.split(':')[1].strip()
        else:
            return value_line.strip()

    with open(file_path, 'r') as f:
        lines = f.readlines()
        while line_number < len(lines):
            line = lines[line_number].strip()
            line_number += 1

            try:
                if line.startswith('node {'):
                    current_node = Node("")
                elif line.startswith('name:'):
                    current_node.name = line.split('"')[1] if '"' in line else line.split(':')[1].strip()
                elif line.startswith('input:'):
                    input_node = line.split('"')[1] if '"' in line else line.split(':')[1].strip()
                    current_node.inputs.append(input_node)
                    graph.add_edge(input_node, current_node.name)
                elif line.startswith('attr {'):
                  if current_node is not None:
                    while not lines[line_number].strip().startswith('}'):
                        attr_line = lines[line_number].strip()
                        line_number += 1
                        if attr_line.startswith('key:'):
                            key = attr_line.split('"')[1] if '"' in attr_line else attr_line.split(':')[1].strip()
                            value_line = lines[line_number].strip()
                            line_number += 1
                            value = parse_value(value_line)
                            current_node.attr[key] = Attribute(key, str(value))
                    line_number += 1  # Skip the closing brace of attr
                elif line == '}' and current_node:
                    node_type = current_node.attr.get('type', Attribute('type', '')).value

                    if node_type.lower() == 'macro':
                        node = Macro(current_node.name, current_node.attr, current_node.inputs)
                    elif node_type.lower() == 'stdcell':
                        node = StandardCell(current_node.name, current_node.attr, current_node.inputs)
                    elif node_type.lower() == 'port':
                        node = Port(current_node.name, current_node.attr, current_node.inputs)
                    elif node_type.lower() == 'macro_pin':
                        node = MacroPin(current_node.name, current_node.attr, current_node.inputs)
                    elif current_node.name.startswith('Grp'):
                        node = SoftMacro(current_node.name, current_node.attr, current_node.inputs)
                    else:
                        node = current_node

                    graph.add_node(node)
                    current_node = None
            except Exception as e:
                print(f"Error parsing line {line_number}: {line}")
                print(f"Error message: {str(e)}")
                raise

    return graph
def load_netlist_data(file_path):
    graph = parse_pb_txt(file_path)
    adj, features = graph.encode()
    return adj, features