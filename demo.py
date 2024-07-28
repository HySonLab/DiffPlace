import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import time
from model import GAE, VGAE 
from input_data import load_netlist_data
import args
import model

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

adj, features = load_netlist_data('/content/drive/MyDrive/Chip_Design/GDM/ariane.pb.txt')

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj.clone()
adj_orig = adj_orig - torch.diag(adj_orig.diag())

# Split edges for train/val/test
def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    n_nodes = adj.size(0)
    n_edges = adj.sum().item() // 2  # Each edge is counted twice in adjacency matrix
    
    all_edges = torch.nonzero(torch.triu(adj, diagonal=1))
    all_edges = all_edges[torch.randperm(all_edges.size(0))]
    
    n_test = int(np.floor(n_edges * 0.1))
    n_val = int(np.floor(n_edges * 0.05))
    
    test_edges = all_edges[:n_test]
    val_edges = all_edges[n_test:n_test+n_val]
    train_edges = all_edges[n_test+n_val:]
    
    # Create false edges
    def create_false_edges(n):
        false_edges = []
        while len(false_edges) < n:
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i != j and adj[i, j] == 0 and (i, j) not in false_edges:
                false_edges.append((i, j))
        return torch.tensor(false_edges)
    
    test_edges_false = create_false_edges(n_test)
    val_edges_false = create_false_edges(n_val)
    
    # Create train adjacency matrix
    adj_train = adj.clone()
    adj_train[test_edges[:, 0], test_edges[:, 1]] = 0
    adj_train[test_edges[:, 1], test_edges[:, 0]] = 0
    adj_train[val_edges[:, 0], val_edges[:, 1]] = 0
    adj_train[val_edges[:, 1], val_edges[:, 0]] = 0
    
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

# Some preprocessing
def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0))
    rowsum = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

adj_norm = normalize_adj(adj_train)

num_nodes = adj.size(0)
num_features = features.size(1)

# Create Model
pos_weight = float(adj.size(0) * adj.size(0) - adj.sum()) / adj.sum()
norm = adj.size(0) * adj.size(0) / float((adj.size(0) * adj.size(0) - adj.sum()) * 2)

# adj_label = adj_train + torch.eye(adj_train.size(0))
adj_label = (adj_train + torch.eye(adj_train.size(0))).clamp(0, 1)

weight_mask = adj_label.view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)) 
weight_tensor[weight_mask] = pos_weight

# init model and optimizer
input_dim = features.size(1)
if args.model == 'VGAE':
    model = VGAE(input_dim, args.hidden1_dim, args.hidden2_dim, args.dropout)
else:
    model = GAE(input_dim, args.hidden1_dim, args.hidden2_dim, args.dropout)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
# model = getattr(model, args.model)(num_features, args.hidden1_dim, args.hidden2_dim, args.dropout)
# optimizer = Adam(model.parameters(), lr=args.learning_rate)

def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# train model
for epoch in range(args.num_epoch):
    t = time.time()
    # A_pred = model(features, adj_norm)
    
    A_pred = model(features, adj_norm)
    adj_label = adj_label.float()
    A_pred = A_pred.float()

    optimizer.zero_grad()
    assert (adj_label >= 0).all() and (adj_label <= 1).all(), "adj_label contains invalid values"
    assert (A_pred >= 0).all() and (A_pred <= 1).all(), "A_pred contains invalid values"

    loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor)
    # loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor)
    if args.model == 'VGAE':
        kl_divergence = 0.5 / A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label)

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))