
#-------------------------------------------------
# GrabPhisher Model Code for Phishing Detection
#-------------------------------------------------

# Imports requiring packages
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.console import Console
from rich import box
from tqdm import trange
from sklearn.metrics import confusion_matrix, classification_report

console = Console()

# Data Loader
df = pd.read_csv('/content/drive/MyDrive/GrabPhisher_Model/final_data_6.csv')
console.print("[bold yellow]\U0001F517 Loading the Data...")
print(df.head())
print(df.columns)

import networkx as nx

# Initialize directed graph
G = nx.DiGraph()

# Add nodes and edges from the dataframe
for idx, row in df.iterrows():
    src = row['from']
    tgt = row['to']
    ts = row['timestamp']
    amt = row['amount']
    label_src = row['fromIsPhi']
    label_tgt = row['toIsPhi']

    G.add_node(src, label=label_src)
    G.add_node(tgt, label=label_tgt)
    G.add_edge(src, tgt, timestamp=ts, amount=amt)

# Print basic graph summary (manual replacement for nx.info)
console.print("\n\n[bold yellow]\u2705 Graph Summary:")
print(f"Type: Directed Graph")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Optionally: check a few nodes
console.print("\n\n[bold green]\u2705 Sample nodes with attributes:")
for node, attr in list(G.nodes(data=True))[:5]:
    print(f"{node}: {attr}")


G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
G = nx.relabel_nodes(G, lambda x: str(x))  # Node2Vec needs string IDs

import os
nodes = sorted(G.nodes(), key=lambda x: int(x))  # ensure correct order

if os.path.exists("n2v_embeddings.npy"):
    embeddings = np.load("n2v_embeddings.npy")
    console.print("\n\n[bold yellow]\u2705 Loaded embeddings from file:", embeddings.shape)
else:
    print("\nGenerating Node2Vec embeddings...\n")
    node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=20, workers=4, seed=42)
    n2v_model = node2vec.fit(window=5, min_count=1, batch_words=128)
    embeddings = np.array([n2v_model.wv[node] for node in nodes])
    np.save("n2v_embeddings.npy", embeddings)
    print("\nGenerated embeddings:\n", embeddings.shape)

x = torch.tensor(embeddings, dtype=torch.float)


from sklearn.utils.class_weight import compute_class_weight

node_mapping = {node: i for i, node in enumerate(G.nodes())}

edges = [(node_mapping[src], node_mapping[tgt]) for src, tgt in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

timestamps = [G.edges[e]['timestamp'] for e in G.edges()]
amounts = [G.edges[e]['amount'] for e in G.edges()]
edge_attr = torch.tensor(np.vstack([timestamps, amounts]).T, dtype=torch.float)

# Example: replace with your own node embeddings
embeddings = np.random.rand(len(G.nodes()), 128)  # Replace with actual node features
x = torch.tensor(embeddings, dtype=torch.float)

labels = [G.nodes[node]['label'] for node in G.nodes()]
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Print summary
print(data)

train_idx, test_idx = train_test_split(
    np.arange(y.shape[0]), test_size=0.2, stratify=y, random_state=42
)

train_mask = torch.zeros(y.shape[0], dtype=torch.bool)
test_mask = torch.zeros(y.shape[0], dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask



class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGCN(x.shape[1], 64, 32, 2).to(device)  # reduced dims
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# -----------------------
# TRAINING
# -----------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)

    labels_np = data.y[data.train_mask].cpu().numpy()
    class_sample_count = np.bincount(labels_np)
    from sklearn.utils.class_weight import compute_class_weight

    # Compute class weights once outside the training loop
    labels_np = data.y[data.train_mask].cpu().numpy()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_np), y=labels_np)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)



    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=weights)
    loss.backward()
    optimizer.step()
    return loss.item()


# -----------------------
# TESTING
# -----------------------
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    return pred



for epoch in trange(1, 51, desc="Training Epochs"):
    loss = train()
    if epoch % 5 == 0:
        console.log(f"[bold cyan]Epoch {epoch}[/] - Loss: {loss:.4f}")

pred = test()




# True labels and predictions
true = data.y[data.test_mask].cpu().numpy()
predicted = pred[data.test_mask].cpu().numpy()

# Confusion matrix

cm = confusion_matrix(true, predicted)
labels = ['Normal', 'Phishing']
report_dict = classification_report(true, predicted, target_names=labels, output_dict=True)
report_str = classification_report(true, predicted, target_names=labels, digits=4)

# Pretty Confusion Matrix
conf_table = Table(title="\nConfusion Matrix", show_lines=True)
conf_table.add_column("", style="bold magenta")
conf_table.add_column("Predicted: Normal", justify="center")
conf_table.add_column("Predicted: Phishing", justify="center")
conf_table.add_row("Actual: Normal", str(cm[0][0]), str(cm[0][1]))
conf_table.add_row("Actual: Phishing", str(cm[1][0]), str(cm[1][1]))
console.print(conf_table)

# Pretty Classification Report
class_report_table = Table(title="\nClassification Report", show_lines=True)
class_report_table.add_column("Class", justify="left", style="bold green")
class_report_table.add_column("Precision")
class_report_table.add_column("Recall")
class_report_table.add_column("F1-Score")
class_report_table.add_column("Support")

for label in labels:
    metrics = report_dict[label]
    class_report_table.add_row(label,
        f"{metrics['precision']:.4f}",
        f"{metrics['recall']:.4f}",
        f"{metrics['f1-score']:.4f}",
        f"{int(metrics['support'])}")

# Add accuracy, macro avg, and weighted avg
for avg in ["accuracy", "macro avg", "weighted avg"]:
    if avg == "accuracy":
        class_report_table.add_row(avg, "-", "-", f"{report_dict[avg]:.4f}", f"{sum(cm[0]) + sum(cm[1])}")
    else:
        avg_data = report_dict[avg]
        class_report_table.add_row(avg,
            f"{avg_data['precision']:.4f}",
            f"{avg_data['recall']:.4f}",
            f"{avg_data['f1-score']:.4f}",
            f"{int(avg_data['support'])}")

console.print(class_report_table)

# ---------------- Custom Summary ------------------
total_test = len(true)
total_normal = sum(true == 0)
total_phishing = sum(true == 1)
normal_correct = cm[0][0]
phishing_correct = cm[1][1]

console.print("\n\n[bold green]\u2705 Test Summary:")
console.print(f"Total nodes in test data: {total_test}")
console.print(f"Total NORMAL nodes (true): {total_normal}")
console.print(f"Total PHISHING nodes (true): {total_phishing}")
console.print(f"Normal correctly classified: {normal_correct} / {total_normal} = {100*normal_correct/total_normal:.2f}%")
console.print(f"Phishing correctly classified: {phishing_correct} / {total_phishing} = {100*phishing_correct/total_phishing:.2f}%")

# ---------------- Performance Comparison Table ------------------

grab_precision = f"{report_dict['Phishing']['precision']:.2f}"
grab_recall = f"{report_dict['Phishing']['recall']:.2f}"
grab_f1 = f"{report_dict['Phishing']['f1-score']:.2f}"

comparison_data = [
    ["Method", "Precision", "Recall", "F1-Score"],
    ["Node2vec", "0.62", "0.65", "0.63"],
    ["WL-kernel", "0.63", "0.67", "0.65"],
    ["GCN", "0.52", "0.55", "0.53"],
    ["Graph2vec", "0.70", "0.75", "0.73"],
    ["TGAT", "0.30", "0.40", "0.35"],
    ["TGN", "0.55", "0.60", "0.57"],
    ["EvolveGCN", "0.68", "0.70", "0.69"],
    ["GrabPhisher", grab_precision, grab_recall, grab_f1]
]

console.print("\n\n[bold yellow]\U0001F517 Performance Comparison Table:")
console.print(tabulate(comparison_data, headers="firstrow", tablefmt="fancy_grid"))
