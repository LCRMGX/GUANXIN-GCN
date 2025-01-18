import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

# Step 1: Load data
train_file_path = 'train_samples（1）.xlsx'
test_file_path = 'test_samples(1).xlsx'
relation_matrix_path = 'relation_matrix.npz'
mapping_file_path = 'index_to_value_mapping.txt'

# Load training and test datasets
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Features to select
features = [
    "SOIL_ID", "NDVI_MEAN", "DEM_ADJ", "ROUGH_MEAN", "SLOPE_MEAN", "SLOPE_VAR",
    "PLANCURV", "POU_WAM", "R_Index", "is_prototype", "similarity",
    "reliability_positive", "reliability_negative"
]
label_col = "label"

# Combine datasets and extract features and labels
combined_data = pd.concat([train_data, test_data], ignore_index=True)
combined_data = combined_data[[
    "SOIL_ID", "NDVI_MEAN", "DEM_ADJ", "ROUGH_MEAN", "SLOPE_MEAN", "SLOPE_VAR",
    "PLANCURV", "POU_WAM", "R_Index", "is_prototype", "similarity",
    "reliability_positive", "reliability_negative", "cat", "value", "label"
]]
x_features = combined_data[features].values
y_labels = combined_data[label_col].values

# Normalize features
x_features = (x_features - x_features.mean(axis=0)) / x_features.std(axis=0)

# Step 2: Load adjacency matrix and mapping
sparse_matrix = load_npz(relation_matrix_path)
relation_matrix = sparse_matrix.toarray()

# Load index-to-value mapping
index_to_value = {}
value_to_index = {}
with open(mapping_file_path, 'r') as f:
    next(f)  # Skip header line
    for line in f:
        index, value = line.strip().split('\t')
        index_to_value[int(index)] = int(value)
        value_to_index[int(value)] = int(index)

# Match nodes using mapping
mapped_indices = [value_to_index.get(int(value), None) for value in combined_data["value"]]
if None in mapped_indices:
    raise ValueError("Some values in the data do not have corresponding indices in the mapping file.")

relation_matrix = relation_matrix[np.ix_(mapped_indices, mapped_indices)]

# Ensure adjacency matrix matches the feature matrix dimensions
num_nodes = x_features.shape[0]
if relation_matrix.shape[0] != num_nodes:
    raise ValueError("The number of nodes in the relation matrix does not match the feature matrix.")

# Convert relation matrix to edge_index format
rows, cols = np.where(relation_matrix > 0)
edge_index = torch.tensor([rows, cols], dtype=torch.long)

# Step 3: Create PyTorch Geometric Data object
x = torch.tensor(x_features, dtype=torch.float)
y = torch.tensor(y_labels, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)

# Split data into training and testing sets
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_indices = range(len(train_data))
test_indices = range(len(train_data), len(combined_data))

train_mask[list(train_indices)] = True
test_mask[list(test_indices)] = True

data.train_mask = train_mask
data.test_mask = test_mask

# Step 4: Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = GCN(input_dim=x.shape[1], hidden_dim=16, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = F.nll_loss

# Step 5: Train the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 6: Test the model with metrics
def test():
    model.eval()
    logits = model(data)
    pred = logits[data.test_mask].argmax(dim=1)
    y_true = data.y[data.test_mask].cpu()
    y_pred = pred.cpu()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, logits[data.test_mask][:, 1].detach().cpu())

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, logits[data.test_mask][:, 1].detach().cpu())
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")

    return acc, precision, recall, f1, auc

# Training loop
for epoch in range(3000):
    loss = train()
    if epoch % 10 == 0:
        acc, precision, recall, f1, auc = test()
        print(
            f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Final test
final_metrics = test()
print(
    f"Final Test Metrics: Accuracy: {final_metrics[0]:.4f}, Precision: {final_metrics[1]:.4f}, Recall: {final_metrics[2]:.4f}, F1: {final_metrics[3]:.4f}, AUC: {final_metrics[4]:.4f}")
