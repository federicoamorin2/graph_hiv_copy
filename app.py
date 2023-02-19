import torch
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import mlflow.pytorch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from src.dataset import CustomDataset
from src.model import GNN, ModelParams


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://localhost:5000")

train_dataset = CustomDataset(root="data/", filename="HIV_train.csv")
test_dataset = CustomDataset(root="data/", filename="HIV_test.csv")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

params = ModelParams(
    embedding_size=1024, 
    attention_heads=3,
    layers=3,
    dropout_rate=0.3,
    top_k_every_n=1,
    top_k_ratio=0.2,
    dense_neurons=128,
    edge_dim=train_dataset.edge_dim
)

model = GNN(feature_size=train_dataset.feature_size, model_params=params)
model.to(device)
print(f"Quantidade de parametros {count_parameters(model)}")
print(model)

weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

NUM_GRAPHS_PER_BATCH = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=NUM_GRAPHS_PER_BATCH,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=NUM_GRAPHS_PER_BATCH,
    shuffle=True
)

def train(epoch):
    all_preds = []
    all_labels = []
    for batch in tqdm(train_loader):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(
            batch.x.float(),
            batch.edge_attr.float(),
            batch.edge_index,
            batch.batch
        )
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())


    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    comupte_metrics(all_preds, all_labels, epoch, "train")
    return loss

def test(epoch):
    all_preds = []
    all_labels = []
    for batch in tqdm(train_loader):
        print("oi")
        batch.to(device)
        pred = model(
            batch.x.float(),
            batch.edge_attr.float(),
            batch.edge_index,
            batch.batch
        )
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())


    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    comupte_metrics(all_preds, all_labels, epoch, "train")
    return loss

def comupte_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")

input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 30), name="x"), 
                       TensorSpec(np.dtype(np.float32), (-1, 11), name="edge_attr"), 
                       TensorSpec(np.dtype(np.int32), (2, -1), name="edge_index"), 
                       TensorSpec(np.dtype(np.int32), (-1, 1), name="batch_index")])

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run():
    for epoch in range(3):
        print("Entrando aqui")
        model.train()
        print(f"Treinando a epoca {epoch}")
        loss = train(epoch)
        print(f"Epoch {epoch} | Train loss {loss}")
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        model.eval()
        if epoch % 2 == 0:
            loss = test(epoch)
            loss = loss.detach().cpu().numpy()
            print(f"Epoch {epoch} | Test loss {loss}")
            mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)            
