import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)
    def fit(self, data, epochs, train_loader, val_loader):
        device = next(self.parameters()).device
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                out = self(batch.x, batch.edge_index).detach() 
                out.requires_grad = True 
                if hasattr(batch, 'y') and batch.y is not None:
                    loss = self.criterion(out, batch.y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")



