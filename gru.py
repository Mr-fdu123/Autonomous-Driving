import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# =========================
# 1. 构造数据集
# =========================
class SineDataset(Dataset):
    def __init__(self, num_samples=500, seq_len=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.x_data = []
        self.y_data = []

        for _ in range(self.num_samples):
            step = 0.1
            start = np.random.uniform(0, 2 * np.pi)
            seq = np.array([np.sin(start + i*step) for i in range(self.seq_len+1)], dtype=np.float32)
            x_seq = seq[:seq_len]
            y_seq = seq[seq_len]

            self.x_data.append(x_seq.reshape(seq_len, 1)) # [T, D=1]
            self.y_data.append(np.array([y_seq], dtype=np.float32))  # (1,)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x_tentor = torch.tensor(self.x_data[idx])
        y_tentor = torch.tensor(self.y_data[idx]) 
        return x_tentor, y_tentor


# =========================
# 2. 定义 GRU 模型
# =========================
# class GRUPredictor(nn.Module):
#     def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_szie=1):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, output_szie)
    
#     def forward(self, x): # [B, T, D]
#         out, _ = self.gru(x) # out [B, T, H]
#         out_last = out[:, -1, :] # [B, H]
#         y_pred = self.fc(out_last) # [B, 1]
#         return y_pred

class MyGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # update z_t
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # update r_t
        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # candicate h_t
        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x_t, h_pre):
        # x_t [B, input] h_pre [B, hidden]
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_pre))
        r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h_pre))

        h_can = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h_pre))
        h_t = h_can * z_t + h_pre * (1-z_t)

        return h_t
    
# 多层GRU
class MyMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers =  num_layers
        self.hidden_dim = hidden_dim
    
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(MyGRUCell(input_dim, hidden_dim))
            else:
                self.layers.append(MyGRUCell(hidden_dim, hidden_dim))
    
    def forward(self, x, h_0=None):
        B, T, D = x.shape
        # 每层初始化 h_t [num_layers, B, H]
        if h_0 is None:
            h_t = [torch.zeros(B, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        else:
            h_t = [h_0[i] for i in range(self.num_layers)]

        out = []
        for t in range(T):
            # x_t 每一层的输入，上一层输出作为下一层输入
            x_t = x[:, t, :] # [B, D]
            for i in range(self.num_layers):
                h_t[i] = self.layers[i](x_t, h_t[i])
                x_t = h_t[i]
            out.append(x_t.unsqueeze(1)) # [B, 1, H]

        output = torch.cat(out, dim=1)
        return output, x_t
        


# class GRUModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.cell = MyGRUCell(input_dim, hidden_dim)
    
#     def forward(self, x, h0=None):
#         # x [B, T, input] 
#         B, T, D = x.shape
#         out = []
#         if h0 is None:
#             h_t = torch.zeros(B, self.hidden_dim, device=x.device)
#         else:
#             h_t = h0
#         for t in range(T):
#             x_t = x[:, t, :]
#             h_t = self.cell(x_t, h_t) # [B, H]
#             out.append(h_t.unsqueeze(1)) # [B, 1, H]
        
#         out = torch.cat(out, dim=1) # [B, T, H]
#         return out, h_t

class GRUPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.gru = MyMultiLayerGRU(
            input_dim=input_size,
            hidden_dim=hidden_size,
            num_layers=num_layers
            )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x [B, T, D]
        output, h = self.gru(x)
        out = output[:, -1, :] # [B, H]
        y_pred = self.fc(out) # [B, 1]
        return y_pred


# =========================
# 3. 训练准备
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
train_dataset = SineDataset(num_samples=1000, seq_len=20)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = GRUPredictor().to(device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


# =========================
# 4. 训练
# =========================
epoch_size = 40

for epoch in range(epoch_size):
    total_loss = 0.0
    model.train()

    for x, y in train_loader:
        x = x.to(device) # [B, T, D]
        y = y.to(device) # [B, 1]
        y_pred = model(x) # [B, 1]

        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{epoch_size}], Loss: {avg_loss:.6f}")

# =========================
# 5. 测试一个样本
# =========================
model.eval()
with torch.no_grad():
    x_t, y_t = train_dataset[10]
    x_t = x_t.unsqueeze(0).to(device=device) # [1, T, D]
    y_p = model(x_t)
    print("Ground Truth:", y_t.item())
    print("Prediction  :", y_p.item())






