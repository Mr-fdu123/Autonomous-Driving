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
# 2. 定义 LSTM 模型
# =========================
# class LSTMPredictor(nn.Module):
#     def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         output, (h, c) = self.lstm(x)
#         output = output[:, -1, :]
#         y_pred = self.fc(output)
#         return y_pred

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 遗忘门f_t
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        # 输入门i_t
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        # 输出门o_t
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        # 候选记忆g_t
        self.W_g = nn.Linear(input_dim, hidden_dim)
        self.U_g = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x_t, h_pre, c_pre):
        # x_t (B, D) h_pre (B, H) c_pre (B, H)
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_pre))
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_pre))
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_pre))
        g_t = torch.tanh(self.W_g(x_t) + self.U_g(h_pre))

        c_t = f_t * c_pre + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LSTMCell(input_dim, hidden_dim))
            else:
                self.layers.append(LSTMCell(hidden_dim, hidden_dim))
    
    def forward(self, x, h_0=None, c_0=None):
        # x (B, T, D) h_0,c_0 (num_layers, B, H) 
        B, T, D = x.shape
        if h_0 is None:
            h_t = [torch.zeros(B, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        else:
            h_t = h_0
        if c_0 is None:
            c_t = [torch.zeros(B, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        else:
            c_t = c_0
        output = []
        
        for t in range(T):
            x_t = x[:, t, :] # (B, D)
            for i in range(self.num_layers):
                h_t[i], c_t[i] = self.layers[i](x_t, h_t[i], c_t[i])
                x_t = h_t[i]
            output.append(x_t.unsqueeze(1))
        
        output = torch.cat(output, dim=1)
        return output, (h_t, c_t)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = MultiLayerLSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        y_pred = self.fc(out) # (B, 1)
        return y_pred


# =========================
# 3. 训练准备
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
train_dataset = SineDataset(num_samples=1000, seq_len=20)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = LSTMPredictor().to(device=device)
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






