import torch
import torch.nn as nn

class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=3, out_features=5)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

model = HousePriceModel()

import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Dữ liệu huấn luyện (diện tích, số phòng ngủ, tuổi nhà) và giá nhà thực tế
x_input = torch.tensor([[1200.0, 3.0, 10.0], [1200.0, 3.0, 5.0], [1500.0, 4.0, 5.0], [1300.0, 3.0, 10.0]], dtype=torch.float32)

y_true = torch.tensor([[5.0], [3.0], [6.0], [4.0]], dtype=torch.float32)
epochs = 1000          
for epoch in range(epochs):
    # print("Trọng số (Weights) hiện tại:\n", model.hidden_layer.weight)
    # print("Độ lệch (Bias) hiện tại:\n", model.hidden_layer.bias)
    optimizer.zero_grad()
    predicted_price = model(x_input)
    loss = criterion(predicted_price, y_true)
    loss.backward()
    # print("\nĐạo hàm của Trọng số (Weight Gradients):\n", model.hidden_layer.weight.grad)
    # print("Đạo hàm của Độ lệch (Bias Gradients):\n", model.hidden_layer.bias.grad)
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

x_val = torch.tensor([[1400.0, 3.0, 7.0]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    predicted_price = model(x_val)
    print("\nGiá nhà dự đoán:\n", predicted_price)
