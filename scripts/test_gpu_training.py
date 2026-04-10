import torch
import time

# Device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Create dummy data
X = torch.randn(1000, 100).to(device)
y = torch.randn(1000, 1).to(device)

# Simple model
model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 1)
).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

print("\n🚀 Training started...")
start = time.time()

for epoch in range(10):
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

elapsed = time.time() - start
print(f"\n✅ Training completed in {elapsed:.2f}s")