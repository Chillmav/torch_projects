import time
import torch
import torch.nn as nn

device = "cpu"

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Sztuczne dane
X = torch.randn(100_000, 512)
y = torch.randint(0, 10, (100_000,))

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True
)

model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

# Synchronizacja jest ważna przy mierzeniu GPU
# if device.type == "cuda":
#     torch.cuda.synchronize()

start = time.perf_counter()

for epoch in range(epochs):
    epoch_start = time.perf_counter()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if device.type == "cuda":
    #     torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"loss={loss.item():.4f} | "
        f"time={epoch_time:.2f}s"
    )

total_time = time.perf_counter() - start

print(f"\nTotal training time: {total_time:.2f}s")