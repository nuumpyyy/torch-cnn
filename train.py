from torch import nn
from torch import optim
from tqdm import tqdm
from data import train_loader
from cnn import device, model

# Define loss function
criterion = nn.CrossEntropyLoss()

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Iterate over 10 epochs
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        # clear gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()