import torch
from torch.optim.lr_scheduler import StepLR
from utilities import BinaryCrossEntropy, Net, prepare_data, test, train

# train
epochs = 2
device = torch.device("cpu")
objective = BinaryCrossEntropy()

# data preparation
train_loader, test_loader = prepare_data(
    objective=objective,
    positive_classes=0,
    train_batch_size=512,
    test_batch_size=1000,
)

# model preparation
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.8)


for epoch in range(1, epochs + 1):
    train(model, objective, device, train_loader, optimizer, epoch)
    test(model, objective, device, test_loader)
    scheduler.step()
