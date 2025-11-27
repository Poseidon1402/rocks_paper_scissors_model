import torch
from tqdm.auto import tqdm
import torch.nn as nn

from models.model import Model
from data.datamodule import train_loader, val_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(DEVICE)

# Training configuration
EPOCHS = 30
LEARNING_RATE = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
BEST_MODEL_PATH = "models/model.pt"

for epoch in range(1, EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} / {EPOCHS} [Train]", leave=False)
    
    for images, labels in train_pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total

    # ---- VALIDATION ----
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for images, labels in val_pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    # ---- Save the best model (lowest validation loss) ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        saved_flag = " [saved best]"
    else:
        saved_flag = ""
    
    print(
        f"Epoch {epoch:02d} / {EPOCHS} "
        f"- train loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
        f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} {saved_flag}"
    )

print(f"\nBest validation loss: {best_val_loss:.4f}")