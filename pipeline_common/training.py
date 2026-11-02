"""
The training loop, factored out so Stage 3's k-fold training and any future re-training (e.g.
if Stage 4's occlusion testing ever needs a retrained comparison model) use identical code.
"""

import torch
import torch.nn as nn


def compute_class_weights(df, n_classes=2, label_col="label"):
    counts = df[label_col].value_counts().reindex(range(n_classes), fill_value=0)
    total = counts.sum()
    weights = torch.tensor([total / max(c, 1) for c in counts], dtype=torch.float32)
    return weights / weights.sum() * n_classes   # normalise


def train_model(train_loader, val_loader, model_fn, device, class_weights=None,
                lr=1e-3, weight_decay=1e-4, epochs=60, patience=10,
                label_smoothing=0.05, verbose=True):
    model = model_fn()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * features.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct / total if total else float("nan")
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:03d}/{epochs} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | lr {cur_lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                if verbose:
                    print("  early stopping.")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for features, labels in loader:
        features = features.to(device)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]   # P(MI)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs.cpu().numpy())
    import numpy as np
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)
