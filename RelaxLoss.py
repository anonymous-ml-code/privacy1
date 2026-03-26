import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# ==========================================
# 1. Dataset Definition
# ==========================================
class Cifar10FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# ==========================================
# 2. RelaxLoss Definition
# ==========================================
class RelaxLoss(nn.Module):
    def __init__(self, alpha=0.5, smoothing=0.2):
        super(RelaxLoss, self).__init__()
        self.alpha = alpha
        self.smoothing = smoothing
        
    def forward(self, logits, targets, epoch):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        soft_loss = F.cross_entropy(logits, targets, label_smoothing=self.smoothing, reduction='none')
        
        relaxed_mask = ce_loss < self.alpha
        final_loss = ce_loss.clone()
        
        if relaxed_mask.any():
            if epoch % 2 == 0:
                final_loss[relaxed_mask] = self.alpha - ce_loss[relaxed_mask]
            else:
                final_loss[relaxed_mask] = soft_loss[relaxed_mask]
                
        return final_loss.mean()

# ==========================================
# 3. Feature Classification Model (MLP)
# ==========================================
class FeatureClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=10):
        super(FeatureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. MIA Evaluation Pipeline (Updated)
# ==========================================
def evaluate_mia(model, train_loader, val_loader, device):
    """
    Performs a confidence-based Membership Inference Attack and extracts
    AUC, TPR@0.1%FPR, Privacy Leakage, and Empirical Epsilon.
    """
    model.eval()
    member_confidences = []
    non_member_confidences = []

    with torch.no_grad():
        # Members (Train)
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            probs = F.softmax(model(inputs), dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            member_confidences.extend(max_probs.cpu().numpy())

        # Non-members (Val)
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            probs = F.softmax(model(inputs), dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            non_member_confidences.extend(max_probs.cpu().numpy())

    y_true = np.concatenate([np.ones(len(member_confidences)), np.zeros(len(non_member_confidences))])
    y_scores = np.concatenate([member_confidences, non_member_confidences])

    # 1. ROC and AUC
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # 2. TPR @ 0.1% FPR (0.001)
    valid_fpr_indices_01 = np.where(fpr <= 0.001)[0]
    tpr_at_0_1_fpr = tpr[valid_fpr_indices_01[-1]] if len(valid_fpr_indices_01) > 0 else 0.0
    
    # 3. Privacy Leakage
    privacy_leakage = 2 * (auc - 0.5)

    # 4. Empirical Epsilon (Two Methods)
    
    # Method A: Standard strict lower-bound: max(ln(TPR/FPR))
    non_zero_fpr = fpr > 0
    safe_tpr = np.clip(tpr[non_zero_fpr], 1e-10, 1.0) # Avoid log(0)
    safe_fpr = fpr[non_zero_fpr]
    empirical_eps_standard = max(0.0, np.max(np.log(safe_tpr / safe_fpr)))
    
    # Method B: Your Opacus Linear Regression mapping (from Appendix)
    # Replace slope (m) and intercept (b) with your exact calibration numbers
    m_slope = 2.5      # Placeholder: Replace with your regression slope
    b_intercept = -1.1 # Placeholder: Replace with your regression intercept
    empirical_eps_regression = max(0.0, (m_slope * auc) + b_intercept)

    return auc, tpr_at_0_1_fpr, privacy_leakage, empirical_eps_standard, empirical_eps_regression

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Features
    print("Loading features...")
    features_ = np.load('./cifar10_features/vit_train_feature.npy')
    targets_ = np.load('./cifar10_features/vit_train_labels.npy')
    
    feature_dim = features_.shape[1] 
    full_dataset = Cifar10FeatureDataset(features_, targets_)
    
    # Split Data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    dataset_train, dataset_val = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    batch_size = 1024
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialize Model
    model = FeatureClassifier(input_dim=feature_dim, num_classes=10).to(device)
    criterion = RelaxLoss(alpha=0.5, smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training Loop
    epochs = 100 
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets, epoch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            train_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    # Final Utility Evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    
    val_acc = 100. * val_correct / val_total
    print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")

    # Privacy Evaluation
    print("\nRunning Membership Inference Attack Evaluation...")
    auc, tpr_01, leakage, eps_std, eps_reg = evaluate_mia(model, train_loader, val_loader, device)
    
    print("\n=== RelaxLoss Privacy Metrics ===")
    print(f"MIA AUC:             {auc:.4f}")
    print(f"TPR @ 0.1% FPR:      {tpr_01:.4f}")
    print(f"Privacy Leakage:     {leakage:.4f}")
    print(f"Empirical Epsilon (Standard Bound): {eps_std:.4f}")
    # print(f"Empirical Epsilon (Your Regression): {eps_reg:.4f}") # Uncomment if using your exact Opacus mapping

if __name__ == '__main__':
    main()

    # == = RelaxLoss
    # Privacy
    # Metrics == =
    # MIA
    # AUC: 0.5397
    # TPR @ 0.1 % FPR: 0.0006
    # Privacy
    # Leakage: 0.0794
    # Empirical
    # Epsilon(Standard
    # Bound): 0.8109

    # == = RelaxLoss
    # Privacy
    # Metrics == =
    # MIA
    # AUC: 0.5343
    # TPR @ 0.1 % FPR: 0.0006
    # Privacy
    # Leakage: 0.0686
    # Empirical
    # Epsilon(Standard
    # Bound): 0.1507