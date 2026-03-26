import warnings
warnings.filterwarnings('ignore')
from DensityLoss_v5 import GaussianKDE
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Manifold_Sampling1 import precompute_anchors, batch_pdf_manifold_loss, estimate_manifold_dimension_pytorch
from sklearn.metrics import (
    roc_auc_score,
)
import torch
import Manifold_Sampling1
save_dir = './cifar10_features'
features_ = np.load('./cifar10_features/vit_train_feature.npy')
targets_ = np.load('./cifar10_features/vit_train_labels.npy')

bw = 0.1
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()


classes_kdes = []
anchors = []
anchor_pdfs = []
anchor_tangent_basiss = []
class_manifolds = []

class Cifar10FeatureDataset(Dataset):
    def __init__(self, features, targets, transform=None):
        self.features = features
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]

        if self.transform:
            feature = self.transform(feature)

        return {
            'feature': torch.FloatTensor(feature),
            'target': torch.LongTensor([target])
        }

def generate_dynamic_anchors(model, raw_features, raw_targets, bw, device):
    """
    Dynamically generates KDE manifolds and tangent bases using the model's
    current latent representations.
    """
    print("--> Generating dynamic manifold anchors...")
    model.eval()

    new_anchors = []
    new_anchor_pdfs = []
    new_anchor_tangent_bases = []
    new_class_manifolds = []

    with torch.no_grad():
        for i in range(10):
            # Isolate raw features for the class
            index = raw_targets == i
            xx = raw_features[index]
            samples_image = torch.tensor(xx).to(device)

            # Pass through the CURRENT state of the model
            latent_features = model(samples_image)

            # Build KDE and sample
            KDE = GaussianKDE(X=latent_features, bw=bw)
            samples_num = int(xx.shape[0] * 0.03)

            MANIFOLD_SAMPLING_PARAMS = {
                'K': 30,
                'alpha': 0.5,
                'diversity_factor': 0.9,
            }

            samples = KDE.sample_manifold_aware(
                num_samples=samples_num,
                **MANIFOLD_SAMPLING_PARAMS
            )

            # Compute Geometry
            dim = estimate_manifold_dimension_pytorch(latent_features)
            anchor_pdf, anchor_tangent_basis = precompute_anchors(samples, kde_bw=bw, manifold_dim=dim)

            # Store components
            new_anchors.append(samples)
            new_anchor_pdfs.append(anchor_pdf)
            new_anchor_tangent_bases.append(anchor_tangent_basis)

            manifold_params = {
                "anchors": samples,
                "anchor_pdf": anchor_pdf,
                "anchor_tangent_basis": anchor_tangent_basis
            }
            new_class_manifolds.append(manifold_params)

    return new_anchors, new_anchor_pdfs, new_anchor_tangent_bases, new_class_manifolds

class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256], output_dim=10, dropout=0.3):
        super(EnhancedClassifier, self).__init__()

        # Feature projection layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_encoder = nn.Sequential(*layers)

        # Final projection to target dimension
        self.final_proj = nn.Linear(prev_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        features = self.feature_encoder(x)
        features = self.final_proj(features)
        features = self.layer_norm(features)
        return features

model = EnhancedClassifier(output_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1)
scheduler = optim.lr_scheduler.StepLR(optimizer,  step_size=50, gamma=0.1)
dataset_train = Cifar10FeatureDataset(features_, targets_)
train_size = int(0.8 * len(dataset_train))
val_size = len(dataset_train) - train_size
dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size])
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, raw_features, raw_targets, bw,
                num_epochs=20):
    best_val_acc = 0.0
    accumulation_steps = 1

    # 1. Generate INITIAL anchors (Epoch 0)
    anchors, anchor_pdfs, anchor_tangent_basiss, class_manifolds = generate_dynamic_anchors(
        model, raw_features, raw_targets, bw, device
    )
    classifier = Manifold_Sampling1.EnsembleManifoldClassifier(class_manifolds=class_manifolds)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 2. THE ANCHOR RE-CAST (Epoch 5)
        # Once the model learns semantic separation, we lock in the permanent geometry.
        # if epoch == 15:
        #     print("\n[!] Epoch 5 reached: Re-casting anchors to learned latent semantic centers...\n")
        #     anchors, anchor_pdfs, anchor_tangent_basiss, class_manifolds = generate_dynamic_anchors(
        #         model, raw_features, raw_targets, bw, device
        #     )
        #     # Re-initialize the classifier with the new high-quality manifolds
        #     classifier = Manifold_Sampling1.EnsembleManifoldClassifier(class_manifolds=class_manifolds)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, inputs in enumerate(train_loader):
            features = inputs['feature'].to(device)
            label = inputs['target'].squeeze().to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            cfeatures = model(features)
            loss = 0

            for class_idx in range(10):
                index = (label == class_idx)
                class_features = cfeatures[index]

                if class_features.shape[0] < 3:
                    continue

                loss += Manifold_Sampling1.improved_batch_pdf_manifold_loss(
                    samples=class_features,
                    anchors=anchors[class_idx],
                    anchor_pdf=anchor_pdfs[class_idx],
                    anchor_tangent_basis=anchor_tangent_basiss[class_idx],
                    alpha=50.0,
                    beta=1.0
                )

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pred_labels, _ = classifier.predict(cfeatures)
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(pred_labels == label).item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset) * 100
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%')

        # Validation phase
        model.eval()
        val_running_corrects = 0
        with torch.no_grad():
            for inputs in val_loader:
                features = inputs['feature'].to(device)
                targets = inputs['target'].squeeze().to(device)

                outputs = model(features)
                pred_labels, _ = classifier.predict(outputs)
                val_running_corrects += torch.sum(pred_labels == targets).item()

        val_epoch_acc = val_running_corrects / len(val_loader.dataset) * 100
        print(f'Val Acc: {val_epoch_acc:.2f}%')

        if val_epoch_acc > best_val_acc and epoch > 5:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), './best_model.pth')
            # (Keep your feature extraction saving code here)

    print('Training complete!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

def evaluate_mia(model, train_loader, val_loader, device):
    """
    Performs a confidence-based Membership Inference Attack.
    Updated to handle dictionary-based DataLoaders.
    """
    model.eval()
    member_confidences = []
    non_member_confidences = []

    with torch.no_grad():
        # Members (Train)
        for batch in train_loader:
            # Extract the tensor using the dictionary key
            inputs = batch['feature'].to(device)

            probs = F.softmax(model(inputs), dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            member_confidences.extend(max_probs.cpu().numpy())

        # Non-members (Val)
        for batch in val_loader:
            # Extract the tensor using the dictionary key
            inputs = batch['feature'].to(device)

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

    # 4. Empirical Epsilon (Standard Bound)
    non_zero_fpr = fpr > 0
    safe_tpr = np.clip(tpr[non_zero_fpr], 1e-10, 1.0)
    safe_fpr = fpr[non_zero_fpr]
    empirical_eps_standard = max(0.0, np.max(np.log(safe_tpr / safe_fpr)))

    # 5. Empirical Epsilon (Regression Mapping - Update with your slope/intercept)
    m_slope = 2.5
    b_intercept = -1.1
    empirical_eps_regression = max(0.0, (m_slope * auc) + b_intercept)

    return auc, tpr_at_0_1_fpr, privacy_leakage, empirical_eps_standard, empirical_eps_regression


if __name__ == '__main__':
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        raw_features=features_,  # Pass the raw numpy array here
        raw_targets=targets_,  # Pass the raw numpy targets here
        bw=bw,
        num_epochs=20
    )
    # Create a strict 1:1 Balanced Split for the MIA Audit (e.g., 10,000 vs 10,000)
    # Assuming dataset_train is 40k and dataset_val is 10k
    audit_sample_size = len(dataset_val)  # Typically 10,000 for CIFAR-10

    # Randomly downsample the training set to match the exact size of the validation set
    indices = torch.randperm(len(dataset_train))[:audit_sample_size]
    balanced_member_dataset = torch.utils.data.Subset(dataset_train, indices)

    # Create the specific loaders for the privacy audit
    mia_member_loader = DataLoader(balanced_member_dataset, batch_size=batch_size, shuffle=False)
    mia_non_member_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Run the evaluation on the strictly balanced sets
    auc, tpr_01, leakage, eps_std, eps_reg = evaluate_mia(model, mia_member_loader, mia_non_member_loader, device)
    print("\n=== Privacy Metrics ===")
    print(f"MIA AUC:             {auc:.4f}")
    print(f"TPR @ 0.1% FPR:      {tpr_01:.4f}")
    print(f"Privacy Leakage:     {leakage:.4f}")
    print(f"Empirical Epsilon (Standard Bound): {eps_std:.4f}")
    # print(f"Empirical Epsilon (Your Regression): {eps_reg:.4f}") # Uncomment if using your exact Opacus mapping


# === RelaxLoss Privacy Metrics ===
# MIA AUC:             0.5057
# TPR @ 0.1% FPR:      0.0007
# Privacy Leakage:     0.0114
# Empirical Epsilon (Standard Bound): 0.6931

# Best validation accuracy: 94.66%
#
# === RelaxLoss Privacy Metrics ===
# MIA AUC:             0.5003
# TPR @ 0.1% FPR:      0.0010
# Privacy Leakage:     0.0006
# Empirical Epsilon (Standard Bound): 0.4700

# Val Acc: 94.15%
# Training complete!
# Best validation accuracy: 94.29%
#
# === RelaxLoss Privacy Metrics ===
# MIA AUC:             0.5047
# TPR @ 0.1% FPR:      0.0006
# Privacy Leakage:     0.0094
# Empirical Epsilon (Standard Bound): 0.2183

# Best validation accuracy: 94.91%
#
# === RelaxLoss Privacy Metrics ===
# MIA AUC:             0.5005
# TPR @ 0.1% FPR:      0.0008
# Privacy Leakage:     0.0010
# Empirical Epsilon (Standard Bound): 1.0986