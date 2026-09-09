"""
Pipeline de fine-tuning ResNet-50 para classificação de nematoides,
com múltiplas rodadas (seeds diferentes) e agregação de métricas.

Principais mudanças em relação à versão anterior:
  1. Seeds fixadas -> resultados reprodutíveis por rodada.
  2. Múltiplas rodadas (N_RUNS) -> média ± desvio-padrão por métrica,
     em vez de um único número que pode ser sorte/azar de uma rodada.
  3. Dropout antes da camada final -> reduz overfitting.
  4. ColorJitter na augmentation -> mais robustez a variação de
     iluminação (relevante para as imagens reais de celular do projeto).
  5. Label smoothing na loss.
  6. Peso do sampler trocado de 1/count para 1/sqrt(count) -> reduz
     a super-amostragem agressiva das classes raras (Amplimerlinius etc.)
     sem eliminar o balanceamento.

Saídas:
  - runs/run_{seed}_*.csv       -> métricas detalhadas de cada rodada
  - runs/run_{seed}_matriz.png  -> matriz de confusão de cada rodada
  - agregado_metricas_globais.csv     -> média ± desvio das métricas globais
  - agregado_metricas_por_especie.csv -> média ± desvio do F1 por classe
"""

import os
import csv
import random
import statistics as stats

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Pasta base do I-Nema (contém as subpastas train/ train_balanced/ val/)
# Pasta base do I-Nema (contém as subpastas train/ train_balanced/ val/).
# Pode ser sobrescrita sem editar o código, definindo a variável de
# ambiente I_NEMA_DIR antes de rodar -- útil para rodar em outra máquina
# ou compartilhar o repositório sem expor caminhos locais:
#   Windows (PowerShell):  $env:I_NEMA_DIR = "D:\dados\I-Nema"
#   Linux/Mac:              export I_NEMA_DIR="/home/user/dados/I-Nema"
I_NEMA_DIR = os.environ.get(
    "I_NEMA_DIR",
    r"C:\Users\defer\OneDrive\Documentos\IC_nematoides\I-Nema"
)

# Troque para "train_balanced" apenas se quiser comparar contra o
# balanceamento estático antigo -- para o motivo de usar "train"
# (balanceamento dinâmico via sampler + augmentation ao vivo), ver nota abaixo.
TRAIN_SUBDIR = "train"

CONFIG = {
    # IMPORTANTE: por padrão aponta para o dataset ORIGINAL (classes
    # desbalanceadas), não para train_balanced. O balanceamento agora é
    # feito dinamicamente pelo WeightedRandomSampler + augmentation ao vivo
    # (ver setup_data_loaders), o que evita treinar repetidamente sobre
    # cópias idênticas pré-geradas das classes raras (Amplimerlinius,
    # Dorylaimus etc.).
    'train_dir': os.path.join(I_NEMA_DIR, TRAIN_SUBDIR),
    'val_dir': os.path.join(I_NEMA_DIR, "val"),
    'batch_size': 16,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,          # antes 1e-4 -> mais regularização
    'dropout': 0.3,
    'label_smoothing': 0.1,
    'freeze_mode': 'partial',      # 'all', 'partial', 'layer3_4', 'none'
    'sampler_power': 0.5,          # 0.5 = 1/sqrt(count); 1.0 = 1/count (original)
                                    # -> teste com 1.0 piorou o desempenho geral sem
                                    #    melhorar o recall médio das classes raras;
                                    #    voltamos para 0.5 e trocamos a estratégia
                                    #    para Focal Loss (ver 'focal_gamma' abaixo).
    'use_focal_loss': True,
    'focal_gamma': 2.0,            # quanto maior, mais peso pros exemplos difíceis
    'patience_early_stop': 6,
    'patience_scheduler': 2,
}

# Cada seed = uma rodada de treino completa e independente.
# 3 é um mínimo razoável para calcular desvio-padrão; 5 é mais robusto.
# Em CPU, cada rodada pode demorar bastante -- ajuste conforme seu tempo disponível.
SEEDS = [42, 123, 2024]

OUTPUT_DIR = "runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 50 if device.type == 'cuda' else 12


def seed_everything(seed: int):
    """Fixa todas as fontes de aleatoriedade para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET
# =============================================================================

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir)
                                if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def setup_data_loaders(train_dir, val_dir, batch_size, sampler_power, seed):
    train_transform, val_transform = get_transforms()

    train_dataset = CustomDataset(train_dir, transform=train_transform)
    val_dataset = CustomDataset(val_dir, transform=val_transform)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Um dos datasets está vazio!")

    print(f"  Treino: {len(train_dataset)} imagens | Validação: {len(val_dataset)} imagens")

    # Sampler: 1/count**sampler_power em vez de 1/count puro.
    # sampler_power=1.0 reproduz o comportamento original (super-amostragem forte);
    # sampler_power=0.5 (1/sqrt) suaviza o peso das classes raras.
    class_counts = Counter(train_dataset.labels)
    class_weights = {i: 1.0 / (count ** sampler_power) for i, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )

    return train_loader, val_loader, train_dataset.classes


# =============================================================================
# MODELO
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) com suporte a label smoothing.

    Reduz o peso dado a exemplos que o modelo já classifica bem (ex.:
    Pratylenchus, Xenocriconema) e concentra o gradiente nos exemplos
    difíceis (ex.: a confusão intra-Dorylaimida) -- sem precisar mexer
    na frequência de amostragem das classes, evitando o trade-off que
    vimos com sampler_power=1.0 (melhorar raras piorando as fáceis).

    gamma=0 equivale a um CrossEntropy comum (com label smoothing, se houver).
    Valores maiores de gamma aumentam o foco nos exemplos difíceis.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # probabilidade estimada da classe correta
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def create_model_resnet50(num_classes, freeze_mode='partial', dropout=0.3):
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes)
    )
    nn.init.xavier_uniform_(model.fc[1].weight)
    nn.init.zeros_(model.fc[1].bias)

    if freeze_mode == 'all':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif freeze_mode == 'partial':
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    elif freeze_mode == 'layer3_4':
        for name, param in model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    elif freeze_mode == 'none':
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model


# =============================================================================
# TREINO
# =============================================================================

def train_model(model, train_loader, val_loader, num_epochs, learning_rate,
                 weight_decay, label_smoothing, patience_scheduler,
                 patience_early_stop, save_path, use_focal_loss=False, focal_gamma=2.0):
    model = model.to(device)

    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"  Loss: Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"  Loss: CrossEntropy (label_smoothing={label_smoothing})")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                patience=patience_scheduler)

    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # ---- TREINO ----
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc='Treinando', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # ---- VALIDAÇÃO ----
        model.eval()
        val_running_loss, val_running_corrects, val_total_samples = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validando', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f'  TREINO   | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        print(f'  VALIDAÇÃO| Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}')

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_epoch_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f'  ⚡ Learning rate reduzido: {old_lr:.6f} → {new_lr:.6f}')

        if val_epoch_acc > best_acc:
            best_acc, best_loss, patience_counter = val_epoch_acc, val_epoch_loss, 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_acc': best_acc, 'best_loss': best_loss, 'history': history
            }, save_path)
            print(f'  ✓ Novo melhor modelo salvo (acc={best_acc:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= patience_early_stop:
            print(f'  ⚠️  Early stopping após {patience_early_stop} épocas sem melhoria')
            break

    print(f'Treino concluído! Melhor acc de validação: {best_acc:.4f}')
    return model, history


# =============================================================================
# AVALIAÇÃO DETALHADA
# =============================================================================

def evaluate_model_detailed(model, data_loader, device, class_names, prefix):
    model = model.to(device)
    model.eval()

    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    total, correct_top1, correct_top2 = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Avaliando", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            preds_top1 = outputs.argmax(dim=1)
            correct_top1 += (preds_top1 == labels).sum().item()

            k = min(2, outputs.size(1))
            preds_top2 = outputs.topk(k, dim=1).indices
            correct_top2 += (preds_top2 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)
            for true_label, pred_label in zip(labels.cpu(), preds_top1.cpu()):
                confusion[int(true_label), int(pred_label)] += 1

    top1_acc = correct_top1 / total
    top2_acc = correct_top2 / total

    per_class_results = []
    precisions, recalls, f1_scores, supports = [], [], [], []

    for i, class_name in enumerate(class_names):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp
        support = confusion[i, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        per_class_results.append({
            "classe": class_name, "precision": precision, "recall": recall,
            "f1_score": f1, "support": support, "acertos": tp
        })

    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1_scores) / num_classes
    total_support = sum(supports)
    weighted_f1 = (sum(f1 * s for f1, s in zip(f1_scores, supports)) / total_support
                   if total_support > 0 else 0.0)

    print(f"  Top-1: {top1_acc:.4f} | Top-2: {top2_acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")

    # Matriz de confusão normalizada (PNG)
    cm = confusion.float()
    row_sums = cm.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(cm_norm.numpy(), vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"Matriz de Confusão Normalizada - {prefix}")
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_norm[i, j].item()
            if value >= 0.01:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_matriz.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "top1_accuracy": top1_acc, "top2_accuracy": top2_acc,
        "macro_precision": macro_precision, "macro_recall": macro_recall,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "per_class": per_class_results
    }


# =============================================================================
# UMA RODADA COMPLETA
# =============================================================================

def run_single_seed(seed):
    print("\n" + "=" * 70)
    print(f"RODADA COM SEED = {seed}")
    print("=" * 70)

    seed_everything(seed)

    train_loader, val_loader, class_names = setup_data_loaders(
        CONFIG['train_dir'], CONFIG['val_dir'],
        batch_size=CONFIG['batch_size'],
        sampler_power=CONFIG['sampler_power'],
        seed=seed
    )

    model = create_model_resnet50(
        num_classes=len(class_names),
        freeze_mode=CONFIG['freeze_mode'],
        dropout=CONFIG['dropout']
    )

    save_path = os.path.join(OUTPUT_DIR, f"best_model_seed{seed}.pth")
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        label_smoothing=CONFIG['label_smoothing'],
        patience_scheduler=CONFIG['patience_scheduler'],
        patience_early_stop=CONFIG['patience_early_stop'],
        save_path=save_path,
        use_focal_loss=CONFIG['use_focal_loss'],
        focal_gamma=CONFIG['focal_gamma']
    )

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics = evaluate_model_detailed(
        model, val_loader, device, class_names, prefix=f"run_seed{seed}"
    )
    metrics['seed'] = seed
    return metrics, model, val_loader, class_names


# =============================================================================
# ENSEMBLE (média das probabilidades softmax dos modelos já treinados)
# =============================================================================

def evaluate_ensemble(models_list, data_loader, class_names, prefix="ensemble"):
    """
    Mesma lógica de avaliação de evaluate_model_detailed, mas combinando
    N modelos por média simples das probabilidades softmax em vez de usar
    um único modelo. Reduz o ruído de cada seed individual (útil quando os
    modelos discordam bastante entre si -- ver desvio-padrão entre seeds).
    """
    for m in models_list:
        m.eval()

    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total, correct_top1, correct_top2 = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Avaliando ensemble", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            avg_probs = None
            for m in models_list:
                probs = torch.softmax(m(inputs), dim=1)
                avg_probs = probs if avg_probs is None else avg_probs + probs
            avg_probs /= len(models_list)

            preds_top1 = avg_probs.argmax(dim=1)
            correct_top1 += (preds_top1 == labels).sum().item()

            k = min(2, avg_probs.size(1))
            preds_top2 = avg_probs.topk(k, dim=1).indices
            correct_top2 += (preds_top2 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)
            for true_label, pred_label in zip(labels.cpu(), preds_top1.cpu()):
                confusion[int(true_label), int(pred_label)] += 1

    top1_acc = correct_top1 / total
    top2_acc = correct_top2 / total

    per_class_results = []
    precisions, recalls, f1_scores, supports = [], [], [], []

    for i, class_name in enumerate(class_names):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp
        support = confusion[i, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        per_class_results.append({
            "classe": class_name, "precision": precision, "recall": recall,
            "f1_score": f1, "support": support
        })

    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1_scores) / num_classes
    total_support = sum(supports)
    weighted_f1 = (sum(f1 * s for f1, s in zip(f1_scores, supports)) / total_support
                   if total_support > 0 else 0.0)

    print(f"\n  [ENSEMBLE] Top-1: {top1_acc:.4f} | Top-2: {top2_acc:.4f} | "
          f"Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")

    cm = confusion.float()
    row_sums = cm.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(cm_norm.numpy(), vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"Matriz de Confusão Normalizada - {prefix}")
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_norm[i, j].item()
            if value >= 0.01:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_matriz.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "top1_accuracy": top1_acc, "top2_accuracy": top2_acc,
        "macro_precision": macro_precision, "macro_recall": macro_recall,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "per_class": per_class_results
    }


def save_ensemble_csv(results):
    with open("ensemble_metricas_globais.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Metrica", "Resultado"])
        for key in ["top1_accuracy", "top2_accuracy", "macro_precision",
                    "macro_recall", "macro_f1", "weighted_f1"]:
            writer.writerow([key, f"{results[key]:.4f}"])

    with open("ensemble_metricas_por_especie.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Especie", "Precision", "Recall", "F1", "N"])
        for row in results["per_class"]:
            writer.writerow([row["classe"], f"{row['precision']:.4f}",
                              f"{row['recall']:.4f}", f"{row['f1_score']:.4f}",
                              row["support"]])

    print("  Arquivos salvos: ensemble_metricas_globais.csv, ensemble_metricas_por_especie.csv")


# =============================================================================
# AGREGAÇÃO DE MÚLTIPLAS RODADAS
# =============================================================================

def aggregate_runs(all_metrics):
    global_keys = ['top1_accuracy', 'top2_accuracy', 'macro_precision',
                    'macro_recall', 'macro_f1', 'weighted_f1']

    # --- Métricas globais: média ± desvio-padrão ---
    global_file = "agregado_metricas_globais.csv"
    with open(global_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Metrica", "Media", "Desvio_padrao"] +
                         [f"Seed_{m['seed']}" for m in all_metrics])
        for key in global_keys:
            values = [m[key] for m in all_metrics]
            mean = stats.mean(values)
            std = stats.stdev(values) if len(values) > 1 else 0.0
            writer.writerow([key, f"{mean:.4f}", f"{std:.4f}"] +
                             [f"{v:.4f}" for v in values])
            print(f"  {key:20s}: {mean:.4f} ± {std:.4f}")

    # --- Métricas por classe: média ± desvio-padrão do F1 ---
    per_class_file = "agregado_metricas_por_especie.csv"
    class_names = [c['classe'] for c in all_metrics[0]['per_class']]

    with open(per_class_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Especie", "F1_medio", "F1_desvio_padrao", "N_val"] +
                         [f"F1_seed_{m['seed']}" for m in all_metrics])

        for idx, class_name in enumerate(class_names):
            f1_values = [m['per_class'][idx]['f1_score'] for m in all_metrics]
            support = all_metrics[0]['per_class'][idx]['support']
            mean = stats.mean(f1_values)
            std = stats.stdev(f1_values) if len(f1_values) > 1 else 0.0
            writer.writerow([class_name, f"{mean:.4f}", f"{std:.4f}", support] +
                             [f"{v:.4f}" for v in f1_values])

    print(f"\nArquivos agregados salvos: {global_file}, {per_class_file}")


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    print(f"Dispositivo: {device}")
    print(f"Épocas por rodada: {NUM_EPOCHS}")
    print(f"Seeds a rodar: {SEEDS}")

    all_metrics = []
    trained_models = []
    shared_val_loader = None
    shared_class_names = None

    for seed in SEEDS:
        try:
            metrics, model, val_loader, class_names = run_single_seed(seed)
            all_metrics.append(metrics)
            trained_models.append(model)
            # val_loader/class_names são os mesmos em todas as seeds
            # (não há aleatoriedade na validação); guardamos o último para o ensemble.
            shared_val_loader = val_loader
            shared_class_names = class_names
        except Exception as e:
            print(f"\n✗ Erro na rodada seed={seed}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_metrics) >= 1:
        print("\n" + "=" * 70)
        print(f"RESULTADO AGREGADO ({len(all_metrics)} rodada(s))")
        print("=" * 70)
        aggregate_runs(all_metrics)

    if len(trained_models) >= 2:
        print("\n" + "=" * 70)
        print(f"ENSEMBLE ({len(trained_models)} modelos)")
        print("=" * 70)
        ensemble_results = evaluate_ensemble(
            trained_models, shared_val_loader, shared_class_names
        )
        save_ensemble_csv(ensemble_results)
    elif len(trained_models) == 1:
        print("\nApenas 1 modelo treinado com sucesso -- ensemble pulado "
              "(precisa de pelo menos 2 para fazer sentido).")
    else:
        print("\n✗ Nenhuma rodada concluída com sucesso.")
