import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")


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
        
        label_counts = Counter(self.labels)
        print(f"\nDataset: {len(self.image_paths)} imagens em {len(self.classes)} classes")
        print("Distribuição de classes:")
        for cls_name, cls_idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            print(f"  {cls_name}: {label_counts.get(cls_idx, 0)} imagens")
    
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

def get_transforms_optimized():

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
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

def setup_data_loaders_weighted(train_dir, val_dir, batch_size=32):
    
    print("\n Carregando datasets")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    
    train_transform, val_transform = get_transforms_optimized()
    
    train_dataset = CustomDataset(train_dir, transform=train_transform)
    val_dataset = CustomDataset(val_dir, transform=val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Um dos datasets está vazio!")
    
    print(f"\n Dados carregados:")
    print(f"  Treino: {len(train_dataset)} imagens")
    print(f"  Validação: {len(val_dataset)} imagens")
    
    # Weighted sampler para balanceamento
    class_counts = Counter(train_dataset.labels)
    class_weights = {i: 1.0 / count for i, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, train_dataset.classes

# CRIAR MODELO OTIMIZADO (ResNet-50)

def create_model_resnet50(num_classes, freeze_mode='partial'):
    """
    Cria ResNet-50
    freeze_mode: 'all', 'partial', 'layer3_4', 'none'
    """
    print(f"\n🔧 Criando ResNet-50...")
    
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet50(pretrained=True)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    
    # Congelar camadas
    if freeze_mode == 'all':
        print("Modo: Congelando TODAS as camadas exceto FC")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif freeze_mode == 'partial':
        print("Modo: Congelando até layer3 (descongelando layer4 + FC)")
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
            
    elif freeze_mode == 'layer3_4':
        print("Modo: Congelando até layer2 (descongelando layer3 + layer4 + FC)")
        for name, param in model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False
            
    elif freeze_mode == 'none':
        print("Modo: Treinando TODAS as camadas")
        for param in model.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model


def train_model_optimized(model, train_loader, val_loader, num_epochs=30, 
                         learning_rate=1e-3, patience=10, save_path='best_model.pth'):
    """Treino otimizado com ReduceLROnPlateau e early stopping"""
    
    model = model.to(device)
    
    # Loss, optimizer e scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                               patience=2)
    
    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    print(f"\n Iniciando treino por {num_epochs} épocas\n")
    
    for epoch in range(num_epochs):
        print(f'{"="*70}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'{"="*70}')
        
        # ---- TREINO ----
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc='Treinando')
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
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
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # ---- VALIDAÇÃO ----
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validando')
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)
                
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print métricas
        print(f'\nTREINO   | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        print(f'VALIDAÇÃO| Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Atualizar scheduler baseado em acurácia
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_epoch_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f'⚡ Learning rate reduzido: {old_lr:.6f} → {new_lr:.6f}')
        
        # Salvar melhor modelo
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_loss = val_epoch_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_loss': best_loss,
                'history': history
            }, save_path)
            print(f'✓ Modelo salvo! Nova melhor acurácia: {best_acc:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\n⚠️  Early stopping após {patience} épocas sem melhoria')
            break
    
    print(f'\n{"="*70}')
    print(f'Treino concluído!')
    print(f'Melhor acurácia de validação: {best_acc:.4f}')
    print(f'Melhor loss de validação: {best_loss:.4f}')
    print(f'{"="*70}')
    
    return model, history

def extract_features(model, data_loader, device):
    """Extrai features da penúltima camada"""
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, lbls in tqdm(data_loader, desc="Extraindo features"):
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu())
            labels.append(lbls)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    
    return features, labels

def predict_by_cosine_similarity(model, image_path, banco_features, banco_labels, 
                                 class_names, transform, device):
    """Predição usando similaridade de cosseno"""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Erro ao carregar {image_path}: {e}")
        return None, None
    
    # Extrair feature da imagem teste
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        test_feature = feature_extractor(image_tensor)
        test_feature = test_feature.view(test_feature.size(0), -1)
        test_feature = test_feature.cpu()
    
    # Calcular similaridade
    similarities = torch.nn.functional.cosine_similarity(test_feature, banco_features)
    idx_max = torch.argmax(similarities).item()
    
    predicted_class = class_names[banco_labels[idx_max]]
    score = similarities[idx_max].item()
    
    print(f"\nImagem: {os.path.basename(image_path)}")
    print(f"Classe prevista: {predicted_class}")
    print(f"Similaridade: {score:.4f}")
    
    return predicted_class, score

def complete_training_pipeline():
    """Pipeline completo otimizado"""
    
    CONFIG = {
        'train_dir': "C:\\Users\\\Adalto.S\\Documents\\I-Nema\\train_balanced",
        'val_dir': "C:\\Users\\Adalto.S\\Documents\\I-Nema\\val",
        'num_epochs': 50 if device.type == 'cuda' else 12,
        'batch_size': 32 if device.type == 'cuda' else 16,
        'learning_rate': 3e-4,
        'patience': 6,
        'freeze_mode': 'partial',  # 'all', 'partial', 'layer3_4', 'none'
        'use_cosine_similarity': True,
    }
    
    print("="*70)
    print("CONFIGURAÇÃO DO FINE-TUNING HÍBRIDO")
    print("="*70)
    for k, v in CONFIG.items():
        print(f"{k:25s}: {v}")
    
    if device.type == 'cpu':
        print("\n⚠️  ATENÇÃO: Rodando em CPU - Treinamento será mais lento")
    else:
        print(f"\n✓ Rodando em GPU: {torch.cuda.get_device_name(0)}")
    
    print("="*70)
    
    # 1. Carregar dados
    try:
        train_loader, val_loader, class_names = setup_data_loaders_weighted(
            CONFIG['train_dir'],
            CONFIG['val_dir'],
            batch_size=CONFIG['batch_size']
        )
    except Exception as e:
        print(f"\n❌ Erro ao carregar dados: {e}")
        return None
    
    # 2. Criar modelo ResNet-18
    model = create_model_resnet50(
        num_classes=len(class_names),
        freeze_mode=CONFIG['freeze_mode']
    )
    
    # 3. Treinar
    model, history = train_model_optimized(
        model,
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        patience=CONFIG['patience'],
        save_path='best_model.pth'
    )
    
    # 4. Plotar histórico
    plot_training_history(history)
    
    # 5. Extrair features (opcional, para classificação por similaridade)
    if CONFIG['use_cosine_similarity']:
        print("\n📊 Extraindo features para banco de dados...")
        
        # Carregar melhor modelo
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Combinar train + val
        full_loader = DataLoader(
            ConcatDataset([train_loader.dataset, val_loader.dataset]),
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        features, labels = extract_features(model, full_loader, device)
        
        torch.save({
            'features': features,
            'labels': labels,
            'class_names': class_names
        }, 'feature_bank.pt')
        
        print("✓ Feature bank salvo em 'feature_bank.pt'")
    
    # 6. Salvar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'config': CONFIG
    }, 'modelo_final.pth')
    
    print("\n✅ Modelos salvos:")
    print("  - best_model.pth (melhor validação)")
    print("  - modelo_final.pth (última época)")
    if CONFIG['use_cosine_similarity']:
        print("  - feature_bank.pt (banco de features)")
    
    return model, class_names, history

def plot_training_history(history):
    """Plota curvas de treino"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curva de Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Acurácia
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia')
    axes[1].set_title('Curva de Acurácia')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n📈 Gráficos salvos em 'training_history.png'")
    #plt.show()

###############################################################################
# EXECUÇÃO
###############################################################################

if __name__ == '__main__':
    print("Iniciando fine-tuning híbrido otimizado...")
    try:
        result = complete_training_pipeline()
        if result:
            print("\n✓ Pipeline concluído com sucesso!")
        else:
            print("\n✗ Pipeline falhou.")
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        import traceback
        traceback.print_exc()