import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models, transforms, datasets
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shutil
import json
from pathlib import Path
from contextlib import nullcontext
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from tqdm import tqdm
import glob
from torchvision.transforms.functional import to_pil_image
import random
import pandas as pd

# CONFIG
# Settings - Real / augmented / Synthetic / SyntheticAugmented 
# Dataset - real: MMAFEDB Synthetic - Self-generated FEDB
DATASET_TYPE = "SyntheticAugmented"  
DATA_DIR = "C:/Users/user/Desktop/COMP7250_ML_MiniProject/SelfSyntheticFEDB/Augmented"
INPUT_SIZE = 256  # Image resize
FEATURE_EXTRACT = False
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# training setup
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER_TYPE = "sgd"  # (SGD or Adam)
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 25
WORKERS = 2  
PIN_MEMORY = True
USE_MIXED_PRECISION = True

# EMA Scheduler and regularization
SCHEDULER_TYPE = "cosine"  
STEP_SIZE = 5
GAMMA = 0.1
USE_EMA = True
EMA_DECAY = 0.999
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 8
AUTO_LR = True  # Auto LR

# VGG + CBAM Model config
MODEL_TYPE = "vgg16_cbam"  # Models: "vgg16", "vgg16_cbam", "resnet50"
USE_LABEL_SMOOTHING = True  
LABEL_SMOOTHING = 0.1
USE_MIXUP = True  # Mix up with combined augmentation
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0

# Debug/ saving checkpt
DEBUG_MODE = False  
SUBSET_SIZE = 1000  # Size of subset for Debug
SAVE_FREQ = 5  # Save model by n epochs
RUN_ID = f"{MODEL_TYPE}_{DATASET_TYPE.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
SAVE_DIR = "C:/Users/user/Desktop/COMP7250_ML_MiniProject/SyntheticAugmented_saved_models"
RESULTS_DIR = f"C:/Users/user/Desktop/COMP7250_ML_MiniProject/results/{RUN_ID}"
LOAD_PRETRAINED = False
PRETRAINED_PATH = None

# Testing and evaluation
TEST_MODE = False
EVALUATE_MISCLASSIFICATIONS = True
VISUALIZE_CAM = True
NUM_IMAGES_TO_PLOT = 20
ENSEMBLE_MODE = False

# CBAM module

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Main models

class VGG16_CBAM(nn.Module):
    def __init__(self, num_classes=7, feature_extract=True):
        super(VGG16_CBAM, self).__init__()
        
        vgg16 = models.vgg16(weights="DEFAULT")
        
        if feature_extract:
            for param in vgg16.parameters():
                param.requires_grad = False
        self.features_list = list(vgg16.features)
        
        # Add CBAM after specific layers
        self.features = self._add_cbam_layers()
        
        # Output size from feature
        dummy_input = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        dummy_output = self.features(dummy_input)
        flatten_size = dummy_output.view(1, -1).shape[1]
        
        # Classifier with correct input size
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        # Only the last layer is trainable for feature
        if feature_extract:
            for param in self.classifier[:-1].parameters():
                param.requires_grad = False
                
    def _add_cbam_layers(self):
        new_features = []
        
        # Add CBAM after each max pooling layer
        for i, layer in enumerate(self.features_list):
            new_features.append(layer)
            
            # After MaxPool layers (indices 4, 9, 16, 23, 30)
            if isinstance(layer, nn.MaxPool2d):
                for j in range(i, -1, -1):
                    if isinstance(self.features_list[j], nn.Conv2d):
                        in_channels = self.features_list[j].out_channels
                        break
                new_features.append(CBAM(in_channels))
                
        return nn.Sequential(*new_features)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def get_activation_maps(self, x, target_layer_idx=-2):
        activations = []
        
        # Forward pass through feature layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, CBAM):
                activations.append(x)  
        # Return the last activation if none specified
        if target_layer_idx == -1:
            return activations[-1]
        else:
            return activations[target_layer_idx]

# Setup selected model
def setup_model(model_type, num_classes, feature_extract):
    if model_type == "vgg16":
        model = models.vgg16(weights="DEFAULT")
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    elif model_type == "vgg16_cbam":
        # VGG16 with CBAM attention
        model = VGG16_CBAM(num_classes=num_classes, feature_extract=feature_extract)
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    elif model_type == "resnet50":
        # ResNet50
        model = models.resnet50(weights="DEFAULT")
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Feature extraction mode: {'Only training classifier layers' if feature_extract else 'Fine-tuning all layers'}")
    
    return model, params_to_update

# EMA for LR

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# Mix up for augmentation

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Applies mixup augmentation to a batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """Applies CutMix augmentation to a batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    h, w = x.size(2), x.size(3)
    bbx1, bby1, bbx2, bby2 = rand_bbox(h, w, lam)

    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    # Return cutmix'ed batch and targets
    return x_cutmix, y, y[index], lam

def rand_bbox(h, w, lam):
    """Generate random bounding box for CutMix"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    return bbx1, bby1, bbx2, bby2

# DATA PREPARATION

def prepare_data(data_dir, input_size, batch_size, workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print(f"Dataset loaded from: {data_dir}")
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    print(f"Total images: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    
    if DEBUG_MODE:
        indices = torch.randperm(len(full_dataset))[:SUBSET_SIZE]
        full_dataset = Subset(full_dataset, indices)
        print(f"Using {SUBSET_SIZE} images for debug mode")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataset_with_aug = Subset(
        datasets.ImageFolder(data_dir, transform=data_transforms['train']),
        train_dataset.indices if hasattr(train_dataset, 'indices') else list(range(len(train_dataset)))
    )
    
    val_dataset_with_aug = Subset(
        datasets.ImageFolder(data_dir, transform=data_transforms['val']),
        val_dataset.indices if hasattr(val_dataset, 'indices') else list(range(len(val_dataset)))
    )
    
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    
    dataloaders = {
        'train': DataLoader(train_dataset_with_aug, batch_size=batch_size, shuffle=True, 
                           num_workers=workers, pin_memory=PIN_MEMORY),
        'val': DataLoader(val_dataset_with_aug, batch_size=batch_size, shuffle=False, 
                         num_workers=workers, pin_memory=PIN_MEMORY)
    }
    
    # Test dataloader by loading a single batch
    print("Testing data loader with a single batch...")
    start_time = time.time()
    for inputs, labels in dataloaders['train']:
        print(f"Successfully loaded first batch in {time.time() - start_time:.2f} seconds")
        print(f"Batch shapes - X: {inputs.shape}, y: {labels.shape}")
        break
    
    dataset_info = {
        'class_names': full_dataset.dataset.classes if hasattr(full_dataset, 'dataset') and hasattr(full_dataset.dataset, 'classes') else CLASSES,
        'class_to_idx': full_dataset.dataset.class_to_idx if hasattr(full_dataset, 'dataset') and hasattr(full_dataset.dataset, 'class_to_idx') else None
    }
    
    return dataloaders, len(dataset_info['class_names']), dataset_info

# Main training

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, 
                num_epochs, model_save_path, model_name=None, plot_name=None):
    
    # Initialize with best validation accuracy so far
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Tracking of statistics
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    lr_history = []
    
    # Epoch-wise tracking
    epoch_train_losses = []
    epoch_train_accs = []
    epoch_val_losses = []
    epoch_val_accs = []
    
    # Mixed precision setup
    scaler = GradScaler('cuda') if USE_MIXED_PRECISION else None
    
    # EMA
    ema = EMA(model, EMA_DECAY) if USE_EMA else None
    
    # Early stopping setup
    if EARLY_STOPPING:
        early_stopping_counter = 0
        early_stopping_min_loss = float('inf')
    
    # Test a single batch through the pipeline to make sure everything works
    def test_single_batch():
        print("\nTesting a single batch before training...")
        
        try:
            model.train()
            inputs, labels = next(iter(dataloaders['train']))
            inputs = inputs.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)
            print("Data successfully moved to device")
            optimizer.zero_grad(set_to_none=True)
            
            # Apply mix up if enabled
            use_mixup = USE_MIXUP and np.random.rand() < 0.5
            use_cutmix = USE_MIXUP and not use_mixup and np.random.rand() < 0.5
            
            if use_mixup:
                # Mix up augmentation
                inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, device)
                with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                    outputs = model(inputs_mixed)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif use_cutmix:
                # Cut Mix augmentation
                inputs_mixed, targets_a, targets_b, lam = cutmix_data(inputs, labels, CUTMIX_ALPHA, device)
                with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                    outputs = model(inputs_mixed)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # Standard forward pass
                with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
            print(f"Forward pass successful, loss: {loss.item():.4f}")
                
            if USE_MIXED_PRECISION:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            if USE_EMA:
                ema.update()
                
            print("Single batch test completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in test batch: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
            
    # Run the single batch test
    batch_test_ok = test_single_batch()
    if not batch_test_ok:
        print("Batch test failed. Please check your configuration.")
        return model, [], [], [], []
    
    total_start_time = time.time()
    print("\nStarting training...")
    
    detailed_tracking_dir = os.path.join(RESULTS_DIR, 'training_details')
    os.makedirs(detailed_tracking_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                if USE_EMA:
                    ema.apply_shadow()
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            batch_count = 0  
            batch_losses = []
            batch_accs = []
            
            for inputs, labels in dataloaders[phase]:
                batch_count += 1
                
                inputs = inputs.to(device, non_blocking=PIN_MEMORY)
                labels = labels.to(device, non_blocking=PIN_MEMORY)
                
                # Zero gradients
                optimizer.zero_grad(set_to_none=True)
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Apply mix up or cut mix if in training mode
                    if phase == 'train' and USE_MIXUP:
                        use_mixup = np.random.rand() < 0.5
                        use_cutmix = not use_mixup and np.random.rand() < 0.5
                        
                        if use_mixup:
                            inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, device)
                            with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                                outputs = model(inputs_mixed)
                                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                            
                            # For accuracy calculation (dominant class)
                            _, preds = torch.max(outputs, 1)
                            # Fixed: Convert the tensor to float before multiplication
                            correct_a = torch.sum(preds == targets_a.data).float()
                            correct_b = torch.sum(preds == targets_b.data).float()
                            running_corrects += (correct_a * lam + correct_b * (1 - lam)).long()
                            
                        elif use_cutmix:
                            inputs_mixed, targets_a, targets_b, lam = cutmix_data(inputs, labels, CUTMIX_ALPHA, device)
                            with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                                outputs = model(inputs_mixed)
                                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                                
                            # For accuracy calculation (dominant class)
                            _, preds = torch.max(outputs, 1)
                            correct_a = torch.sum(preds == targets_a.data).float()
                            correct_b = torch.sum(preds == targets_b.data).float()
                            running_corrects += (correct_a * lam + correct_b * (1 - lam)).long()
                            
                        else:
                            # Standard forward pass
                            with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                            
                            _, preds = torch.max(outputs, 1)
                            running_corrects += torch.sum(preds == labels.data)
                    else:
                        with autocast('cuda') if USE_MIXED_PRECISION and phase == 'train' else nullcontext():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == labels.data)
                    
                    # Backward pass and optimization
                    if phase == 'train':
                        if USE_MIXED_PRECISION:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        
                        # Update EMA model
                        if USE_EMA:
                            ema.update()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Record batch statistics
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc.item())
                
                # Print progress every 100 batches
                if batch_count % 100 == 0 or batch_count == len(dataloaders[phase]):
                    current_acc = running_corrects.double() / (batch_count * inputs.size(0))
                    current_loss = running_loss / (batch_count * inputs.size(0))
                    print(f"{phase} progress: {batch_count}/{len(dataloaders[phase])} batches | " 
                          f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Save history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                lr_history.append(optimizer.param_groups[0]['lr'])
                
                # Save detailed batch stats
                epoch_train_losses = batch_losses
                epoch_train_accs = batch_accs
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                epoch_val_losses = batch_losses
                epoch_val_accs = batch_accs
                
                # Save the best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict() if not USE_EMA else {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                    
                    # Save model
                    if model_save_path and (epoch+1) % SAVE_FREQ == 0:
                        os.makedirs(model_save_path, exist_ok=True)
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': best_acc,
                            'loss': epoch_loss,
                        }, os.path.join(model_save_path, f"{model_name}_best.pth"))

                if EARLY_STOPPING:
                    if epoch_loss < early_stopping_min_loss:
                        early_stopping_min_loss = epoch_loss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        print(f"Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
                        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                            print("Early stopping triggered")
                            break
        
        if epoch % 1 == 0:  # Can adjust frequency
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            axes[0].plot(epoch_train_losses, 'b-', alpha=0.7, label='Train')
            axes[0].plot(epoch_val_losses, 'r-', alpha=0.7, label='Val')
            axes[0].set_title(f'Epoch {epoch+1} - Batch Losses')
            axes[0].set_xlabel('Batch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            axes[1].plot(epoch_train_accs, 'b-', alpha=0.7, label='Train')
            axes[1].plot(epoch_val_accs, 'r-', alpha=0.7, label='Val')
            axes[1].set_title(f'Epoch {epoch+1} - Batch Accuracies')
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_tracking_dir, f'epoch_{epoch+1}_details.png'))
            plt.close()
            
            plot_training_progress(train_loss_history, val_loss_history, train_acc_history, val_acc_history,
                                 lr_history, os.path.join(detailed_tracking_dir, f'progress_epoch_{epoch+1}.png'))
        
        if USE_EMA and epoch != num_epochs - 1:  # Don't restore on last epoch
            ema.restore()
        
        # Step scheduler
        if scheduler is not None and SCHEDULER_TYPE != 'plateau':
            scheduler.step()
        elif scheduler is not None and SCHEDULER_TYPE == 'plateau':
            scheduler.step(val_loss_history[-1])
        
        # Auto LR adjustment
        if AUTO_LR and epoch > 0 and epoch % 5 == 0:
            # If validation loss hasn't improved for 5 epochs, reduce LR
            if len(val_loss_history) > 5 and val_loss_history[-1] > val_loss_history[-5]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    print(f"Auto-adjusting learning rate to {param_group['lr']}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch complete in {epoch_time:.2f}s")
        
        # Save checkpoint
        if model_save_path and (epoch+1) % SAVE_FREQ == 0:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': epoch_acc,
                'loss': epoch_loss,
            }, os.path.join(model_save_path, f"{model_name}_epoch{epoch+1}.pth"))
        
        if EARLY_STOPPING and early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            break
    
    total_time = time.time() - total_start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    
    # Restore best model weights
    model.load_state_dict(best_model_wts)
    
    # Save and plot training history
    if plot_name:
        save_and_plot_results(train_loss_history, val_loss_history, train_acc_history, val_acc_history, 
                             lr_history, plot_name)
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def plot_training_progress(train_loss, val_loss, train_acc, val_acc, lr_history, save_path):
    """Plot and save the training progress up to the current epoch"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(lr_history)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    # Add a summary panel
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    best_train_acc = max(train_acc) if train_acc else 0
    best_val_acc = max(val_acc) if val_acc else 0
    best_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0
    final_lr = lr_history[-1] if lr_history else 0
    
    summary_text = (
        f"Training Progress (Epoch {len(train_loss)})\n\n"
        f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})\n"
        f"Best Training Accuracy: {best_train_acc:.4f}\n"
        f"Current Learning Rate: {final_lr:.8f}\n"
        f"Epochs so far: {len(train_loss)}\n"
    )
    
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Evaluation and data visualisation

def evaluate_model(model, dataloader, criterion, device, dataset_info, results_dir=None):
    """Comprehensive model evaluation"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # For misclassification analysis
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    batch_idx = 0
    image_paths = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)
            
            with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track misclassified samples
            if EVALUATE_MISCLASSIFICATIONS:
                for i in range(inputs.size(0)):
                    if preds[i] != labels[i]:
                        # Store with batch offset
                        img_idx = batch_idx * dataloader.batch_size + i
                        if hasattr(dataloader.dataset, 'dataset'):
                            # Handle case where dataset is a Subset
                            dataset = dataloader.dataset.dataset
                            if hasattr(dataloader.dataset, 'indices'):
                                idx = dataloader.dataset.indices[i]
                                img_path = dataset.samples[idx][0] if hasattr(dataset, 'samples') else None
                            else:
                                img_path = None
                        else:
                            # Handle case where dataset is the full dataset
                            dataset = dataloader.dataset
                            img_path = dataset.samples[img_idx][0] if hasattr(dataset, 'samples') else None
                        
                        # Store the misclassified information
                        misclassified_images.append(inputs[i].cpu())
                        misclassified_preds.append(preds[i].item())
                        misclassified_labels.append(labels[i].item())
                        image_paths.append(img_path)
            
            batch_idx += 1
    
    # Calculate final metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f"Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    # Generate classification report
    class_names = dataset_info['class_names']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    plt.close()
    
    if EVALUATE_MISCLASSIFICATIONS and results_dir and len(misclassified_images) > 0:
        analyze_misclassifications(misclassified_images, misclassified_preds, misclassified_labels, 
                                  class_names, image_paths, results_dir)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def analyze_misclassifications(images, preds, labels, class_names, image_paths, results_dir):
    """Analyze and visualize misclassified samples"""
    print(f"Analyzing {len(images)} misclassified images...")
    
    # Debug information
    print(f"Type of class_names: {type(class_names)}")
    print(f"Sample class_names: {class_names[:3]}")
    print(f"Type of a label: {type(labels[0])}")
    print(f"Sample labels: {labels[:3]}")
    print(f"Type of a pred: {type(preds[0])}")
    print(f"Sample preds: {preds[:3]}")
    
    # Create directory for misclassifications
    misclass_dir = os.path.join(results_dir, 'misclassifications')
    os.makedirs(misclass_dir, exist_ok=True)
    
    # Count misclassifications by class
    misclass_counts = {}
    for true_label, pred_label in zip(labels, preds):
        # Fixed: Convert to integer indices if they're not already
        true_class = class_names[int(true_label)]
        pred_class = class_names[int(pred_label)]
        key = f"{true_class} -> {pred_class}"
        misclass_counts[key] = misclass_counts.get(key, 0) + 1
    
    # Sort by most common misclassifications
    sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Plot most common misclassifications
    plt.figure(figsize=(12, 8))
    labels_text = [x[0] for x in sorted_misclass[:10]]
    values = [x[1] for x in sorted_misclass[:10]]
    plt.barh(labels_text, values)
    plt.xlabel('Count')
    plt.title('Top 10 Misclassifications')
    plt.tight_layout()
    plt.savefig(os.path.join(misclass_dir, 'misclassification_counts.png'))
    plt.close()
    
    # Save misclassification analysis to CSV
    misclass_df = pd.DataFrame(sorted_misclass, columns=['Misclassification', 'Count'])
    misclass_df.to_csv(os.path.join(misclass_dir, 'misclassification_analysis.csv'), index=False)
    
    # Plot a selection of misclassified images
    num_to_plot = min(len(images), NUM_IMAGES_TO_PLOT)
    
    # Select a diverse set of misclassifications
    if len(images) > num_to_plot:
        # Try to get at least one example of each misclassification type
        selected_indices = []
        seen_pairs = set()
        
        for i in range(len(labels)):
            pair = (labels[i], preds[i])
            if pair not in seen_pairs and len(selected_indices) < num_to_plot:
                selected_indices.append(i)
                seen_pairs.add(pair)
        
        # If we still need more, add random ones
        while len(selected_indices) < num_to_plot:
            idx = random.randint(0, len(images) - 1)
            if idx not in selected_indices:
                selected_indices.append(idx)
    else:
        selected_indices = range(len(images))
        
    # Plot the selected misclassifications
    for i, idx in enumerate(selected_indices):
        plt.figure(figsize=(6, 6))
        img = images[idx]
        img = to_pil_image(img)
        plt.imshow(img)
        # Fixed: Convert labels and preds to integers
        plt.title(f'True: {class_names[int(labels[idx])]}\nPredicted: {class_names[int(preds[idx])]}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(misclass_dir, f'misclassified_{i}.png'))
        plt.close()
        
        if image_paths[idx]:
            try:
                img_name = os.path.basename(image_paths[idx])
                # Fixed: Convert to integer indices
                true_class = class_names[int(labels[idx])]
                pred_class = class_names[int(preds[idx])]
                shutil.copy(image_paths[idx], 
                           os.path.join(misclass_dir, f'original_{i}_{true_class}_{pred_class}_{img_name}'))
            except Exception as e:
                print(f"Could not copy original image: {e}")
    
    print(f"Misclassification analysis complete - see {misclass_dir}")

def visualize_class_activation_maps(model, dataloader, class_names, device, results_dir):
    """Generate class activation maps to visualize model attention"""
    print("Generating class activation maps...")
    cam_dir = os.path.join(results_dir, 'activation_maps')
    os.makedirs(cam_dir, exist_ok=True)
    
    # Get a batch of validation images
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    
    # Only visualize a few images
    num_to_visualize = min(5, inputs.size(0))
    inputs = inputs[:num_to_visualize]
    labels = labels[:num_to_visualize]
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    
    # Generate CAM for each image
    for i in range(num_to_visualize):
        img_tensor = inputs[i].cpu()
        img = to_pil_image(img_tensor)
        
        # Prepare figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display original image
        axes[0].imshow(img)
        axes[0].set_title(f'True: {class_names[labels[i]]}, Pred: {class_names[preds[i]]}')
        axes[0].axis('off')
        
        # Get activation maps
        if hasattr(model, 'get_activation_maps'):
            # Use model's CAM extraction if available
            img_input = inputs[i].unsqueeze(0)
            
            # Get activation map
            activation = model.get_activation_maps(img_input)
            activation = activation.squeeze().cpu().detach()
            
            # Average the channels
            cam = torch.mean(activation, dim=0)
            
            # Normalize the CAM
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            # Resize to match input image
            cam = cam.numpy()
            cam = cv2.resize(cam, (img.width, img.height))
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Superimpose on the original image
            img_array = np.array(img)
            superimposed = 0.6 * heatmap + 0.4 * img_array
            superimposed = np.uint8(superimposed)
            
            # Display the heatmap
            axes[1].imshow(superimposed)
            axes[1].set_title('Class Activation Map')
            axes[1].axis('off')
        else:
            # If model doesn't support CAM, just display the feature map
            axes[1].text(0.5, 0.5, "CAM not available for this model", 
                        horizontalalignment='center', verticalalignment='center')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cam_dir, f'cam_{i}.png'))
        plt.close()
    
    print(f"CAM visualization complete - see {cam_dir}")

def save_and_plot_results(train_loss, val_loss, train_acc, val_acc, learning_rates, save_dir):
    """Save and plot training/validation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results as JSON
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'learning_rates': learning_rates
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    # Add a summary panel
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    best_train_acc = max(train_acc)
    best_val_acc = max(val_acc)
    best_epoch = val_acc.index(best_val_acc) + 1
    final_lr = learning_rates[-1]
    
    summary_text = (
        f"Training Summary\n\n"
        f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})\n"
        f"Best Training Accuracy: {best_train_acc:.4f}\n"
        f"Final Learning Rate: {final_lr:.8f}\n"
        f"Total Epochs: {len(train_loss)}\n"
    )
    
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'))
    plt.close()
    
    # Also create a combined plot showing everything in one
    plt.figure(figsize=(10, 6))
    
    # Plot loss and accuracy together
    plt.subplot(1, 1, 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Val Loss', color='red')
    
    # Create a twin y-axis for accuracy
    ax2 = plt.twinx()
    ax2.plot(train_acc, label='Train Acc', color='blue', linestyle='--')
    ax2.plot(val_acc, label='Val Acc', color='red', linestyle='--')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Accuracy')
    
    # Add learning rate markers
    for i, lr in enumerate(learning_rates):
        if i == 0 or lr != learning_rates[i-1]:
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
    
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_plot.png'))
    plt.close()
    
    # Additional trend visualization
    visualize_training_trends(train_loss, val_loss, train_acc, val_acc, save_dir)

def visualize_training_trends(train_loss, val_loss, train_acc, val_acc, save_dir):
    """Create detailed trend visualizations for loss and accuracy"""
    # Create a directory for trend plots
    trends_dir = os.path.join(save_dir, 'trends')
    os.makedirs(trends_dir, exist_ok=True)
    
    # Detailed loss trend plot
    plt.figure(figsize=(14, 8))
    epochs = np.arange(1, len(train_loss) + 1)
    
    # Plot loss values
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    
    # Add trend lines (moving average)
    window_size = min(5, len(train_loss))
    if window_size > 1:
        train_loss_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
        val_loss_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
        smooth_epochs = np.arange(window_size, len(train_loss) + 1)
        plt.plot(smooth_epochs, train_loss_smooth, 'b--', alpha=0.6, linewidth=1.5, label='Train Loss Trend')
        plt.plot(smooth_epochs, val_loss_smooth, 'r--', alpha=0.6, linewidth=1.5, label='Val Loss Trend')
    
    # Add best epoch marker
    best_epoch = val_loss.index(min(val_loss)) + 1
    plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, 
                label=f'Best Epoch ({best_epoch})')
    
    # Styling
    plt.title('Loss Trends During Training', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotations for key points
    min_val_loss = min(val_loss)
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}', 
                xy=(best_epoch, min_val_loss),
                xytext=(best_epoch + 1, min_val_loss * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Final loss gap
    final_gap = abs(train_loss[-1] - val_loss[-1])
    plt.annotate(f'Final Gap: {final_gap:.4f}', 
                xy=(len(train_loss), (train_loss[-1] + val_loss[-1])/2),
                xytext=(len(train_loss) - 3, (train_loss[-1] + val_loss[-1])/2 * 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(trends_dir, 'loss_trends.png'), dpi=300)
    plt.close()
    
    # Create a detailed accuracy trend plot
    plt.figure(figsize=(14, 8))
    
    # Plot accuracy values
    plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
    
    # Add trend lines (moving average)
    if window_size > 1:
        train_acc_smooth = np.convolve(train_acc, np.ones(window_size)/window_size, mode='valid')
        val_acc_smooth = np.convolve(val_acc, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smooth_epochs, train_acc_smooth, 'b--', alpha=0.6, linewidth=1.5, label='Train Acc Trend')
        plt.plot(smooth_epochs, val_acc_smooth, 'r--', alpha=0.6, linewidth=1.5, label='Val Acc Trend')
    
    # Add best epoch marker
    best_epoch = val_acc.index(max(val_acc)) + 1
    plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, 
                label=f'Best Epoch ({best_epoch})')
    
    # Styling
    plt.title('Accuracy Trends During Training', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotations for key points
    max_val_acc = max(val_acc)
    plt.annotate(f'Max Val Acc: {max_val_acc:.4f}', 
                xy=(best_epoch, max_val_acc),
                xytext=(best_epoch + 1, max_val_acc * 0.95),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Final accuracy gap
    final_gap = abs(train_acc[-1] - val_acc[-1])
    plt.annotate(f'Final Gap: {final_gap:.4f}', 
                xy=(len(train_acc), (train_acc[-1] + val_acc[-1])/2),
                xytext=(len(train_acc) - 3, (train_acc[-1] + val_acc[-1])/2 * 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(trends_dir, 'accuracy_trends.png'), dpi=300)
    plt.close()

# Model testing run

def test_on_single_image(model, image_path, class_names, device, transform=None):
    """Test model on a single image"""
    if not transform:
        transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
    
    # Get prediction and confidence
    _, pred = torch.max(outputs, 1)
    confidence = probabilities[0][pred].item()
    
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    top3_classes = [class_names[idx] for idx in top3_indices[0].cpu().numpy()]
    top3_probs = top3_prob[0].cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Prediction: {class_names[pred]} ({confidence:.2%})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top3_classes))
    plt.barh(y_pos, top3_probs)
    plt.yticks(y_pos, top3_classes)
    plt.xlabel('Probability')
    plt.title('Top 3 Predictions')
    
    plt.tight_layout()
    
    return {
        'prediction': class_names[pred],
        'confidence': confidence,
        'top3_classes': top3_classes,
        'top3_probs': top3_probs
    }

def load_pretrained_model(model, model_path, device):
    """Load a pretrained model from disk"""
    print(f"Loading pretrained model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (Epoch {checkpoint.get('epoch', 'unknown')}, "
              f"Accuracy {checkpoint.get('accuracy', 'unknown')})")
        return model, checkpoint.get('accuracy', 0)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0

def ensemble_predictions(models, dataloader, device):
    """Combine predictions from multiple models"""
    all_preds = []
    all_probs = []
    all_labels = []
    
    for i, model in enumerate(models):
        model.eval()
        model_preds = []
        model_probs = []
        labels = []
        
        print(f"Getting predictions from model {i+1}/{len(models)}...")
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader):
                inputs = inputs.to(device, non_blocking=PIN_MEMORY)
                
                with autocast('cuda') if USE_MIXED_PRECISION else nullcontext():
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                
                model_preds.extend(preds.cpu().numpy())
                model_probs.append(probs.cpu().numpy())
                labels.extend(targets.numpy())
        
        all_preds.append(model_preds)
        all_probs.append(np.vstack(model_probs))
        all_labels = labels
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Combine predictions (voting)
    ensemble_preds = []
    for i in range(len(all_labels)):
        # Get votes for this sample
        votes = all_preds[:, i]
        # Count occurrences of each class
        vote_counts = np.bincount(votes, minlength=len(CLASSES))
        # Find the class with the most votes
        ensemble_preds.append(np.argmax(vote_counts))
    
    # Combine probabilities (averaging)
    ensemble_probs = np.mean(all_probs, axis=0)
    
    # Calculate accuracy
    ensemble_acc = np.mean(np.array(ensemble_preds) == np.array(all_labels))
    
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    
    return ensemble_preds, ensemble_probs, all_labels, ensemble_acc


# Main function

def run_experiment():
    """Run the complete emotion recognition experiment"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # GPU - cuda else cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Get total GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*50)
    print("CONFIGURATION")
    print("="*50)
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Feature Extract: {FEATURE_EXTRACT}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Optimizer: {OPTIMIZER_TYPE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {WORKERS}")
    print(f"Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"Scheduler: {SCHEDULER_TYPE}")
    print(f"EMA: {USE_EMA}")
    print(f"MixUp: {USE_MIXUP}")
    print(f"Label Smoothing: {USE_LABEL_SMOOTHING}")
    print(f"Run ID: {RUN_ID}")
    print("="*50 + "\n")
    
    with open(os.path.join(RESULTS_DIR, 'config.json'), 'w') as f:
        json.dump({
            'dataset_type': DATASET_TYPE,
            'data_dir': DATA_DIR,
            'input_size': INPUT_SIZE,
            'feature_extract': FEATURE_EXTRACT,
            'model_type': MODEL_TYPE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer_type': OPTIMIZER_TYPE,
            'epochs': NUM_EPOCHS,
            'workers': WORKERS,
            'mixed_precision': USE_MIXED_PRECISION,
            'scheduler_type': SCHEDULER_TYPE,
            'ema': USE_EMA,
            'mixup': USE_MIXUP,
            'label_smoothing': USE_LABEL_SMOOTHING,
            'run_id': RUN_ID,
        }, f, indent=4)
    
    dataloaders, num_classes, dataset_info = prepare_data(DATA_DIR, INPUT_SIZE, BATCH_SIZE, WORKERS)
    
    model, params_to_update = setup_model(MODEL_TYPE, num_classes, FEATURE_EXTRACT)
    model = model.to(device)
    
    # Load pretrained model if specified
    if LOAD_PRETRAINED and not TEST_MODE:
        model, pretrained_acc = load_pretrained_model(model, PRETRAINED_PATH, device)
        if model is None:
            print("Failed to load pretrained model. Exiting.")
            return
    
    # Setup loss function
    if USE_LABEL_SMOOTHING:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        print(f"Using CrossEntropyLoss with label smoothing {LABEL_SMOOTHING}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Setup optimizer
    if OPTIMIZER_TYPE == "adam":
        optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER_TYPE == "adamw":
        optimizer = optim.AdamW(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(params_to_update, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    
    # Setup scheduler
    if SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    elif SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif SCHEDULER_TYPE == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=GAMMA, patience=3)
    else:
        scheduler = None
    
    # Ensemble mode
    if ENSEMBLE_MODE:
        # Find all model files
        model_files = glob.glob(os.path.join(SAVE_DIR, "*.pth"))
        models = []
        
        for model_file in model_files:
            # Create a new model instance
            model_instance, _ = setup_model(MODEL_TYPE, num_classes, FEATURE_EXTRACT)
            model_instance = model_instance.to(device)
            
            # Load weights
            model_instance, _ = load_pretrained_model(model_instance, model_file, device)
            models.append(model_instance)
        
        # Perform ensemble prediction
        if len(models) > 0:
            ensemble_preds, ensemble_probs, ensemble_labels, ensemble_acc = ensemble_predictions(
                models, dataloaders['val'], device)
            
            # Save ensemble results
            with open(os.path.join(RESULTS_DIR, 'ensemble_results.json'), 'w') as f:
                json.dump({
                    'accuracy': float(ensemble_acc),
                    'num_models': len(models),
                    'model_files': model_files
                }, f, indent=4)
                
            print(f"Ensemble evaluation complete - Accuracy: {ensemble_acc:.4f}")
            return
    
    # Test mode
    
    if TEST_MODE:
        if LOAD_PRETRAINED:
            model, _ = load_pretrained_model(model, PRETRAINED_PATH, device)
            if model is None:
                print("Failed to load model for testing. Exiting.")
                return
                
            # Evaluate the model
            evaluate_model(model, dataloaders['val'], criterion, device, dataset_info, RESULTS_DIR)
            
            # Visualize class activation maps if enabled
            if VISUALIZE_CAM and hasattr(model, 'get_activation_maps'):
                visualize_class_activation_maps(model, dataloaders['val'], 
                                             dataset_info['class_names'], device, RESULTS_DIR)
            
            print("Testing complete.")
            return
        else:
            print("Error: Test mode requires a pretrained model path.")
            return
    
    # Train mode
    print("\nStarting training...")
    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device, 
        NUM_EPOCHS, SAVE_DIR, RUN_ID, RESULTS_DIR
    )
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(SAVE_DIR, f"{RUN_ID}_final.pth"))
    
    # Evaluate the model on validation set
    print("\nEvaluating final model...")
    evaluate_model(model, dataloaders['val'], criterion, device, dataset_info, RESULTS_DIR)
    
    # Visualize class activation maps if enabled
    if VISUALIZE_CAM and hasattr(model, 'get_activation_maps'):
        visualize_class_activation_maps(model, dataloaders['val'], 
                                     dataset_info['class_names'], device, RESULTS_DIR)
    
    print("\nExperiment completed.")
    return model, train_loss, val_loss, train_acc, val_acc


if __name__ == "__main__":
    run_experiment()