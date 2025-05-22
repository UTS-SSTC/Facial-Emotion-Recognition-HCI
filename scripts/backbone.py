import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from transformers import (
    AutoModelForImageClassification,
    AutoModel,
    DeiTForImageClassification
)


class VisionFeatureExtractor:
    """
    Generic feature extractor using HuggingFace vision models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. 'facebook/deit-base-distilled-patch16-224'
    feature_dim : int
        Dimensionality of the CLS token or pooled output
    device : str, optional
        Device to run the model on. Defaults to CUDA if available.
    freeze_ratio : float
        Fraction of transformer layers to freeze during fine-tuning.
    """

    def __init__(
            self,
            model_name: str = 'facebook/deit-base-distilled-patch16-224',
            feature_dim: int = 8,
            device: Optional[str] = None,
            freeze_ratio: float = 0.8
    ):
        # Set device
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.freeze_ratio = freeze_ratio
        self.is_finetuned = False

        # Load classification and base models
        print(f"[INFO] Loading classification and base vision models: {model_name} on {self.device}")
        if "deit" in model_name.lower():
            self.classification_model = DeiTForImageClassification.from_pretrained(model_name)
        else:
            self.classification_model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move to device
        self.classification_model.to(self.device)
        self.model.to(self.device)

        # Freeze all parameters initially
        self._freeze_all_parameters()

    def _freeze_all_parameters(self):
        for param in self.classification_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

    def prepare_for_finetuning(self, num_classes: int):
        model = self.classification_model
        cfg = model.config
        old_num_labels = getattr(cfg, "num_labels", None)

        # Classification header attribute name
        HEAD_ATTRS = [
            "classifier",  # ViT/DeiT
            "distillation_classifier",  # Distillation model
            "fc",  # ResNet
            "head",  # Custom model
            "heads",  # ModuleList/Dict
            "logits",  # Rare
        ]

        head_modules = []
        replaced = False

        # Try one by one, replace the classification header
        for attr in HEAD_ATTRS:
            if not hasattr(model, attr):
                continue
            module = getattr(model, attr)

            # For a single Linear, replace directly
            if isinstance(module, torch.nn.Linear):
                in_f = module.in_features
                new_head = torch.nn.Linear(in_f, num_classes).to(self.device)
                setattr(model, attr, new_head)
                head_modules.append(new_head)
                replaced = True

            # If it is a ModuleList or Dict (multiple head), replace all Linear layers in it
            elif isinstance(module, (torch.nn.ModuleList, dict)):
                for k, sub in (module.items() if isinstance(module, dict) else enumerate(module)):
                    if isinstance(sub, torch.nn.Linear) and sub.out_features == old_num_labels:
                        new_sub = torch.nn.Linear(sub.in_features, num_classes).to(self.device)
                        if isinstance(module, dict):
                            module[k] = new_sub
                        else:
                            module[k] = new_sub
                        head_modules.append(new_sub)
                        replaced = True
            if replaced:
                break

        # Find the first Linear that matches old_num_labels
        if not replaced and old_num_labels is not None:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == old_num_labels:
                    # Positioning parent module
                    parent = model
                    *path, leaf = name.split(".")
                    for p in path:
                        parent = getattr(parent, p)
                    # Replace
                    new_head = torch.nn.Linear(module.in_features, num_classes).to(self.device)
                    setattr(parent, leaf, new_head)
                    head_modules.append(new_head)
                    replaced = True
                    break

        if not replaced:
            raise RuntimeError(
                "[ERROR] No replacement classification header was found, please check the model structure.")

        # If there is a distillation head, replace and thaw together
        if hasattr(model, "distillation_classifier"):
            d = model.distillation_classifier
            if isinstance(d, torch.nn.Linear):
                new_d = torch.nn.Linear(d.in_features, num_classes).to(self.device)
                model.distillation_classifier = new_d
                head_modules.append(new_d)

        # Set all head module parameters to trainable
        for m in head_modules:
            for p in m.parameters():
                p.requires_grad = True

        # Find the backbone module
        backbone = getattr(model, "base_model", model)

        # If it is a TimmWrapper of HuggingFace, remove the .timm_model
        timm_wrap = getattr(backbone, "timm_model", None)
        if timm_wrap is not None:
            backbone = timm_wrap

        # Determine if it is a Transformer
        encoder = getattr(backbone, "encoder", None)
        blocks = None
        if encoder is not None:
            # Transformer: encoder.layer
            if hasattr(encoder, "layer"):
                blocks = list(encoder.layer)
            # HuggingFace ResNet: encoder.stages
            elif hasattr(encoder, "stages"):
                blocks = list(encoder.stages)
        # torchvision ResNet
        if blocks is None and all(hasattr(backbone, f"layer{i}") for i in (1, 2, 3, 4)):
            blocks = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        # EfficientNet
        if blocks is None and hasattr(backbone, "features"):
            blocks = list(backbone.features.children())
        # timm ViT
        if blocks is None and hasattr(backbone, 'blocks'):
            blocks = list(backbone.blocks)
        if blocks is None:
            raise RuntimeError(f"[ERROR] Unrecognized backbone for freezing logic: {type(backbone)}")

        # Unfreeze the last 20% of the blocks
        total = len(blocks)
        num_to_unfreeze = max(1, int(total * (1 - self.freeze_ratio)))
        for block in blocks[-num_to_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

        # Count trainable parameters
        tot = sum(p.numel() for p in model.parameters())
        tun = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Fine-tuning {tun:,}/{tot:,} params ({tun / tot:.2%})")

    def finetune(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            num_classes: int = 8,
            epochs: int = 5,
            learning_rate: float = 2e-5,
            weight_decay: float = 0.01
    ) -> Dict[str, List[float]]:
        self.prepare_for_finetuning(num_classes)
        self.classification_model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.classification_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader)
        )
        criterion = torch.nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.classification_model(imgs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * imgs.size(0)
            history['train_loss'].append(epoch_loss / len(train_loader.dataset))

            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(
                    f"Epoch {epoch + 1}: Train {history['train_loss'][-1]:.4f}, Val {val_loss:.4f}, Acc {val_acc:.4f}")
            else:
                print(f"Epoch {epoch + 1}: Train {history['train_loss'][-1]:.4f}")

        self._sync_models()
        self.is_finetuned = True
        return history

    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        self.classification_model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                out = self.classification_model(imgs)
                v_loss += criterion(out.logits, labels).item() * imgs.size(0)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return v_loss / len(val_loader.dataset), correct / total

    def _sync_models(self):
        # Copy backbone weights into base model
        try:
            backbone = self.classification_model.base_model
            self.model.load_state_dict(backbone.state_dict(), strict=False)
        except AttributeError:
            # fallback for models without .base_model
            self.model.load_state_dict(self.classification_model.state_dict(), strict=False)
        self.model.eval()

    def extract_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from images using fine tuned model.

        Parameters
        ----------
        dataloader : DataLoader
            PyTorch DataLoader containing the images.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (features, labels) as NumPy arrays.
        """
        self.model.eval()
        all_feats = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device)
            with torch.no_grad():
                out = self.model(images)

                # Transformer series model: pooler_output
                if hasattr(out, 'pooler_output'):
                    feat = out.pooler_output  # (B, hidden_size)

                # There is last_hidden_state and it is 4D (usually ViT)
                elif hasattr(out, 'last_hidden_state') and out.last_hidden_state.ndim == 4:
                    feat_map = out.last_hidden_state  # (B, C, H, W)
                    feat = feat_map.mean(dim=[2, 3])  # (B, C)

                # ResNetModel, ConvNextModel
                elif hasattr(out, 'features') and out.features.ndim == 4:
                    feat_map = out.features  # (B, C, H, W)
                    feat = feat_map.mean(dim=[2, 3])  # (B, C)

                else:
                    tensor = next(t for t in out.__dict__.values() if isinstance(t, torch.Tensor))
                    feat = tensor.view(tensor.size(0), -1)  # (B, ∏dims)

            all_feats.append(feat.cpu())
            all_labels.append(labels.cpu())

        # Splice and turn numpy
        feats_tensor = torch.cat(all_feats, dim=0)  # (N, feat_dim)
        labels_tensor = torch.cat(all_labels, dim=0)  # (N,)
        feats_tensor = feats_tensor.view(feats_tensor.size(0), -1)

        X = feats_tensor.numpy()
        y = labels_tensor.numpy()

        # print(f"[DEBUG] extract_features -> X.shape = {X.shape}, y.shape = {y.shape}")
        return X, y
