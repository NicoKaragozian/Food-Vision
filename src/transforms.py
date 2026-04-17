"""
Transformaciones de imagen para entrenamiento e inferencia.
Normalizadas con estadísticas de ImageNet (estándar para transfer learning).
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_train_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Transformaciones de entrenamiento con data augmentation.
    RandomResizedCrop + ColorJitter demostrado efectivo en Food-101.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Transformaciones de validación/test — sin augmentación."""
    return transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Alias de get_val_transform para inferencia de una sola imagen."""
    return get_val_transform(input_size)


def denormalize(tensor):
    """Desnormaliza un tensor para visualización (invierte Normalize)."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
