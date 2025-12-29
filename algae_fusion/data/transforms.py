
from torchvision import transforms
from algae_fusion.config import IMG_SIZE

def get_transforms(split='val'):
    """
    Returns standarized image transforms for the pipeline.
    
    Args:
        split (str): 'train' for augmentation, 'val' or 'test' for inference.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        # val or test
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
