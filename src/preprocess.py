# src/transforms.py

from torchvision import transforms

def to_rgb(img):
    return img.convert("RGB")

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
size= (224, 224)
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize(size),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),

        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),

    'val': transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),

    'test': transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}