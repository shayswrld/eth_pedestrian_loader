import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ETHPedestrianDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load dataset files here

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Load an image and its label
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image

# Define any transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = ETHPedestrianDataset(root_dir='path/to/dataset', transform=transform)

# Create data loaders
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the data
for images in data_loader:
    # Your training code here
    pass
