class DeepfakeDataset(Dataset):
    def __init__(self, split, transform=None, is_training=True):
        self.transform = transform
        self.is_training = is_training
        
        # Load image paths and labels
        self.data = []
        self.labels = []
        
        # Load real images
        real_dir = os.path.join('data', split, 'real')
        if os.path.exists(real_dir):
            real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
            self.data.extend(real_images)
            self.labels.extend([0] * len(real_images))
        
        # Load fake images
        fake_dir = os.path.join('data', split, 'fake')
        if os.path.exists(fake_dir):
            fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]
            self.data.extend(fake_images)
            self.labels.extend([1] * len(fake_images))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 