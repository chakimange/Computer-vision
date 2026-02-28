import torch
from torch import nn
from pathlib import Path
import torch. functional as f
import time
import torchvision.transforms as t
from torch.utils.data import DataLoader
import torch.optim as opti

from Evaluation import evaluate_model_comprehensive,calculate_metrics
from CNN_CustomData import CustomDataset

class Alex_Net(nn.Module):
    def __init__(self, in_channels,out_classes):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_classes = out_classes
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,out_channels=96, kernel_size=11,stride=4,padding=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256, kernel_size=5,stride=1,padding=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384, kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384, kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256, kernel_size=3,stride=1,padding=1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.drop = nn.Dropout(p=0.5)
        
        self.pooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        
        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=out_classes)
        self._initialize_weights()
    def forward(self, x):
        
        #Convolutional layer
        x = self.conv1(x)
        x = self.lrn(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        x = self.conv2(x)
        x = self.lrn(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        #Flatten the image
        x = x.view(-1,6*6*256)
        
        #FC layer
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        # MODIF: Initialisation correcte des poids
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
def one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in loader :
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output,labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = output.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time
    
    return epoch_loss, epoch_acc, elapsed
    
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad() :
        for images, labels in loader :
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            output = model(images)
            loss = criterion(output,labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = output.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        elapsed = time.time() - start_time
        
        return epoch_loss, epoch_acc, elapsed
            
batch_size = 32
num_epoch = 20
learning_rate = 1e-3
weight_decay = 1e-3
device = "cpu"


model = Alex_Net(in_channels=3,out_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = opti.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)
best_model_path = save_dir / "best_cnn.pth"

data_dir = r"C:\Users\Dell\Documents\Nouveau dossier\Computer-vision\Dataset\POC_Dataset"

transforms = t.Compose([
    t.ToPILImage(),
    t.Grayscale(num_output_channels=3),
    t.Resize((256, 256)),
    t.RandomCrop(227),
    t.RandomHorizontalFlip(p=0.5),
    t.RandomRotation(10), 
    t.ColorJitter(brightness=0.2, contrast=0.2),  
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = CustomDataset(data_dir, data_type="Training",transform=transforms, is_augment=True)
test_dataset = CustomDataset(data_dir, data_type="Testing", transform=transforms, is_augment=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_val_acc = 0.0
for epoch in range(num_epoch) : 
    print(f"\nEpoch [{epoch+1}/{num_epoch}]")
    
    train_loss, train_acc, train_time = one_epoch(model, train_dataloader, optimizer, criterion, device)
    print(f"Train | Loss : {train_loss:.4f} Acc : {train_acc*100:.2f}% Time : {train_time:.1f}s")
    
    val_loss, val_acc, val_time = evaluate(model, test_dataloader, criterion, device)
    print(f"Val | Loss : {val_loss:.4f} Acc : {val_acc*100:.2f}% Time : {val_time:.1f}s")
    
    
    if val_acc > best_val_acc :
        best_val_acc = val_acc
        torch.save({"epoch" : epoch + 1, "model_state_dict" : model.state_dict(), "optimizer_state_dict" : optimizer.state_dict(),
                    "val_acc" : best_val_acc},best_model_path)
        print(f" >> New best model saved with val acc : {best_val_acc * 100:.2f}%")
        
print(f"\nTraining completed. Best val acc : {best_val_acc * 100:.2f}")
print(f"Best model path :{best_model_path}")
    

checkpoint = torch.load("checkpoints/best_cnn.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

class_names = ["Classe_0", "Classe_1", "Classe_2", "Classe_3"]  

train_metrics, test_metrics = evaluate_model_comprehensive(
    model, train_dataloader, test_dataloader, device, class_names
)

metrics = calculate_metrics(model, test_dataloader, device)
print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"F1-Score (macro): {metrics['f1_macro']:.3f}")
print(f"Precision (macro): {metrics['precision_macro']:.3f}")
print(f"Recall (macro): {metrics['recall_macro']:.3f}")


#epoch = 20 
# LR = 2e-3, bias = 0 , wd = 1e-3 : Acc = 75.25%, F1 = .752, Pre = .786, Re = .744
# LR = 1e-3, bias = 0.1, wd = 1e-3 : Acc = 68.89%, F1 = .691, Pre = .730, Re = .683
# LR = 3e-3, bias = 0.1, wd = 1e-3 : Acc = 77.37%, F1 = .756, Pre = .768, Re = .751
# LR = 1e-3, bias = 0.1, wd = 2e-3 : Acc = 73.74%, F1 = .714, Pre = .767, Re = .712
# LR = 3e-3, bias = 0, wd = 3e-3 : Acc = 75.84%, F1 = .750, Pre = .767, Re = .745

#epoch = 25 
# : Acc = 71.34%