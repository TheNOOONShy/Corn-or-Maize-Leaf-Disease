import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

# Hyper Parameters
EPOCH = 10           
BATCH_SIZE = 32
LR = 0.001          


transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = datasets.ImageFolder(
    root='/Users/youli/Desktop/CV/data',  
    transform=transform
)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv2d(
                in_channels=3,     
                out_channels=16,    
                kernel_size=5,      
                stride=1,           
                padding=2,     
            ),      
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(16, 32, 5, 1, 2),  
            nn.ReLU(),  
            nn.MaxPool2d(2),  
        )
        self.out = nn.Linear(32 * 37 * 37, 4)   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)  

optimizer = optim.Adam(cnn.parameters(), lr=LR)   
loss_func = nn.CrossEntropyLoss()   


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   
        output = cnn(b_x)               
        loss = loss_func(output, b_y)   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()                

        if step % 50 == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')


test_x, test_y = next(iter(test_loader))
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y.numpy(), 'real number')