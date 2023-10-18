import torch

from load_data import get_GoDataLoader
from CNN import CNN

from tqdm.autonotebook import tqdm
from torch import nn
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau



def train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, scheduler, epochs, device):
    with torch.autograd.set_detect_anomaly(True):
        for e in range(epochs):
            print(f"------------------------------EPOCH {e}------------------------------")
            model.train()
            progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=len(train_dataloader))
            losses = 0
            top1_correct = 0
            top5_correct = 0
            size = 0
            for batch_num, (data, label) in progress:
                data, label = data.to(device), label.to(device)
                pred = model(data)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, maxk = torch.topk(pred, 5, dim = -1)
                _, label = torch.topk(label, 1, dim=-1)
                top1_correct += torch.eq(maxk[:, 0], label[:, 0]).sum().item()
                top5_correct += torch.eq(maxk, label).sum().item()

                losses += loss.item()
                size += len(data)
                progress.set_description("Loss: {:.7f}, Top-1 Accuracy: {:.4f}, Top-5 Accuracy: {:.4f}".format(losses/(batch_num+1), top1_correct/size, top5_correct/size))
        
            valid_loss, valid_top1_accuracy, valid_top5_accuracy = test(valid_dataloader, model, loss_fn, device)
            print(f'Valid Loss: {valid_loss}, Valid Top1: {valid_top1_accuracy}%, Valid Top5: {valid_top5_accuracy}%')
            scheduler.step(valid_loss)
    return losses/(batch_num+1), top1_correct/size * 100, top5_correct/size * 100

def test(dataloader, model, loss_fn, device):
    size = 0
    num_batches = 0
    top1_correct = 0
    top5_correct = 0
    test_loss = 0
    model.eval()
    for data, label in dataloader:
        data = data.to(device); label = label.to(device)
        pred = model(data)

        test_loss += loss_fn(pred, label).item()
        _, maxk = torch.topk(pred, 5, dim = -1)
        _, label = torch.topk(label, 1, dim=-1)
        top1_correct += torch.eq(maxk[:, 0], label[:, 0]).sum().item()
        top5_correct += torch.eq(maxk, label).sum().item()

        size += len(data)
        num_batches += 1 
    test_loss /= num_batches
    top1_accuracy = top1_correct / size
    top5_accuracy = top5_correct / size
    return test_loss, top1_accuracy, top5_accuracy

lr = 0.1
epochs = 200
in_channels = 4
out_channels = 19*19
input_size = [19, 19]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

train_loader, valid_loader, test_loader = get_GoDataLoader('./data/Train/dan_train.csv', valid_size = 0.1, test_size = 0.2)

model = CNN(in_channels = in_channels, out_channels = out_channels).to(device)
print(model)
summary(model, input_size = (in_channels, *input_size))

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

train_loss, train_top1_accuracy, train_top5_accuracy = train(train_loader, valid_loader, model, loss_fn, optimizer, scheduler, epochs, device)
test_loss, test_top1_accuracy, test_top5_accuracy = test(test_loader, model, loss_fn, device)
print(f'Train Top1 Accuracy: {train_top1_accuracy}, Train Top5 Accuracy: {train_top5_accuracy}, Train Loss: {train_loss}')
print(f'Test Top1 Accuracy: {test_top1_accuracy}, Test Top5 Accuracy: {test_top5_accuracy}, Test Loss: {test_loss}')

checkpoint = {'model': CNN(in_channels = 4),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict(),
          'scheduler': scheduler.state_dict()}
os.makedirs('./result/exp/', exist_ok=True)
torch.save(checkpoint, f'./result/exp/dan_CNN_epoch{epochs}.pth')


            
            

            



