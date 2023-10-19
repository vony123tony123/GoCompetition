import torch
import os
import wandb

from load_data import get_GoDataLoader
from CNN import CNN

from tqdm.autonotebook import tqdm
from torch import nn
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau



def train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, epochs, device):
    with torch.autograd.set_detect_anomaly(True):
        min_valid_loss = float('inf')
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
            print('Valid Loss: {:.7f}, Valid Top1: {:.4f}, Valid Top5: {:.4f}'.format(valid_loss, valid_top1_accuracy, valid_top5_accuracy))
            scheduler.step(valid_loss)

            metrics = {
                "train/loss": losses/(batch_num+1),
                "train/epoch": e,
                "train/top1_accuracy": top1_correct/size,
                "train/top5_accuracy": top5_correct/size,
                "train/learnrate": optimizer.param_groups[0]['lr'],
                "valid/loss": valid_loss,
                "valid/top1_accuracy": valid_top1_accuracy/100,
                "valid/top5_accuracy":valid_top5_accuracy/100
            }

            wandb.log(metrics, step=e)

            if valid_loss < min_valid_loss:
                checkpoint_model_state = model.state_dict()
                checkpoint_optimizer = optimizer.state_dict()
                checkpoint_scheduler = scheduler.state_dict()
                best_train_loss = losses/(batch_num+1)
                best_top1_accuracy = top1_correct/size
                best_top5_accuracy = top5_correct/size

        best_checkpoint = {
          'model': CNN(in_channels = 4),
          'model_weights': checkpoint_model_state,
          'optimizer' : checkpoint_optimizer,
          'scheduler': checkpoint_scheduler
        }


    return best_train_loss, best_top1_accuracy * 100, best_top5_accuracy * 100, best_checkpoint

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
    return test_loss, top1_accuracy * 100, top5_accuracy * 100



lr = 0.001
epochs = 200
in_channels = 4
out_channels = 19*19
input_size = [19, 19]
batch_size = 128
dataset = 'kyu'
description = ""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

model = CNN(in_channels = in_channels, out_channels = out_channels).to(device)

train_loader, valid_loader, test_loader = get_GoDataLoader(f'./data/Train/{dataset}_train.csv', valid_size = 0.1, test_size = 0.1, batch_size = batch_size)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="GoCompetition",

    name = f"CNN_{dataset}",

    notes = description,

    group = "CNN",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "epochs": epochs,
    "architecture": "CNN",
    "dataset": dataset,
    "train_data_num": len(train_loader.sampler),
    "valid_data_num": len(valid_loader.sampler),
    "test_data_num": len(test_loader.sampler),
    "batch_size": batch_size,
    "input shape": (in_channels, *input_size),
    "out_channels": out_channels,
    "lr scheduler": "ReduceLROnPlateau",
    "optimizer": "Adam",
    "loss_fn": "CrossEntropyLoss"
    }
)

summary(model, input_size = (in_channels, *input_size))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min')
wandb.watch(model, loss_fn, log="all", log_freq=1)

train_loss, train_top1_accuracy, train_top5_accuracy, best_checkpoint = train(train_loader, valid_loader, model, loss_fn, optimizer, epochs, device)
print(f'Train Loss: {train_loss}, Train Top1 Accuracy: {train_top1_accuracy}, Train Top5 Accuracy: {train_top5_accuracy}')

model.load_state_dict(best_checkpoint['model_weights'])

test_loss, test_top1_accuracy, test_top5_accuracy = test(test_loader, model, loss_fn, device)
print(f'Test Loss: {test_loss}, Test Top1 Accuracy: {test_top1_accuracy}, Test Top5 Accuracy: {test_top5_accuracy}')

# Record result into Wandb
wandb.summary['train_avg_loss'] = train_loss
wandb.summary['train_top1_accuracy'] = train_top1_accuracy
wandb.summary['train_top5_accuracy'] = train_top5_accuracy
wandb.summary['test_avg_loss'] = test_loss
wandb.summary['test_top1_accuracy'] = test_top1_accuracy
wandb.summary['test_top5_accuracy'] = test_top5_accuracy

os.makedirs('./result/exp/', exist_ok=True)
torch.save(best_checkpoint, f'./result/exp/{dataset}_CNN_epoch{epochs}.pth')

art = wandb.Artifact(f"{dataset}_CNN", type="model")
art.add_file(f'./result/exp/{dataset}_CNN_epoch{epochs}.pth')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()


            
            

            



