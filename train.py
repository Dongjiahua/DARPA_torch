import argparse
import torch 
from model.unet import MAP_UNet
from data.dataset import MAPData
from torch.utils.data import DataLoader
import torchmetrics
import os 
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default="/media/jiahua/FILE/uiuc/NCSA/processed/training", help='Root train data path')
    parser.add_argument('--val_data', type=str, default="/media/jiahua/FILE/uiuc/NCSA/processed/validation",  help='Root val data path')
    parser.add_argument('--out_dir', type=str, default="./output", help='output_dir')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')

    # parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')

    return parser.parse_args()

def train_epoch(train_loader, model,optimizer,scheduler,criterion):
    model.train()
    metric = torchmetrics.Accuracy(task="binary").cuda()
    for i, data in enumerate(train_loader):

        optimizer.zero_grad()
        loss,acc= run_iter(data, model,metric,criterion)
        loss.backward()
        optimizer.step()
        if i%100==0:
            print(f"Accuracy on batch {i}/{len(train_loader)}: {acc}")
    scheduler.step()
    return metric.compute()
    
@torch.no_grad()
def val_epoch(train_loader, model):
    model.eval()
    metric = torchmetrics.Accuracy(task="binary").cuda()
    for i, data in tqdm(enumerate(train_loader)):
        _,_ = run_iter(data, model,metric)

    return metric.compute()

def run_iter(data, model, metric,criterion):
    map, legend, seg = data['map_img'].cuda(), data['legend_img'].cuda(), data['seg_img'].cuda()
    model_input = torch.cat([map,legend],dim=1)
    output = model(model_input)
    
    output = torch.nn.functional.interpolate(output,size=seg.shape[-2:],mode="nearest")
    loss = criterion(output, seg)
    acc = metric(output, seg)
    return loss,acc 

def train():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    
    model = MAP_UNet(n_channels=6,n_classes=1,pretrained=True).to(device)
    train_dataset = MAPData(data_path=args.train_data,type="poly")
    val_dataset = MAPData(data_path=args.val_data,type="poly")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.workers)
    
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    best_val_acc = 0
    for epoch in range(1,args.epochs+1):
        train_acc = train_epoch(train_loader, model,optimizer,scheduler,criterion)
        val_acc = val_epoch(val_loader, model)
        
        print(f"Epoch {epoch} train acc: {train_acc} val acc: {val_acc}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpts", 'best_model.pth'))
        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpts", 'latest.pth'))
    
if __name__ == "__main__":
    train()