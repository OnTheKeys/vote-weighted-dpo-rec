import torch
import model
from torchmetrics.retrieval import RetrievalNormalizedDCG
from torch.utils.data import DataLoader

def train_loop(dataloader : DataLoader, model : model.two_tower, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    for batch_num,batch in enumerate(dataloader):
        #Sends all tensors to model device 
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        y_hat = model(batch)
        loss = loss_fn(y_hat, batch['votes'])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if batch_num % 100 == 0:
            rmse = torch.sqrt(loss)
            loss, current = loss.item(), batch_num * dataloader.batch_size + batch['votes'].shape[0] 
            print(f"loss: {loss:>7f}, rmse: {rmse:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    print(f"Test loss: {train_loss:>8f}")

    return train_loss

    

def test_loop(dataloader, model, loss_fn, device):
    ndcg_metric = RetrievalNormalizedDCG(top_k=10)
    size = len(dataloader.dataset)
    model.eval()
    test_loss= 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch_num,batch in enumerate(dataloader):
            #Sends all tensors to model device 
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            y_hat = model(batch)
            loss = loss_fn(y_hat, batch['votes'])
            test_loss += loss.item()
            ndcg_metric.update(y_hat, batch['votes'], batch['users'])
            
            if batch_num % 100 == 0:
                rmse = torch.sqrt(loss)
                loss_val, current = loss.item(), batch_num * dataloader.batch_size + batch['votes'].shape[0] 
                print(f"loss: {loss:>7f}, rmse: {rmse:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= num_batches
    ndcg = ndcg_metric.compute()
    print(f"Test loss: {test_loss:>8f}, NDCG: {ndcg:>8f}")

    return test_loss

            
    
