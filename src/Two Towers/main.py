import argparse
import model
import dataset
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from engine import train_loop, test_loop
import pandas as pd
import pickle
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from enum import Enum 




def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the latest state (overwrites previous to save disk space)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(state, latest_path)
    
    # Copy to best if it's the top-performing epoch so far
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        shutil.copyfile(latest_path, best_path)


def get_hf_token(platform):
    if platform == 'colab':
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    elif platform == 'kaggle':
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    else:
        try:
            load_dotenv()
            return os.getenv('HF_TOKEN')
        except ImportError:
            print("No HF_TOKEN found in environment variables, trying colab.")

def load_data(hf_token):

    REPO_ID = "RadnitzO/vndb"

    # Load train and test DataFrames
    train_set = pd.read_parquet(
        hf_hub_download(REPO_ID, filename="train_set.parquet", repo_type="dataset", token=hf_token)
    )
    test_set = pd.read_parquet(
        hf_hub_download(REPO_ID, filename="test_set.parquet", repo_type="dataset", token=hf_token)
    )

    #Load Dictionaries from Pickles
    with open(hf_hub_download(REPO_ID, filename="user_tags.pickle", repo_type="dataset", token=hf_token), "rb") as f:
        user_tags = pickle.load(f)

    with open(hf_hub_download(REPO_ID, filename="vn_tags.pickle", repo_type="dataset", token=hf_token), "rb") as f:
        vn_tags = pickle.load(f)

    with open(hf_hub_download(REPO_ID, filename="tags.pickle", repo_type="dataset", token=hf_token), "rb") as f:
        tags = pickle.load(f)


    return train_set, test_set, user_tags, vn_tags, tags

def main():

    PLATFORM_MAP = {
    "1": "kaggle",
    "kaggle": "kaggle",
    "2": "colab",
    "colab": "colab",
    "3": "local",
    "local": "local",
}

    parser = argparse.ArgumentParser()
    #Training args
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')

    #Model hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--num_tags', type=int, default=20, help='embedding dimension')
    parser.add_argument('--user_layers', nargs='+', type=int, default=[256, 128], help='user tower layers')
    parser.add_argument('--item_layers', nargs='+', type=int, default=[256, 128], help='item tower layers')

    #Automatically detect if cuda is available
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, help='device')

    #Checkpoint and resume logic
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint if available')
    parser.add_argument('--platform', type=str, default='local', help='platform to use for running')
    args = parser.parse_args()

    hf_token = get_hf_token(PLATFORM_MAP[args.platform.lower()])
    train_data, test_data, user_tags, vn_tags, tags = load_data(hf_token)



    train_data = dataset.vn_vote_dataset(train_data, user_tags, vn_tags, args.num_tags)
    test_data = dataset.vn_vote_dataset(test_data, user_tags, vn_tags, args.num_tags)
    #Connect and load dataset 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    len_unique_vns=27707 #Self explanatory magic number
    len_unique_users=70703#Self explanatory magic number

    two_tower_model = model.two_tower(args.embedding_dim, len_unique_users+1, len_unique_vns+1, len(tags)+1, args.user_layers, args.item_layers).to(args.device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')

    writer = SummaryWriter(Path(__file__).resolve().parent/'runs')

    if args.resume and os.path.exists(checkpoint_path):
        print(f" Found checkpoint at {checkpoint_path}. Resuming...")
        
        #Map_location ensures safe loading across different GPU types 
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
        two_tower_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed successfully! Starting from Epoch {start_epoch + 1}")
    else:
        print("Starting fresh training run.")
        
    for epoch in range(start_epoch,args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss =train_loop(train_loader, two_tower_model, loss_fn, optimizer, args.device)
        test_loss = test_loop(test_loader, two_tower_model, loss_fn, args.device)

        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss

        # Save checkpoint at the end of every epoch
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': two_tower_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'best_loss': best_loss,
            'args': vars(args) 
        }, is_best=is_best, checkpoint_dir=args.checkpoint_dir)
        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('best_loss', best_loss, epoch)
    
    writer.close()

if __name__ == "__main__":
    main()


    