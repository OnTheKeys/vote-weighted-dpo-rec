import torch
import torch.nn.functional as F
import heapq 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

class vn_vote_dataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): Dataset class for VN votes
    """
    def __init__(self, df: DataFrame, user_tags: dict, vn_tags: dict, num_tags: int):
        """_summary_

        Args:
            df (DataFrame): Dataframe of user,vn interactions with columns user_idx, vn_idx, vote
            user_tags (dict): Tags associated with users as well as their weights, pulled from their top 10 ranked VNs.
            vn_tags (dict): Tags associated with a given VN as well as their weights, voted on by users of VNDB.
            num_tags (int): Number of tags to use for each user and VN, padded with zeros or truncated to this number for the sake of symmetry.
        """
        self.df = df
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.vns = torch.tensor(df['vn_idx'].values, dtype=torch.long)
        self.votes = torch.tensor(scaler.fit_transform(df['vote'].values.reshape(-1, 1)), dtype=torch.float).squeeze()
        self.user_tags = user_tags
        self.vn_tags = vn_tags
        self.num_tags = num_tags
        
    def __getitem__(self, idx):
        #Initialize tensors for user and vn tag IDs and weights to zeroes, for num length, will be overwritten if tags are found
        user_tag_ids = torch.zeros(self.num_tags, dtype=torch.long)
        user_tag_weights = torch.zeros(self.num_tags, dtype=torch.float)
    
        vn_tag_ids = torch.zeros(self.num_tags, dtype=torch.long)
        vn_tag_weights = torch.zeros(self.num_tags, dtype=torch.float)
        
        if self.users[idx].item() in self.user_tags:
            user_tag_list = self.user_tags[self.users[idx].item()]
            top_n_user_tags = heapq.nlargest(self.num_tags, user_tag_list.items(), key=lambda item: item[1])
            
            #Overwrites the first n found number of tags found up to num_tags for tensors, rest are zeroes for padding
            n_user = len(top_n_user_tags)
            user_tag_ids[:n_user] = torch.tensor([tag for tag, _ in top_n_user_tags], dtype=torch.long)
            user_tag_weights[:n_user] = torch.tensor([weight for _, weight in top_n_user_tags], dtype=torch.float)

        if self.vns[idx].item() in self.vn_tags:
            vn_tag_list = self.vn_tags[self.vns[idx].item()]

            top_n_vn_tags = heapq.nlargest(self.num_tags, vn_tag_list.items(), key=lambda item: item[1]['weight'])
    
            n_vn = len(top_n_vn_tags)
            vn_tag_ids[:n_vn] = torch.tensor([tag for tag, _ in top_n_vn_tags], dtype=torch.long)
            vn_tag_weights[:n_vn] = torch.tensor([data['weight'] for _, data in top_n_vn_tags], dtype=torch.float)

        return { 'users': self.users[idx], 
                'items': self.vns[idx], 
                'votes': self.votes[idx], 
                'user_tag_ids': user_tag_ids,
                'user_tag_weights': user_tag_weights,
                'item_tag_ids': vn_tag_ids,
                'item_tag_weights': vn_tag_weights}
    
    def __len__(self):
        return len(self.df)