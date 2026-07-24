#set Up user embedings for two towers
import torch
import torch.nn as nn
import torch.nn.functional as F


class two_tower(nn.Module):
    def __init__(self, embedding_dim, num_users, num_items, num_tags, user_layers, item_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.tag_id_embedding = nn.Embedding(num_tags, embedding_dim)
        self.user_layers = user_layers
        self.item_layers = item_layers
        
        #Define user tower
        self.user_tower = []
        self.user_tower.append(nn.Linear(embedding_dim*2, user_layers[0]))
        self.user_tower.append(nn.ReLU())
        
        for i in range (len(self.user_layers)-1):
            self.user_tower.append(nn.Linear(user_layers[i], user_layers[i+1]))
            self.user_tower.append(nn.ReLU())
        self.user_tower = nn.Sequential(*self.user_tower)

        #Define item tower
        self.item_tower = []
        self.item_tower.append(nn.Linear(embedding_dim*2, item_layers[0]))
        self.item_tower.append(nn.ReLU())
        
        for i in range (len(self.item_layers)-1):
            self.item_tower.append(nn.Linear(item_layers[i], item_layers[i+1]))
            self.item_tower.append(nn.ReLU())
        self.item_tower = nn.Sequential(*self.item_tower)

    def get_tag_embeddings(self, user_tag_weights, user_tags_ids, item_tag_ids, item_tag_weights):
        
        #Get tag embeddings for user and item
        user_tag_embeddings = self.tag_id_embedding(user_tags_ids)
        item_tag_embeddings = self.tag_id_embedding(item_tag_ids)
        #turn the tag weights into probabilities via softmax
        probs_users = F.softmax(user_tag_weights, dim=1)
        probs_items = F.softmax(item_tag_weights, dim=1)
        #Add dummy dimension so that we can multiply probabilities with tag embeddings
        probs_users = probs_users.unsqueeze(-1)
        probs_items = probs_items.unsqueeze(-1)

        weighted_user_tag_embeddings = user_tag_embeddings * probs_users
        weighted_item_tag_embeddings = item_tag_embeddings * probs_items

        #Global average pooling of tags 
        final_user_tag_embeddings = torch.sum(weighted_user_tag_embeddings, dim=1)
        final_item_tag_embeddings = torch.sum(weighted_item_tag_embeddings, dim=1)

        return final_user_tag_embeddings, final_item_tag_embeddings

    def forward(self, batch):

        user_id_embed = self.user_id_embedding(batch['users'])
        item_id_embed = self.item_id_embedding(batch['items'])
            
        user_tag_embeddings, item_tag_embeddings = self.get_tag_embeddings(batch['user_tag_weights'], batch['user_tag_ids'], batch['item_tag_ids'], batch['item_tag_weights'])

        user_embedding =torch.cat((user_id_embed, user_tag_embeddings), dim=1)
        item_embedding = torch.cat((item_id_embed, item_tag_embeddings), dim=1)

        #Throw user embedding into user tower
        user = self.user_tower(user_embedding)
        #Throw item embedding into item tower
        item = self.item_tower(item_embedding)
        
        #Takes in user and item, multiplies them element wise then sums out the embedding dimension 
        dot_product = torch.einsum('bd,bd->b', user, item)

        #Sigmoid to squash dot prodcut between 0 and 1 to align with normalized votes
        pred = F.sigmoid(dot_product)

        return pred
     