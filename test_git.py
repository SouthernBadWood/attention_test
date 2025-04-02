import torch
import math
import torch.nn as nn

#input_shape = batchsize, seq_len, dim
class model(nn.Module):
    def __init__(self,dim,hiddin_dim,drop_rate=0.1,heads=8):
        super(model, self).__init__()
        self.hidden_dim = hiddin_dim
        self.drop_rate = drop_rate
        self.dim = dim
        self.heads = heads
        self.qkv_proj = nn.Linear(dim,hiddin_dim*3)
        self.drop_layer = nn.Dropout(p = drop_rate)
    def forward(self,x):
        Q,K,V = self.qkv_proj(x).chunk(3,dim=-1)
        attention_weight = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(self.hidden_dim)
        attention_weight = torch.softmax(attention_weight,dim=-1)
        attention_weight = self.drop_layer(attention_weight)
        attention_output = torch.matmul(attention_weight,V)
        return attention_output

if __name__ == "__main__":
    x = torch.randn(1,10,64) # batch_size=1, seq_len=10, hidden_dim=64
    model = model(dim=64,hiddin_dim=128,drop_rate=0.1)
    output = model(x)
    print(output.shape)  # should be (1, 10, 64)

