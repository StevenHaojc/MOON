import torch
import torch.nn as nn
import torch.nn.functional as F

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)
        self.fc_z = nn.Linear(input_dim, output_dim)
    def forward(self, x, y, z):
        output = self.fc_x(x) + self.fc_y(y) + self.fc_z(z)
        out_x = torch.mm(x, torch.transpose(self.fc_x.weight, 0, 1)) + self.fc_x.bias
        out_y = torch.mm(y, torch.transpose(self.fc_y.weight, 0, 1)) + self.fc_y.bias
        out_z = torch.mm(z, torch.transpose(self.fc_z.weight, 0, 1)) + self.fc_z.bias
        out_x = torch.sigmoid(out_x)
        out_y = torch.sigmoid(out_y)
        out_z = torch.sigmoid(out_z)
        return out_x,out_y,out_z,output
    

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, num_organs =3, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim*num_organs, output_dim)
    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        output = self.fc_out(output)
        out_y = (torch.mm(y, torch.transpose(self.fc_out.weight[:, :512], 0, 1)) + self.fc_out.bias / 3)
        out_x = (torch.mm(x, torch.transpose(self.fc_out.weight[:, 512:1024], 0, 1)) + self.fc_out.bias / 3)
        out_z = (torch.mm(z, torch.transpose(self.fc_out.weight[:, 1024:], 0, 1)) + self.fc_out.bias / 3)
        out_x = torch.sigmoid(out_x)
        out_y = torch.sigmoid(out_y)
        out_z = torch.sigmoid(out_z)
        return out_x,out_y,out_z,output

class FilmFusion(nn.Module):
    def __init__(self, input_dim=512, dim=512, output_dim=100):
        super(FilmFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * dim)
        self.fc2 = nn.Linear(input_dim, 2 * dim)
        self.fc3 = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.dim = dim

    def forward(self, x, y, z):
        gamma1, beta1 = torch.split(self.fc1(x), self.dim, 1)
        gamma2, beta2 = torch.split(self.fc2(y), self.dim, 1)
        gamma3, beta3 = torch.split(self.fc3(z), self.dim, 1)
        
        combined_gamma = gamma1 * gamma2 * gamma3
        combined_beta = beta1 + beta2 + beta3
        
        modulated_features = combined_gamma * (x + y + z) + combined_beta
        output = self.fc_out(modulated_features)
        return x, y, z, output


if __name__ == '__main__':
    input_dim_x = 512
    input_dim_y = 512
    input_dim_z = 512

    batch_size = 4

    x = torch.randn(batch_size, input_dim_x)
    y = torch.randn(batch_size, input_dim_y)
    z = torch.randn(batch_size, input_dim_z)


    sum_fusion = SumFusion(output_dim=4)
    _,_,_,sum_output = sum_fusion(x, y, z)
    print("SumFusion output shape:", sum_output.shape)

    concat_fusion = ConcatFusion(output_dim=4)
    _,_,_,concat_output = concat_fusion(x, y, z)
    print("ConcatFusion output shape:", concat_output.shape)

    film = FilmFusion(output_dim=4)
    _,_,_,film_output = film(x, y, z)
    print("FiLM output shape:", film_output.shape)
