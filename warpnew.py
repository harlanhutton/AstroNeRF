import torch

def get_normalized_pixel_grid(opt):
    y_range = ((torch.arange(opt.H,dtype=torch.float32,device=opt.device)+0.5)/opt.H*2-1)*(opt.H/max(opt.H,opt.W))
    x_range = ((torch.arange(opt.W,dtype=torch.float32,device=opt.device)+0.5)/opt.W*2-1)*(opt.W/max(opt.H,opt.W))
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    #xy_grid = xy_grid.repeat(opt.batch_size,1,1) # [B,HW,2]
    return xy_grid

def get_normalized_pixel_grid_crop(opt):
    y_crop = (opt.H//2-opt.H_crop//2,opt.H//2+opt.H_crop//2)
    x_crop = (opt.W//2-opt.W_crop//2,opt.W//2+opt.W_crop//2)
    y_range = ((torch.arange(*(y_crop),dtype=torch.float32,device=opt.device)+0.5)/opt.H*2-1)*(opt.H/max(opt.H,opt.W))
    x_range = ((torch.arange(*(x_crop),dtype=torch.float32,device=opt.device)+0.5)/opt.W*2-1)*(opt.W/max(opt.H,opt.W))
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    #xy_grid = xy_grid.repeat(opt.batch_size,1,1) # [B,HW,2]
    return xy_grid

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def warp_grid(opt,xy_grid,warp):
    xy_grid_hom = to_hom(xy_grid)
    warp_matrix = lie.sl3_to_SL3(warp)
    warped_grid_hom = xy_grid_hom@warp_matrix.transpose(-2,-1)
    warped_grid = warped_grid_hom[...,:2]/(warped_grid_hom[...,2:]+1e-8) # [HW,2]
    return warped_grid


class Lie():
    def sl3_to_SL3(self,h):
        # homography: directly expand matrix exponential
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
        h1,h2,h3,h4,h5,h6,h7,h8 = h.chunk(8,dim=-1)
        A = torch.stack([torch.cat([h5,h3,h1],dim=-1),
                         torch.cat([h4,-h5-h6,h2],dim=-1),
                         torch.cat([h7,h8,h6],dim=-1)],dim=-2)
        H = A.matrix_exp()
        return H

lie = Lie()