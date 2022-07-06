# -*- coding: utf-8 -*-
import os
import PIL
import PIL.Image,PIL.ImageDraw
import torch
import torchvision.transforms.functional as torchvision_F
from easydict import EasyDict as edict
import tqdm
import numpy as np
import torch.nn.functional as torch_F
import util
import warpnew as warp
import imageio


class Model():
  def __init__(self,opt):
        super().__init__()

  def load_data(self,opt):
    img_path = "/mnt/home/hhutton/barfplanar/runs/summer/0_seed3/data/images/"
    img_list = os.listdir(img_path)
    img_dict = {}
    for x in range(1,len(img_list)+1):
      img = PIL.Image.open(img_path+'img{}.png'.format(x)).convert('RGB')
      print('DIVIDE BY 4')
      new_width  = img.size[0]//4
      new_height = img.size[1]//4
      if new_width%2!=0:
          new_width-=1
      if new_height%2 !=0:
          new_height-=1
      img = img.resize((new_width, new_height), PIL.Image.ANTIALIAS)
      img_dict["img{0}".format(x)] = torchvision_F.to_tensor(img).to(opt.device)
      print('image shape: ',img_dict["img{0}".format(x)].shape)
    self.img_dict = img_dict

  def build_networks(self,opt):
    print("building networks...")
    self.graph = Graph(opt).to(opt.device)
    self.graph.warp_param = torch.nn.Embedding(opt.batch_size,opt.warp.dof).to(opt.device)
    torch.nn.init.zeros_(self.graph.warp_param.weight)

  def setup_optimizer(self,opt):
    print("setting up optimizers...")
    optim_list = [
          dict(params=self.graph.neural_image.parameters(),lr=opt.optim.lr),
          dict(params=self.graph.warp_param.parameters(),lr=opt.optim.lr_warp)
      ]
    optimizer = getattr(torch.optim,opt.optim.algo)
    #self.optim = optimizer(optim_list,weight_decay=1e-8)
    self.optim = optimizer(optim_list)
    if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

  def train_iteration(self,opt,var,loader):
    self.optim.zero_grad()
    for idx,img in enumerate(self.img_dict.keys()):
        _,opt.H_crop,opt.W_crop = self.img_dict[img].shape
        var[img] = self.graph.forward(opt,var[img],idx)
        var[img]['image_pert'] = self.img_dict[img]
        loss = self.graph.compute_loss(opt,var[img])
        loss.render.backward()
    self.optim.step()
    if opt.optim.sched:
      self.sched.step()
    self.it+=1
    loader.set_postfix(it=self.it,loss="{:.3f}".format(loss.render))
    self.graph.neural_image.progress.data.fill_(self.it/opt.max_iter)
    return loss

  def train(self,opt):
    # before training
    print("TRAINING START")
    self.ep = self.it = self.vis_it = 0
    var = edict()
    for img in self.img_dict.keys():
      var[img] = {}
    self.graph.train()
    # train
    #var = util.move_to_device(var,opt.device)
    loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
    for it in loader:
      # train iteration
      loss = self.train_iteration(opt,var,loader)
    print("TRAINING DONE")

  def predict_entire_image(self,opt):
    #xy_grid = warp.get_normalized_pixel_grid(opt)[:1]
    xy_grid = warp.get_normalized_pixel_grid(opt)
    print('xy_grid: ',xy_grid.shape)
    rgb = self.graph.neural_image.forward(opt,xy_grid) # [B,HW,3]
    print('rgb shape: ', rgb)
    #image = rgb.view(self.H,self.W,3).detach().cpu().numpy()
    image = rgb.view(opt.H,opt.W,3).detach().cpu().permute(2,0,1).permute(1,2,0).numpy()
    destination = opt.output_path+'/pred.png'
    imageio.imwrite(destination, im=image)


class Graph(torch.nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.neural_image = NeuralImageFunction(opt)

    def forward(self,opt,var,idx):
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt,xy_grid,self.warp_param.weight[idx])
        print(xy_grid_warped.shape)
        # render images
        var.rgb_warped = self.neural_image.forward(opt,xy_grid_warped) # [HW,3]
        print('rgb_warped:',var.rgb_warped)
        #var.rgb_warped_map = var.rgb_warped.view(opt.H_crop,opt.W_crop,3).permute(2,0,1) # [3,H,W]
        return var

    def l1_loss(self,pred,label=0):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()

    def MSE_loss(self,pred,label=0):
        loss = (pred.contiguous()-label)**2
        return loss.mean()

    def compute_loss(self,opt,var):
        loss = edict()
        print('image pert: ',var.image_pert.shape)
        image_pert = var.image_pert.view(3,opt.H_crop*opt.W_crop).permute(1,0)
        if opt.loss_type == 'mse':
          loss.render = self.MSE_loss(var.rgb_warped,image_pert)
        elif opt.loss_type == 'l1':
          loss.render = self.l1_loss(var.rgb_warped,image_pert)
        return loss


class NeuralImageFunction(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def define_network(self,opt):
        input_2D_dim = 2+4*opt.arch.posenc.L_2D if opt.arch.posenc else 2
        # point-wise RGB prediction
        self.mlp = torch.nn.ModuleList() 
        #L = get_layer_dims(self.layers)
        L = util.get_layer_dims(opt.arch.layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_2D_dim
            if li in opt.arch.skip: k_in += input_2D_dim
            linear = torch.nn.Linear(k_in,k_out)
            if opt.barf_c2f and li==0:
                # rescale first layer init (distribution was for pos.enc. but only xy is first used)
                scale = np.sqrt(input_2D_dim/2.)
                linear.weight.data *= scale
                linear.bias.data *= scale
            self.mlp.append(linear)

    def forward(self,opt,coord_2D): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,coord_2D,L=opt.arch.posenc.L_2D)
            points_enc = torch.cat([coord_2D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = coord_2D
        feat = points_enc
        print(feat.shape)
        # extract implicit features
        for li,layer in enumerate(self.mlp):
            if li in opt.arch.skip: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]
        return rgb

    def positional_encoding(self,opt,input,L): # [B,...,N]
      shape = input.shape
      freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
      spectrum = input[...,None]*freq # [B,...,N,L]
      sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
      input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
      input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
      # coarse-to-fine: smoothly mask positional encoding for BARF
      if opt.barf_c2f is not None:
          # set weights for different frequency bands
          start,end = opt.barf_c2f
          alpha = (self.progress.data-start)/(end-start)*L
          k = torch.arange(L,dtype=torch.float32,device=opt.device)
          weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
          # apply weights
          shape = input_enc.shape
          input_enc = (input_enc.view(-1,L)*weight).view(*shape)
      return input_enc


