# -*- coding: utf-8 -*-
import os,time
import PIL
import PIL.Image,PIL.ImageDraw
import torch
import torchvision.transforms.functional as torchvision_F
from easydict import EasyDict as edict
import tqdm
import numpy as np
import torch.nn.functional as torch_F
import imageio
import random
import util
from util import log
import warp


class Model():
  def __init__(self,opt):
        super().__init__()

  def load_data(self,opt):
    img_list = os.listdir(opt.data_path+'/images/')
    opt.batch_size = len(img_list)
    self.img_dict = {}

    for x in range(len(img_list)):
        raw_img = PIL.Image.open(opt.data_path+'/images/'+img_list[x])
        self.img_dict["img{0}".format(x)] = torchvision_F.to_tensor(raw_img).to(opt.device)
    self.image_total = torch.stack(list(self.img_dict.values()))

  def build_networks(self,opt):
    self.graph = Graph(opt).to(opt.device)

  def setup_optimizer(self,opt):
    optim_list = [
          dict(params=self.graph.neural_image.parameters(),lr=opt.optim.lr)
      ]
    optimizer = getattr(torch.optim,opt.optim.algo)
    #self.optim = optimizer(optim_list,weight_decay=1e-8)
    self.optim = optimizer(optim_list)
    if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

  def summarize_loss(self,opt,loss):
    loss_all = 0.
    assert("all" not in loss)
    # weigh losses
    for key in loss:
        assert(key in opt.loss_weight)
        assert(loss[key].shape==())
        if opt.loss_weight[key] is not None:
            assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
            assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
            loss_all += 10**float(opt.loss_weight[key])*loss[key]
    loss.update(all=loss_all)
    return loss

  def train_iteration(self,opt,var,loader):
    # before train iteration
    self.timer.it_start = time.time()
    # train iteration
    self.optim.zero_grad()
    var.rgb = self.graph.forward(opt)
    loss = self.graph.compute_loss(opt,var)
    loss = self.summarize_loss(opt,loss)
    loss.all.backward()
    self.optim.step()
    if opt.optim.sched:
      self.sched.step()
    self.timer.it_end = time.time()
    self.it += 1
    self.timer.it_end = time.time()
    loader.set_postfix(it=self.it,loss="{:.3f}".format(loss.all))
    lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
    util.update_timer(opt,self.timer,self.it)
    log.loss_train(opt,self.it,lr,loss.all,self.timer)
    self.graph.neural_image.progress.data.fill_(self.it/opt.max_iter)
 
  def train(self,opt):
    # before training
    self.timer = edict(start=time.time(),it_mean=None)
    self.ep = self.it = self.vis_it = 0
    self.graph.train()
    var = edict(idx=torch.arange(opt.batch_size))
    var.images = self.image_total
    # train
    var = util.move_to_device(var,opt.device)
    loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
    var.rgb = self.graph.forward(opt)
    for _ in loader:
      # train iteration
      self.train_iteration(opt,var,loader)

  def predict_entire_image(self,opt):
    xy_grid = warp.get_normalized_pixel_grid(opt)[:1]
    rgb = self.graph.neural_image.forward(opt,xy_grid) # [B,HW,3]
    image = rgb.view(opt.H,opt.W,3).detach().cpu().permute(2,0,1).permute(1,2,0).numpy()
    destination = opt.output_path+'/pred.png'
    imageio.imwrite(destination, im=image)


class Graph(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.neural_image = NeuralImageFunction(opt)

    def forward(self,opt):
        xy_grid = warp.get_normalized_pixel_grid(opt)
        rgb = self.neural_image.forward(opt,xy_grid) # [B,HW,3]
        return rgb

    def l1_loss(self,pred,label=0):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()

    def MSE_loss(self,pred,label=0):
        loss = (pred.contiguous()-label)**2
        return loss.mean()

    def compute_loss(self,opt,var):
        loss = edict()
        images = var.images.view(opt.batch_size,3,opt.H*opt.W).permute(0,2,1)
        if opt.loss_type == 'mse':
          loss.render = self.MSE_loss(var.rgb,images)
        elif opt.loss_type == 'l1':
          loss.render = self.l1_loss(var.rgb,images)
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
        print('feat: ', feat.shape)
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


