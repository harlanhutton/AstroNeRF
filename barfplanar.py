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

class File():
    def __init__(self,opt):
        self.file = open(opt.output_path+'/about.txt','a')
    def write(self,info):
        self.file.write(info+"\n")
    def close(self):
        self.file.close()

class Model():
  def __init__(self,opt):
        super().__init__()

  def load_data(self,opt):
    self.gt = PIL.Image.open(opt.data_path+'/gt.jpg')
    self.gt_tnsr = torchvision_F.to_tensor(self.gt).to(opt.device)
    img_list = os.listdir(opt.data_path+'/images/')
    opt.batch_size = len(img_list)
    self.img_dict = {}

    for x in range(len(img_list)):
        self.img_dict["img{0}".format(x)] = PIL.Image.open(opt.data_path+'/images/'+img_list[x])
        self.img_dict["img{0}".format(x)] = torchvision_F.to_tensor(self.img_dict["img{0}".format(x)]).to(opt.device)

    self.image_total = torch.stack(list(self.img_dict.values()))

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
    self.optim = optimizer(optim_list,weight_decay=1e-8)
    #self.optim = optimizer(optim_list)
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

  def generate_warp_perturbation(self,opt):
        # pre-generate perturbations (translational noise + homography noise)
        warp_pert_all = torch.zeros(opt.batch_size,opt.warp.dof,device=opt.device)
        trans_pert = [(0,0)] + [(round(x,2),round(y,2)) for x in np.linspace(-opt.warp.noise_t,opt.warp.noise_t,opt.batch_size)
                                                        for y in np.linspace(-opt.warp.noise_t,opt.warp.noise_t,opt.batch_size)
                                                        if abs(x)==abs(y) and abs(x)!=0 and abs(y)!=0]
        def create_random_perturbation():
            warp_pert = torch.randn(opt.warp.dof,device=opt.device)*opt.warp.noise_h
            warp_pert[0] += trans_pert[i][0]
            warp_pert[1] += trans_pert[i][1]
            return warp_pert
        for i in range(opt.batch_size):
            warp_pert = create_random_perturbation()
            while not warp.check_corners_in_range(opt,warp_pert[None]):
                warp_pert = create_random_perturbation()
            warp_pert_all[i] = warp_pert
        if opt.warp.fix_first:
            warp_pert_all[0] = 0
        # create warped image patches
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt,xy_grid,warp_pert_all)
        xy_grid_warped = xy_grid_warped.view([opt.batch_size,opt.H_crop,opt.W_crop,2])
        xy_grid_warped = torch.stack([xy_grid_warped[...,0]*max(opt.H,opt.W)/opt.W,
                                      xy_grid_warped[...,1]*max(opt.H,opt.W)/opt.H],dim=-1)
        
        
        crops, corners_all = self.visualize_patches(opt,warp_pert_all)
        transparency = 0.35 # Degree of transparency, 0-100%
        opacity = int(255 * transparency)
        img_dict = {}
        for i in range(opt.batch_size):
          this_im = self.image_total[i]
          image_pil = torchvision_F.to_pil_image(this_im).convert("RGBA")
          draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
          draw = PIL.ImageDraw.Draw(draw_pil)
          avg_x = torch.mean(corners_all[i][:,0]).cpu().numpy()
          avg_y = torch.mean(corners_all[i][:,1]).cpu().numpy()
          shape = random.choice(['ellipse','rectangle','polygon'])
          # 25,35 for small, 40,50 for big
          shape_size  = random.randint(35,45) # random size
          rand_loc = random.choice([-1,1])*np.random.randint(0,opt.W_crop//3-shape_size//2) # location in crop
          if shape == 'ellipse':
            stretch = random.randint(1,20) # random stretch
            bbox = (avg_x+rand_loc,avg_y+rand_loc,avg_x+rand_loc+shape_size,avg_y+rand_loc+shape_size+stretch) # coords
            draw.ellipse(bbox,fill=tuple(np.append(self.box_colors[i],opacity)))
          elif shape == 'rectangle':
            bbox = (avg_x+rand_loc,avg_y+rand_loc,avg_x+rand_loc+shape_size,avg_y+rand_loc+shape_size)
            draw.rectangle(bbox,fill=tuple(np.append(self.box_colors[i],opacity)))
          elif shape == 'polygon':
            # bbox = (avg_x+rand_loc,avg_y+rand_loc,
            #         avg_x+rand_loc+shape_size,avg_y+rand_loc+shape_size,
            #         avg_x+shape_size,avg_y-shape_size)
            # bbox = (avg_x+rand_loc,avg_y+rand_loc,
            #         avg_x+2*rand_loc+shape_size,avg_y+rand_loc+rand_loc+shape_size,
            #         avg_x-2*rand_loc-shape_size,avg_y-2*rand_loc-shape_size)
            bbox = (avg_x,avg_y+rand_loc,
                    avg_x,avg_y+rand_loc+shape_size,
                    avg_x-shape_size,avg_y+rand_loc+shape_size)
            draw.polygon(bbox,fill=tuple(np.append(self.box_colors[i],opacity)))
          image_pil.alpha_composite(draw_pil)
          image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
          img_dict['img{}'.format(i)] = image_tensor
        self.image_total = torch.stack(list(img_dict.values()))
        image_raw_batch = self.image_total.to(opt.device)
        image_pert_all = torch_F.grid_sample(image_raw_batch,xy_grid_warped,align_corners=False)
        return warp_pert_all,image_pert_all

  def train_iteration(self,opt,var,loader):
    # before train iteration
    self.timer.it_start = time.time()
    # train iteration
    self.optim.zero_grad()
    var = self.graph.forward(opt,var)
    loss = self.graph.compute_loss(opt,var)
    loss = self.summarize_loss(opt,loss)
    loss.all.backward()
    self.optim.step()
    if opt.optim.sched:
      self.sched.step()
    print(self.it)
    #if self.it%1000==0: self.visualize(opt)
    self.timer.it_end = time.time()
    self.it += 1
    self.timer.it_end = time.time()
    loader.set_postfix(it=self.it,loss="{:.3f}".format(loss.all))
    lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
    util.update_timer(opt,self.timer,self.it)
    log.loss_train(opt,self.it,lr,loss.all,self.timer)
    self.graph.neural_image.progress.data.fill_(self.it/opt.max_iter)
    return loss

  def train(self,opt,file):
    # before training
    print("TRAINING START")
    self.timer = edict(start=time.time(),it_mean=None)
    self.ep = self.it = self.vis_it = 0
    self.graph.train()
    var = edict(idx=torch.arange(opt.batch_size))
    self.warp_pert,var.image_pert = self.generate_warp_perturbation(opt)
    # train
    var = util.move_to_device(var,opt.device)
    loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
    # visualize initial state
    var = self.graph.forward(opt,var)
    self.visualize(opt)
    for it in loader:
      # train iteration
      loss = self.train_iteration(opt,var,loader)
      if opt.warp.fix_first:
        self.graph.warp_param.weight.data[0] = 0
    print("TRAINING DONE")
    file.write("time:{}".format(util.blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*util.get_time(self.timer.elapsed)),bold=True)))


  # def add_obstructions(self,opt,warp_param):
  #   crops, corners_all = self.visualize_patches(opt,warp_param)
  #   transparency = 0.5 # Degree of transparency, 0-100%
  #   opacity = int(255 * transparency)
  #   img_dict = {}

  #   for i in range(opt.batch_size):
  #     this_im = crops[i]
  #     image_pil = torchvision_F.to_pil_image(this_im).convert("RGBA")
  #     draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
  #     draw = PIL.ImageDraw.Draw(draw_pil)
  #     avg_x = torch.mean(corners_all[i][:,0]).cpu().numpy()
  #     avg_y = torch.mean(corners_all[i][:,1]).cpu().numpy()
  #     shape = random.choice(['ellipse','rectangle','polygon'])
  #     shape_size  = random.randint(10,20) # random size
  #     rand_loc = random.choice([-1,1])*np.random.randint(0,opt.W_crop//2-shape_size) # location in crop

  #     if shape == 'ellipse':
  #       stretch = random.randint(1,20) # random stretch
  #       bbox = (avg_x+rand_loc,avg_y+rand_loc,avg_x+rand_loc+shape_size,avg_y+rand_loc+shape_size+stretch) # coords
  #       draw.ellipse(bbox,fill=tuple(np.append(self.box_colors[i],opacity)))
  #     elif shape == 'rectangle':
  #       bbox = (avg_x+rand_loc,avg_y+rand_loc,avg_x+rand_loc+shape_size,avg_y+rand_loc+shape_size)
  #       draw.rectangle(bbox,fill=tuple(np.append(self.box_colors[i],opacity)))
  #     # elif shape == 'polygon':
  #     #   bbox = (avg_x+rand_loc,avg_y+rand_loc,
  #     #           avg_x+rand_loc+shape_size,avg_y+rand_loc+shape_size,
  #     #           avg_x+shape_size,avg_y-shape_size-rand_loc)
  #     elif shape == 'polygon':
  #       bbox = (avg_x+rand_loc,avg_y+rand_loc,
  #               avg_x-rand_loc-shape_size,avg_y-rand_loc-shape_size,
  #               avg_x+shape_size,avg_y+shape_size+rand_loc)
  #       draw.polygon(bbox,fill=tuple(np.append(self.box_colors[i],opacity)))
      
  #     image_pil.alpha_composite(draw_pil)
  #     image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
  #     img_dict['img{}'.format(i)] = image_tensor

  #   total = torch.stack(list(img_dict.values()))
  #   self.image_total = total
  #   return total


  def visualize_patches(self,opt,warp_param):
    img_dict = {}
    self.box_colors =["#" + "%06x" % random.randint(0, 0xFFFFFF) for _ in range(opt.batch_size)]
    self.box_colors = list(map(util.colorcode_to_number,self.box_colors))
    self.box_colors = np.array(self.box_colors).astype(int)
    corners_all = warp.warp_corners(opt,warp_param)
    corners_all[...,0] = (corners_all[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
    corners_all[...,1] = (corners_all[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5

    for i in range(opt.batch_size):
      this_im = self.image_total[i]
      image_pil = torchvision_F.to_pil_image(this_im).convert("RGBA")
      draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
      draw = PIL.ImageDraw.Draw(draw_pil)
      P = [tuple(float(n) for n in corners_all[i][j]) for j in range(4)]
      draw.line([P[0],P[1],P[2],P[3],P[0]],fill=(255,255,255),width=3)
      image_pil.alpha_composite(draw_pil)
      image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
      img_dict['img{}'.format(i)] = image_tensor

    total = torch.stack(list(img_dict.values()))
    return total, corners_all

  
  def generate_final_boxes(self,opt,warp_param):
    img_dict = {}
    corners_all = warp.warp_corners(opt,warp_param)
    corners_all[...,0] = (corners_all[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
    corners_all[...,1] = (corners_all[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5

    for i in range(opt.batch_size):
      this_im = self.image_total[i]
      image_pil = torchvision_F.to_pil_image(this_im).convert("RGBA")
      draw_pil = PIL.Image.new("RGBA",image_pil.size,(255,255,255))
      draw = PIL.ImageDraw.Draw(draw_pil)
      P = [tuple(float(n) for n in corners_all[i][j]) for j in range(4)]
      draw.line([P[0],P[1],P[2],P[3],P[0]],fill=(0,0,0),width=3)
      #image_pil.alpha_composite(draw_pil)
      image_tensor = torchvision_F.to_tensor(draw_pil.convert("RGB"))
      img_dict['img{}'.format(i)] = image_tensor

    total = torch.stack(list(img_dict.values()))
    return total


  def visualize(self,opt):
    if self.it == 0:
      frames, _ = self.visualize_patches(opt,self.warp_pert)
      #self.frames_GT = frames
      self.it += 1
    else:
      frames, _ = self.visualize_patches(opt,self.graph.warp_param.weight)
    for i in range(opt.batch_size):
      this_im = frames[i].permute(1,2,0)
      image = this_im.detach().cpu().numpy()
      newdir = opt.output_path+'/viz/iter{}/'.format(self.it)
      os.makedirs(newdir,exist_ok=True)
      destination = newdir+'img'+str(i)+'.png'
      imageio.imwrite(destination, im=image)
    if self.it == opt.max_iter:
      frames = self.generate_final_boxes(opt,self.graph.warp_param.weight)
      for i in range(opt.batch_size):
        this_im = frames[i].permute(1,2,0)
        image = this_im.detach().cpu().numpy()
        newdir = opt.output_path+'/viz/boxpreds/'
        os.makedirs(newdir,exist_ok=True)
        destination = newdir+'img'+str(i)+'.png'
        imageio.imwrite(destination, im=image)
  


  def predict_entire_image(self,opt,file):
    xy_grid = warp.get_normalized_pixel_grid(opt)[:1]
    rgb = self.graph.neural_image.forward(opt,xy_grid) # [B,HW,3]
    #image = rgb.view(self.H,self.W,3).detach().cpu().numpy()
    image = rgb.view(opt.H,opt.W,3).detach().cpu().permute(2,0,1).permute(1,2,0).numpy()
    final_pred = rgb.view(3,opt.H,opt.W)
    loss = edict()
    #print(self.graph.MSE_loss(final_pred,self.gt_tnsr))
    loss.render = self.graph.MSE_loss(final_pred,self.gt_tnsr)
    final_psnr = -10*loss.render.log10().item()
    print('FINAL PSNR - GROUND TRUTH COMPARED TO FINAL PRED: {}'.format(final_psnr))
    log.loss(loss.render)
    destination = opt.output_path+'/pred.png'
    imageio.imwrite(destination, im=image)
    file.write("final pred loss compared to gt: {}".format(loss))
    file.write("final psnr: {}".format(final_psnr))
    file.close()



class Graph(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.neural_image = NeuralImageFunction(opt)

    def forward(self,opt,var):
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt,xy_grid,self.warp_param.weight)
        # render images
        var.rgb_warped = self.neural_image.forward(opt,xy_grid_warped) # [B,HW,3]
        var.rgb_warped_map = var.rgb_warped.view(opt.batch_size,opt.H_crop,opt.W_crop,3).permute(0,3,1,2) # [B,3,H,W]
        return var

    def l1_loss(self,pred,label=0):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()

    def MSE_loss(self,pred,label=0):
        loss = (pred.contiguous()-label)**2
        return loss.mean()

    def compute_loss(self,opt,var):
        loss = edict()
        image_pert = var.image_pert.view(opt.batch_size,3,opt.H_crop*opt.W_crop).permute(0,2,1)
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


