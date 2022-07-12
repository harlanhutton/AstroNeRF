import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import make_lupton_rgb
import sys
#sys.path.append('/mnt/home/hhutton/miniconda3/envs/barf-env/lib/python3.9/site-packages/')
import artpop
import os
import random


class ArtPopGenerator():
    def __init__(self,opt):
        super().__init__()
        self.img_path = "{}/images".form
        at(opt.data_path)
        os.makedirs(self.img_path,exist_ok=True)

    def source(self,opt,seed):
        # if seed == 0:
        #     rand_pixscale = 0.2
        # else:
        #     rand_pixscale = random.choice(opt.obj_source.pixel_scale)
        # print('seed {}:'.format(seed),'pix scale: {}'.format(rand_pixscale))
        self.src = artpop.MISTSersicSSP(log_age=opt.obj_source.log_age,
                                        feh=opt.obj_source.feh,
                                        n=opt.obj_source.n,
                                        ellip=opt.obj_source.ellip,
                                        num_stars=opt.obj_source.num_stars,
                                        phot_system=opt.obj_source.phot_system,
                                        xy_dim=opt.obj_source.xy_dim,
                                        pixel_scale=opt.obj_source.pixel_scale,
                                        r_eff=opt.obj_source.r_eff_base*u.pc,
                                        theta=opt.obj_source.theta_base*u.deg,
                                        distance=opt.obj_source.distance_base*u.Mpc,
                                        random_state=opt.obj_source.random_state
        )
    
    def image(self,opt,seed):
        # if seed == 0:
        #     rand_noise = 0
        # else:
        #     rand_noise = random.choice(opt.imager.noises)
        self.imager = artpop.ArtImager(phot_system=opt.imager.phot_system,
                                       read_noise=self.rand_noise,
                                       diameter=opt.imager.diameter*u.m,
                                       random_state=opt.obj_source.random_state)

    def observation(self,opt,seed):
        # if seed == 0:
        #     rand_fwhm = 0.1
        # else:
        #     rand_fwhm = random.choice(opt.observe.fwhms)
        # print('seed {}:'.format(seed),'fwhm: {}'.format(rand_fwhm))
        psf = artpop.moffat_psf(fwhm=self.rand_fwhm*u.arcsec)
        self.obs_g = self.imager.observe(source=self.src,
                                         bandpass='LSST_g',
                                         psf=psf,
                                         exptime=opt.observe.exptime*u.min,
                                         sky_sb=opt.observe.sky_sb
        )
        self.obs_r = self.imager.observe(source=self.src,
                                         bandpass='LSST_r',
                                         psf=psf,
                                         exptime=opt.observe.exptime*u.min,
                                         sky_sb=opt.observe.sky_sb
        )
        self.obs_i = self.imager.observe(source=self.src,
                                         bandpass='LSST_i',
                                         psf=psf,
                                         exptime=opt.observe.exptime*u.min, 
                                         sky_sb=opt.observe.sky_sb
        )

    def save(self,opt):
        for i in range(opt.batch_size+1): # first image will be ground truth
            #res_choices = ['low','high']
            res = random.choice(['low','high'])

            if res == 'low':
                opt.obj_source.pixel_scale = 0.2
                self.rand_fwhm = random.choice([0.5, 1.0, 1.5])
                self.rand_noise = random.choice([300,400])
                self.rand_Q = random.choice([3,6])
                self.rand_stretch = 0.5
            else: 
                opt.obj_source.pixel_scale = 0.1
                self.rand_fwhm = 0.15
                self.rand_noise = random.choice([150,200])
                self.rand_Q = random.choice([2,4])
                self.rand_stretch = 0.2

            print('img {},'.format(i),
                  'res {},'.format(res),
                  'noise {},'.format(self.rand_noise),
                  'fwhm {},'.format(self.rand_fwhm),
                  'Q {}'.format(self.rand_Q),
                  'stretch {}'.format(self.rand_stretch))

            self.source(opt,seed=i)
            self.image(opt,seed=i)
            self.observation(opt,seed=i)
            rgb = make_lupton_rgb(self.obs_i.image, self.obs_r.image, self.obs_g.image, stretch=self.rand_stretch, Q=self.rand_Q)
            fig = plt.figure(frameon=False)
            fig.set_size_inches(opt.figsize[0], opt.figsize[1])
            ax = plt.Axes(fig, [0., 0., 1., 1.], )
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(rgb)
            if i == 0:
                fname = opt.data_path + '/gt.jpg'
            else:
                fname = self.img_path + '/img{}.jpg'.format(i)
            plt.savefig(fname, dpi=opt.dpi)


        