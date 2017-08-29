import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import allensdk.brain_observatory.stimulus_info as si
from config import Allen_Brain_Observatory_Config
config=Allen_Brain_Observatory_Config()
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
#boc = BrainObservatoryCache(manifest_file=config.data_loc+'nwb_Aug17/manifest.json')
boc = BrainObservatoryCache(manifest_file=config.manifest_file)

""" From Brain Observatory Monitor Notebook"""
""" To get the stimulus parameters shown on acquisition frame 1010:"""

nwb_dataset = boc.get_ophys_experiment_data(527550473)
nwb_dataset.get_stimulus_epoch_table()

params, template = nwb_dataset.get_stimulus(1010)
pd.Series(params[2])

"""This is what was on the moniter during that acquision frame:"""

img = plt.imshow(template, cmap=plt.cm.gray, interpolation='none')

#Sometimes it is nice to have this stimulus visualized exactly as it was shown on the monitor:
m = si.BrainObservatoryMonitor()
img_screen = m.lsn_image_to_screen(template)
m.show_image(img_screen, warp=True)

"""During a Brain Observatory experiment, the image on the screen (for example, above) 
is warped so that spherical coordinates could be displayed on a flat monitor. 
When an stimulus frame is displayed without this warping, it is useful to overlay 
a mask of the eventual maximal extent of the stimulus that will be shown on the monitor."""

m = si.BrainObservatoryMonitor()
img_screen = m.lsn_image_to_screen(template)
m.show_image(img_screen, warp=False, mask=True)

"""Using the map_stimulus_coordinate_to_monitor_coordinate function, 
we can assosicate a position on the monitor (eitherpre- or post-warp) 
between multiple types of stimuli, despite the fact the templates for these stimuli
 might have different sizes and aspect ratios."""
y0, x0 = (5,12)
y1, x1 = m.map_stimulus((y0, x0), si.LOCALLY_SPARSE_NOISE, si.NATURAL_MOVIE_ONE)
ym, xm = si.map_stimulus_coordinate_to_monitor_coordinate((y0, x0), 
                                                          (m.n_pixels_r, m.n_pixels_c), 
                                                          si.LOCALLY_SPARSE_NOISE)

img_lsn =  nwb_dataset.get_stimulus_template('locally_sparse_noise')[0,:,:]
img_movie =  nwb_dataset.get_stimulus_template('natural_movie_one')[0,:,:]


fig, ax = plt.subplots(2,2, figsize=(10,5))
m.show_image(m.natural_movie_image_to_screen(img_movie, origin='upper'), 
             ax=ax[0,1], 
             show=False, 
             origin='upper', 
             mask=True)
ax[0,1].plot([xm], [ym], 'r.')
ax[0,1].set_title('Natural Movie One, Monitor\n1200x1920')

m.show_image(m.lsn_image_to_screen(img_lsn, origin='upper'), 
             ax=ax[0,0], 
             show=False, 
             origin='upper', 
             mask=True)
ax[0,0].plot([xm], [ym], 'r.')
ax[0,0].set_title('Locally Sparse Noise, Monitor\n1200x1920')

ax[1,0].imshow(np.flipud(img_lsn), interpolation='none', cmap=plt.cm.gray, 
               extent=[0,img_lsn.shape[1],img_lsn.shape[0],0])
ax[1,0].plot([x0], [y0], 'r.')
ax[1,0].axes.get_xaxis().set_visible(False)
ax[1,0].axes.get_yaxis().set_visible(False)
ax[1,0].set_xlim((0, img_lsn.shape[1]-1))
ax[1,0].set_ylim((img_lsn.shape[0]-1, 0))
ax[1,0].set_title('Locally Sparse Noise, Template\n16x28')

ax[1,1].imshow(img_movie, interpolation='none', cmap=plt.cm.gray,
               extent=[0,img_movie.shape[1],img_movie.shape[0],0])
ax[1,1].plot([x1], [y1], 'r.')
ax[1,1].axes.get_xaxis().set_visible(False)
ax[1,1].axes.get_yaxis().set_visible(False)
ax[1,1].set_xlim((0, img_movie.shape[1]-1))
ax[1,1].set_ylim((img_movie.shape[0]-1, 0))
ax[1,1].set_title('Natural Movie One, Template\n304x608')


fig.tight_layout()

"""For example, in the top row of the figure above, the same position on the monitor (pre-warp) 
is marked in two different stimuli (left: locally sparse noise, right: natural movie). 
Below, the same point is reproduced on the template image for reach stimuli. 
You can see that in both images, the relative location of the marked point (red) is the same in both 
the pre-warp monitor image and the template image. This example demonstrates how to co-register locations 
between stimulus sets."""