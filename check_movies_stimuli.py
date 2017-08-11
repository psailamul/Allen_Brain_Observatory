# Check movies stimuli


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
import time

# This class uses a 'manifest' to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If 'manifest_file' is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_movie_frame(frame,movie):
	plt.imshow(movie[frame,:,:], cmap='gray')
	plt.axis('off')
	plt.title('frame %d' % frame)
	plt.show()

def plot_stimulus_table(stim_table, title):
    fstart = stim_table.start.min()
    fend = stim_table.end.max()
    
    fig = plt.figure(figsize=(15,1))
    ax = fig.gca()
    for i, trial in stim_table.iterrows():    
        x1 = float(trial.start - fstart) / (fend - fstart)
        x2 = float(trial.end - fstart) / (fend - fstart)            
        ax.add_patch(patches.Rectangle((x1, 0.0), x2 - x1, 1.0, color='r'))
    ax.set_xticks((0,1))
    ax.set_xticklabels((fstart, fend))
    ax.set_yticks(())
    ax.set_title(title)
    ax.set_xlabel("frames")


download_time = time.time()
data_set = boc.get_ophys_experiment_data(501498760)

# read in the natural movie one clip
movie = data_set.get_stimulus_template('natural_movie_one')
pprint.pprint(movie)
print "Download complete: Time %s" %(time.time() - download_time)

# display a random frame for reference
frame = 200
plt.imshow(movie[frame,:,:], cmap='gray')
plt.axis('off')
plt.title('frame %d' % frame)
plt.show()


data_set = boc.get_ophys_experiment_data(501498760)

# read in the stimulus table, which describes when a given frame is displayed
stim_table = data_set.get_stimulus_table('natural_movie_one')

# find out when a particular frame range is displayed
frame_range = [ 100, 120 ]
stim_table = stim_table[(stim_table.frame >= frame_range[0]) & (stim_table.frame <= frame_range[1])]

plot_stimulus_table(stim_table, "frames %d -> %d " % (frame_range[0], frame_range[1]))





data_set = boc.get_ophys_experiment_data(501498760)

# read in the stimulus table, which describes when a given frame is displayed
stim_table = data_set.get_stimulus_table('natural_movie_one')

# find out when a particular frame range is displayed
frame_range = [ 100, 120 ]
stim_table = stim_table[(stim_table.frame >= frame_range[0]) & (stim_table.frame <= frame_range[1])]

plot_stimulus_table(stim_table, "frames %d -> %d " % (frame_range[0], frame_range[1]))



### Start here 


# Find all of the experiments for an experiment container
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint

# This class uses a 'manifest' to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If 'manifest_file' is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.
DATA_LOC = "/media/data/pachaya/AllenData/"
boc = BrainObservatoryCache(manifest_file=DATA_LOC+'boc/manifest.json')

expsid = 576208803
exps = boc.get_ophys_experiments(experiment_container_ids=[expsid])
print("Experiments for experiment_container_id %d: %d\n" % (expsid, len(exps)))
pprint.pprint(exps)



data_set = boc.get_ophys_experiment_data(580095647) # Sess A : 580095647,  Sess B : 578220711, Sess C : 577663639
#select experiment
movie1 = data_set.get_stimulus_template('natural_movie_one')





