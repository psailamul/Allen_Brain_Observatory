
# coding: utf-8

# # Receptive Field Analysis
# This notebook demonstrates how to run the `brain_observatory.receptive_field_analysis` module.  This module uses a cell's responses to the locally sparse noise stimulus to characterize the spatial receptive field, including on and off subunits.  We highly recommend reading through the the stimulus analysis <a href='http://help.brain-map.org/display/observatory/Documentation]'>whitepaper</a> to understand the locally sparse noise stimulus and the analysis methodology.
# 
# Download this file in .ipynb format <a href='receptive_fields.ipynb'>here</a>.
# 
# First we import packages.

# In[2]:


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.visualization as rfvis
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf
import matplotlib.pyplot as plt



# Given a cell of interest, we now identify the experiment that contains the locally sparse noise stimulus and download its NWB file.  We also look in the NWB file to figure out the position/index of the cell that has the ID we're interested in.

# In[4]:


cell_specimen_id = 587377366
DATA_LOC = "/media/data/pachaya/AllenData/"
boc = BrainObservatoryCache(manifest_file=DATA_LOC+'boc/manifest.json')

exps = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id],
                                 stimuli=['locally_sparse_noise'])

data_set = boc.get_ophys_experiment_data(exps[0]['id'])

cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]

print("cell %d has index %d" % (cell_specimen_id, cell_index))


# ## Compute receptive fields
# The following method in the `receptive_field_analysis` module will characterize on and off receptive fields and perform a per-pixel significance test.

# In[ ]:


rf_data = rf.compute_receptive_field_with_postprocessing(data_set, 
                                                         cell_index, 
                                                         'locally_sparse_noise', 
                                                         alpha=0.5, 
                                                         number_of_shuffles=10000)


# ## Chi^2 significance map
# Per-pixel chi-square tests identify cells that show non-uniform distributions of responses across pixels.  The `receptive_field_analysis.visualization` module has function to plot that significance as a heat map.

# In[2]:


rfvis.plot_chi_square_summary(rf_data)
plt.show()


# ## Response-triggered stimulus field
# The response-triggered stimulus field shows, for a given pixel, how many trials contained a detected calcium event.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_rts_summary(rf_data, ax1, ax2)


# ## Blurred response-triggered stimulus field
# The RTS field is convolved with a Gaussian to pool the contributions of neighboring stimulus pixels.  

# In[6]:


fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_rts_blur_summary(rf_data, ax1, ax2)


# ## p value field
# 
# Per-pixel p-values are estimated from the blurred RTS field to understand the significance of the response to each pixel.  

# In[7]:


fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_p_values(rf_data, ax1, ax2)


# ## Significance mask
# 
# The significance mask is p-value field after applying a binary threshold to remove insignificant pixels.

# In[8]:


fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_mask(rf_data, ax1, ax2)


# ## Gaussian fit
# Each identified subunit of the on and off receptive fields are fit with a Gaussian in the `receptive_field_analysis.postprocessing` module.

# In[9]:


fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_gaussian_fit(rf_data, ax1, ax2)


# ## 4.65 vs 9.3 degree sparse noise stimuli
# Newer experiments switched from using a single locally sparse noise stimulus with 4.54 visual-degree pixels to two blocks of stimuli with different pixel sizes (a 4.65 degree block and an 9.3 degree block that are each half the length of the original 4.65-degree-only stimulus).  You can characterize the receptive fields from reponses to each stimulus block separately.

# In[10]:


cell_specimen_id = 559109414
exps = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id])
for exp in exps:
    print(boc.get_ophys_experiment_stimuli(exp['id']))


# This cell comes from an experiment that has the new 4.65 degree and 9.3 degree stimulus blocks.  Let's find the experiment that contains the 9.3 degree stimulus.
# 
# **Note:** the NWB files refer to these stimuli as `locally_sparse_noise_4deg` and `locally_sparse_noise_8deg` respectively.

# In[11]:


exp = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id],
                                stimuli=['locally_sparse_noise_8deg'])
data_set = boc.get_ophys_experiment_data(exps[0]['id'])
cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]
print("cell %d has index %d" % (cell_specimen_id, cell_index))


# Now we can run the receptive field analysis as before and see what this looks like.

# In[12]:


rf_data = rf.compute_receptive_field_with_postprocessing(data_set, 
                                                         cell_index, 
                                                         'locally_sparse_noise_8deg', 
                                                         alpha=0.5, 
                                                         number_of_shuffles=10000)


# In[13]:


rfvis.plot_chi_square_summary(rf_data)
plt.show()


# In[14]:


fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_rts_summary(rf_data, ax1, ax2)
