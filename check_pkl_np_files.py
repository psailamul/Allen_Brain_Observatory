# Checking stimulus template  files
from config import Allen_Brain_Observatory_Config
cf=Allen_Brain_Observatory_Config()


from os import listdir
from os.path import isfile, join
mypath=cf.stimulus_template_loc

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#import glob
#pkl_list = glob.glob("%s*.pkl"%(mypath))

nm1name = 'natural_movie_one_template'
nm2name = 'natural_movie_two_template'
nm3name = 'natural_movie_three_template'
nsname = 'natural_scenes_template'

import numpy as npnm
import helper_funcs as hf

flist = ['natural_movie_one_template',
	'natural_movie_two_template',
	'natural_movie_three_template',
	'natural_scenes_template']


	for ff in flist:
		datpkl =hf.load_object(mypath+ff+'.pkl')
		datnp = np.load(mypath+ff+'.npy')
		[a,b,c] = datnp.shape
		test = np.equal(datnp,datpkl)
		sumcheck = np.sum(np.sum(np.sum(test))) == a*b*c
		print ff
		print sumcheck
