from config import Allen_Brain_Observatory_Config
config=Allen_Brain_Observatory_Config()
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file=config.data_loc+'boc/manifest.json')
import time
import pprint
import pandas as pd
import numpy as np
    
        
#def main():
df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
exps=boc.get_ophys_experiments(experiment_container_ids=exp_con_ids)
#start = 1; end =199 #g13
#start = 199; end = 398 #x7
start = 398; end =len(exps) #3

for idx in np.arange(start,end):
    print('====================================================================')
    exp=exps[idx]
    print("IDX[%g,%g) = %g, experiment container ID = %s, nwb file ID = %s"%(start,end,idx,exp['experiment_container_id'],exp['id']))
    download_time = time.time()
    data_set = boc.get_ophys_experiment_data(exp['id']) 
    print('-------------------------------------------------------------------')
    print("Download time : %s"%(time.time() - download_time))
    print('-------------------------------------------------------------------')
    pprint.pprint(exp)
    print('====================================================================')
