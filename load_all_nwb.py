from config import Allen_Brain_Observatory_Config
config=Allen_Brain_Observatory_Config()
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file=config.data_loc+'nwb_Aug17/manifest.json')
import time
import pprint
import pandas as pd
import numpy as np
import sys
sys.stdout.flush()
from tqdm import tqdm
#python load_all_nwb.py 0 597 2>&1 | tee -a load_log_all.txt

#def main():
print "Start ..."
df = pd.read_csv('Aug17_update_query.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
exps=boc.get_ophys_experiments(experiment_container_ids=exp_con_ids)
start = 0
end =10
if len(sys.argv) >1:
    start=int(sys.argv[1])
    end=int(sys.argv[2])
idx_range =np.arange(start,end)
for idx in tqdm(
            idx_range,
            desc='Download Aug update nwb files',
            total=len(idx_range)):
    sys.stdout.write("====================================================================")
    exp=exps[idx]
    ptxt="IDX[%g,%g) = %g, experiment container ID = %s, nwb file ID = %s"%(start,end,idx,exp['experiment_container_id'],exp['id'])
    print ptxt
    sys.stdout.write(ptxt)
    download_time = time.time()
    data_set = boc.get_ophys_experiment_data(exp['id']) 
    sys.stdout.write("-------------------------------------------------------------------")
    timetxt="Download time : %s"%(time.time() - download_time)
    sys.stdout.write(timetxt)
    sys.stdout.write("-------------------------------------------------------------------")
    pprint.pprint(exp)
    sys.stdout.write("====================================================================")
    

#if __name__ == "__main__":
    #main()