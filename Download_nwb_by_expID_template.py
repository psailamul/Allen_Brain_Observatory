# Download nwb files 
DATA_LOC = '/media/data/pachaya/AllenData/'   
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file=DATA_LOC+'boc/manifest.json')
import pprint
import time
import sys
#########################################################################
# Main script
########################################################################

def main():
    start_time = time.time()
    ec_id=511510650
    if len(sys.argv) > 1:
        for ii in range(1,len(sys.argv)):
            arg = sys.argv[ii]
            print(arg)
            exec(arg)
    print("=======================================================================")
    exps = boc.get_ophys_experiments(experiment_container_ids=[ec_id])
    print("Experiments for experiment_container_id %d: %d\n" % (ec_id, len(exps)))
    pprint.pprint(exps)
    print("=======================================================================\n")
    for e_id in range(len(exps)):
        print("    loop#%d : Downloading Experiment ID %d ......" % (e_id,exps[e_id]['id']))
        download_time = time.time()
        data_set = boc.get_ophys_experiment_data(exps[e_id]['id'])  
        pprint.pprint(data_set.get_metadata())
        print("    Download complete: Time %s" %(time.time() - download_time))
        print("=======================================================================")
    print("%d experiments in experiment container id %d are downloaded......" % (len(exps),ec_id))
    print("Total run time %s" %(time.time() - start_time))

if __name__ == "__main__":
    main()
"""
511507650
511509529
511510650
511510670
511510718
511510736
511510855
511510884
517328083
536323956
543677425
"""