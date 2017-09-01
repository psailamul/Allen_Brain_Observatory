
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from config import Allen_Brain_Observatory_Config

config=Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)

df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])

sess_with_mc_errors = []
for ec in tqdm(
        exp_con_ids,
        desc="Finding Motion_correction error",
        total=len(exp_con_ids)):
    exp_session = boc.get_ophys_experiments(experiment_container_ids=[ec])
    for sess in exp_session:
        data_set= boc.get_ophys_experiment_data(sess['id'])
        try:
            mc = data_set.get_motion_correction()
        except:
            print "Session %s : Error occur in motion_correction"%sess['id']
            sess_with_mc_errors.append(sess['id'])
            mc=None