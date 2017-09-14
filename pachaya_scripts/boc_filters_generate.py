# Generate Filters at http://observatory.brain-map.org/visualcoding/search/cell_list  --- select the filter_json definition part 

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import json

# ============================================================ 
# paste the filter_json here
# ============================================================           
filter_json = """
[
    {
        "field": "p_ns",
        "op": "<=",
        "value": 0.05
    },
    {
        "field": "reliability_nm1_a",
        "op": "between",
        "value": [
            0.01,
            1
        ]
    },
    {
        "field": "reliability_nm1_b",
        "op": "between",
        "value": [
            0.01,
            1
        ]
    },
    {
        "field": "reliability_nm1_c",
        "op": "between",
        "value": [
            0.01,
            1
        ]
    },
    {
        "field": "reliability_nm2",
        "op": "between",
        "value": [
            0.01,
            1
        ]
    },
    {
        "field": "reliability_nm3",
        "op": "between",
        "value": [
            0.01,
            1
        ]
    },
    {
        "field": "rf_chi2_lsn",
        "op": "<=",
        "value": 0.05
    },
    {
        "field": "area",
        "op": "in",
        "value": [
            "VISp"
        ]
    },
    {
        "field": "imaging_depth",
        "op": "in",
        "value": [
            175
        ]
    },
    {
        "field": "all_stim",
        "op": "is",
        "value": true
    }
]
"""
# ============================================================ 
# End of snippet
# ============================================================ 
FNAME = 'filters_VISp_175_sigNS_nonzeroNM_reliableRF.pkl'
filters = json.loads(filter_json)
import cPickle as pickle
save_at = open(FNAME,'wb')
pickle.dump(filters, save_at)
save_at.close()
#boc = BrainObservatoryCache(manifest_file=config.data_loc+"brain_observatory/manifest.json")
#cells = boc.get_cell_specimens(filters=filters)
        
