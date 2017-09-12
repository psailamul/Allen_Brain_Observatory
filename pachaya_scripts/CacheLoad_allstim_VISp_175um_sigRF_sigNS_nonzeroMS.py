from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import json

DATA_LOC = '/media/data/pachaya/AllenData/' 
"""
Filters : VISp, 
			depth = 175 um, 
			has all stimuli, 
			Receptive field Chi^2 <= 0.0, 
			P value - natural scenes <= 0.05, 
			reliability of all the natural  movies >= 0.1 (nonzero) 
			= 221 cells
http://observatory.brain-map.org/visualcoding/search/cell_list?p_ns=0.05&reliability_nm1_a=0.01,1&reliability_nm1_b=0.01,1&reliability_nm1_c=0.01,1&reliability_nm2=0.01,1&reliability_nm3=0.01,1&rf_chi2_lsn=0.05&area=VISp&imaging_depth=175&all_stim=true&sort_field=p_ns&sort_dir=asc

"""

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
       
filters = json.loads(filter_json)
         
boc = BrainObservatoryCache(manifest_file=DATA_LOC+"brain_observatory/manifest.json")
cells = boc.get_cell_specimens(filters=filters)