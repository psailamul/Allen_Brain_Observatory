from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import json
DATA_LOC = '/media/data/pachaya/AllenData/'          
filter_json = """
[
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

