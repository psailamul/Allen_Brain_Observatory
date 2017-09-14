###############################################################################################################
"""
CREATE TABLE cells (_id bigserial primary key, cell_specimen_id int, session varchar, drifting_gratings boolean, locally_sparse_noise boolean, locally_sparse_noise_four_deg boolean, locally_sparse_noise_eight_deg boolean, natural_movie_one boolean, natural_movie_two boolean, natural_movie_three boolean, natural_scenes boolean, spontaneous boolean, static_gratings boolean, specimen_recording_pointer varchar, traces_loc_pointer varchar, ROImask_loc_pointer varchar, stim_table_loc_pointer varchar)

CREATE TABLE rf (_id bigserial primary key, cell_specimen_id int, lsn_name varchar, experiment_container_id int, found_on boolean, found_off boolean, alpha float, number_of_shuffles int, on_distance float, on_area float, on_overlap float, on_height float, on_center_x float, on_center_y float, on_width_x float, on_width_y float, on_rotation float, off_distance float, off_area float, off_overlap float, off_height float, off_center_x float, off_center_y float, off_width_x float, off_width_y float, off_rotation float)

ALTER TABLE cells ADD CONSTRAINT unique_cells UNIQUE (cell_specimen_id, session , drifting_gratings , locally_sparse_noise , locally_sparse_noise_four_deg , locally_sparse_noise_eight_deg , natural_movie_one , natural_movie_two , natural_movie_three , natural_scenes , spontaneous , static_gratings , specimen_recording_pointer , traces_loc_pointer , ROImask_loc_pointer , stim_table_loc_pointer )

ALTER TABLE rf ADD CONSTRAINT unique_rfs UNIQUE (cell_specimen_id , lsn_name , experiment_container_id , found_on , found_off , alpha , number_of_shuffles , on_distance , on_area , on_overlap , on_height , on_center_x , on_center_y , on_width_x , on_width_y , on_rotation , off_distance , off_area , off_overlap, off_height , off_center_x , off_center_y , off_width_x , off_width_y , off_rotation )

"""
###############################################################################################################
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import matplotlib.pyplot as plt
from config import Allen_Brain_Observatory_Config
import numpy as np
from db import db
from helper_funcs import *
from matplotlib.patches import Ellipse
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

main_config=Allen_Brain_Observatory_Config()

def main():
    queries = [
        {  # Get a range
          'x_min': 9,
          'x_max': 50,
          'y_min': 9,
          'y_max': 20,
        },
        {  # Get all
          'x_min': -10000,
          'x_max': 10000,
          'y_min': -10000,
          'y_max': 10000,
        }
    ]
    queries_label = ['center x in (9,50), y in (9,20)','all cells with on RF']
    #import ipdb; ipdb.set_trace()
    all_data_dicts = db.get_cells_by_rf(queries)
    visual_space_h=np.floor(main_config.LSN_size_in_deg['height'])
    visual_space_w=np.floor(main_config.LSN_size_in_deg['width'])
    color='b' # b for on , r for off

    for data_dicts, label in zip(all_data_dicts,queries_label):
        fig = plt.figure()
        for dat in data_dicts:
            plt.scatter(dat['on_center_x'],dat['on_center_y'])
        plt.xlim(0,visual_space_w)
        plt.ylim(0,visual_space_h)
        plt.title(label)
        plt.show()

    #import ipdb; ipdb.set_trace()
    for data_dicts, label in zip(all_data_dicts,queries_label):
        ells=[]
        fig, ax = plt.subplots()
        ax.set_xlim(0,visual_space_w)
        ax.set_ylim(0,visual_space_h)
        import ipdb; ipdb.set_trace()
        for dat in data_dicts:
            xy = (dat['on_center_x'],dat['on_center_y'])
            width = 3 * np.abs(dat['on_width_x'])
            height = 3 * np.abs(dat['on_width_y'])
            angle = dat['on_rotation']
            if np.logical_not(any(np.isnan(xy))):
                ellipse = mpatches.Ellipse(xy, width=width, height=height, angle=angle, lw=2, edgecolor=color,
                                           facecolor=color,alpha=0.1)
                ax.add_artist(ellipse)
        import ipdb; ipdb.set_trace()
        plt.xlim(0,visual_space_w)
        plt.ylim(0,visual_space_h)
        plt.title(label)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
