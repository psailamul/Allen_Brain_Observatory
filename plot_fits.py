import os
import sys
import argparse
import numpy as np
from glob import glob
from allen_config import Allen_Brain_Observatory_Config
from declare_datasets_loop import query_hp_hist 
from matplotlib import pyplot as plt
import pandas as pd
# from ggplot import *
import seaborn as sns


def plot_fits(
        experiment='760_cells_2017_11_04_16_29_09'):
    """Plot fits across the RF."""
    main_config = Allen_Brain_Observatory_Config()
    sys.path.append(main_config.cc_path)
    from db import credentials
    db_config = credentials.postgresql_connection()
    files = glob(
        os.path.join(
            main_config.multi_exps,
            experiment, '*.npz'))
    out_data, xs, ys, perfs, model_types = [], [], [], [], []
    for f in files:
        data = np.load(f)
        d = {
            'x': data['rf_data'].item()['on_center_x'],
            'y': data['rf_data'].item()['on_center_y'],
            # x: files['dataset_method'].item()['x_min'],
            # y: files['dataset_method'].item()['y_min'],
        }
        exp_name = {
            'experiment_name': data['dataset_method'].item()['experiment_name']}
        perf, mt = query_hp_hist(exp_name, pass_creds=db_config)
        if perf is None:
            print 'No fits for: %s' % exp_name['experiment_name']
        else:
            d['perf'] = perf
            d['max_val'] = np.max(perf)
            d['mt'] = mt
            out_data += [d]
            xs += [np.round(d['x'] * 10)]
            ys += [np.round(d['y'] * 10)]
            perfs += [np.max(d['perf'])]
            model_types += [mt]

    # Package as a df
    lcs = ['red', '#05D9F6', '#E1FA7']
    hcs = ['white', '#5011D1', '#F46FEE'] 
    xs = np.round(np.asarray(xs) * 10).astype(int)
    ys = np.round(np.asarray(ys) * 10).astype(int)
    perfs = np.asarray(perfs)
    model_types = np.asarray(model_types)
    umt, model_types_inds = np.unique(model_types, return_inverse=True)
    cps = [sns.cubehelix_palette(n_colors=100, as_cmap=True), sns.cubehelix_palette(n_colors=100, as_cmap=True, rot=-.4), sns.cubehelix_palette(n_colors=100, as_cmap=True, rot=.1, start=2.8)]
    # cps = ["Blues", "Reds", "Greens"]
    f, ax = plt.subplots()
    p = []
    for idx, (imt, cp) in enumerate(zip(umt, cps)):
        it_xs = xs[model_types == imt] + (idx * 1)
        it_ys = ys[model_types == imt] + (idx * 1)
        it_perfs = perfs[model_types == imt]
        df = pd.DataFrame(np.vstack((it_xs, it_ys, it_perfs)).transpose(), columns=['x', 'y', 'perf'])
        # p += [sns.lmplot(x='x', y='y', hue='perf', palette=cp, data=df, markers='s', fit_reg=False, x_jitter=1, y_jitter=1)]
        p += [ax.scatter(it_xs, it_ys, c=it_perfs, cmap=cp)]
        print (imt)
    [plt.colorbar(ip, ticks=np.linspace(-.1, 1, 100, endpoint=True)) for ip in p]
    ax.legend()
    plt.xlim([-100, 1500 * 10])
    plt.ylim([-100, 1000 * 10])
    # plt.xlabel('X-axis degrees of visual-angle')
    # plt.xlabel('Y-axis degrees of visual-angle')
    plt.show()
    # for idx, imt in enumerate(umt):
    #     it_xs = xs[mt == imt]
    #     it_ys = ys[mt == imt]
    #     it_perfs = perfs[mt == imt]
    #     df = pd.DataFrame(np.vstack((it_xs, it_ys, it_perfs)).transpose(), columns=['x', 'y', 'perf'])
    #     if idx == 0:
    #         p = ggplot(df, aes(x='x', y='y', color='perf'))
    #     else:
    #         p += ggplot(df, aes(x='x', y='y', color='perf'))
    #     p += geom_jitter()
    #     p += scale_color_gradient(low=lcs[idx], high=hcs[idx])
    # p += xlab('X-axis degrees of visual-angle')
    # p += ylab('Y-axis degrees of visual-angle')
    # p += xlim(low=0, high=1500)
    # p += ylim(low=0, high=1000)
    # p += ggtitle('V1 fits')
    # p.save('test.png')

    # df = pd.DataFrame(np.vstack((xs, ys, perfs)).transpose(), columns=['x', 'y', 'perf'])
    # p = ggplot(df, aes(x='x', y='y', color='perf')) +\
    # geom_point() +\
    # scale_color_gradient(low='#05D9F6', high='#5011D1') +\
    # xlab('X-axis degrees of visual-angle') +\
    # ylab('Y-axis degrees of visual-angle') +\
    # xlim(low=0, high=1500) +\
    # ylim(low=0, high=1000) +\
    # ggtitle('V1 fits')
    # p.save('test.png')
    # Plot results

    # f, ax = plt.subplots()
    # cmap = plt.get_cmap('magma')
    # ls = []
    # for idx, d in enumerate(out_data):
    #     x = np.round(d['x'] * 10)
    #     y = np.round(d['y'] * 10)
    #     l = plt.scatter(x, y, marker='s', c=cmap(idx*100))
    #     ls += [l]
    # plt.colorbar(ls)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot_fits',
        type=str,
        default='760_cells_2017_11_05_16_36_55',
        dest='experiment',
        help='Name of experiment.')
    args = parser.parse_args()
    plot_fits(**vars(args))

