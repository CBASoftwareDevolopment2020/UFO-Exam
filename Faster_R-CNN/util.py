import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess


def get_system_info():
    # traverse the info
    info = subprocess.check_output(['systeminfo']).decode('utf-8').split('\n')
    new = []

    # arrange the string into clear info
    for item in info:
        new.append(str(item.split("\r")[:-1]))

    for i in new:
        print(i[2:-2])


def calc_stats(dataset: pd.DataFrame, dataname: str):
    mean = float(np.round(np.mean(dataset), 3))
    median = float(np.round(np.median(dataset), 3))
    min_value = float(np.round(dataset.min(), 3))
    max_value = float(np.round(dataset.max(), 3))
    quartile_1 = float(np.round(dataset.quantile(0.25), 3))
    quartile_3 = float(np.round(dataset.quantile(0.75), 3))
    iqr = np.round(quartile_3 - quartile_1, 3)
    lower_bound = np.round(quartile_1 - iqr * 1.5, 3)
    upper_bound = np.round(quartile_3 + iqr * 1.5, 3)

    print(f'{dataname} summary statistics')
    print(f'Min                      : {min_value}')
    print(f'Mean                     : {mean}')
    print(f'Max                      : {max_value}')
    print('')
    print(f'25th percentile          : {quartile_1}')
    print(f'Median                   : {median}')
    print(f'75th percentile          : {quartile_3}')
    print(f'Interquartile range (IQR): {iqr}')
    print('')
    print(f'Lower outlier bound      : {lower_bound}')
    print(f'Upper outlier bound      : {upper_bound}')
    print('--------------------------------')


def show_boxplot(times: dict, fname: str):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(times.values())
    # add patch_artist=True option to ax.boxplot() to get fill color
    bp = ax.boxplot(times.values(), patch_artist=True)
    # Custom x-axis labels
    ax.set_xticklabels(times.keys())
    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Show the figure
    plt.show()
    # Save the figure
    plt.savefig(f'../src/imgs/{fname}_out.png')


def save_to_file(times: dict, sheet_name: str):
    df = pd.DataFrame(times)
    df.to_excel(f'../data.xlsx', sheet_name=sheet_name)
