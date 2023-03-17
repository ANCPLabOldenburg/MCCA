import json
import os

import numpy as np


def get_dir_from_user(title='Select data directory'):
    import tkinter as tk
    import tkinter.filedialog
    root = tk.Tk()
    root.withdraw()
    return tk.filedialog.askdirectory(title=title) + '/'


def load_config():
    global config
    if os.path.exists('./config.json'):
        changed = False
        with open('./config.json', 'r') as config_file:
            config = json.load(config_file)
            if not os.path.isdir(config['data_folder']):
                changed = True
                config['data_folder'] = get_dir_from_user()
        if changed:
            with open('./config.json', 'w') as config_file:
                json.dump(config, config_file, indent=4)
    else:
        config = {'data_folder': './data/',
                  'raw_folder': '',
                  'results_folder': './results/',
                  'cache_dir': './joblib/'}
        if not os.path.isdir(config['data_folder']):
            config['data_folder'] = get_dir_from_user()
        with open('./config.json', 'w') as config_file:
            json.dump(config, config_file, indent=4)
    if not os.path.exists(config['results_folder']):
        os.mkdir(config['results_folder'])


config = {}
load_config()
results_folder = config['results_folder']


def load_single_trial_data(tsss_realignment=False):
    fn = config['data_folder'] + ("single_trial_tSSS.npz" if tsss_realignment else "single_trial_no_tSSS.npz")
    data = np.load(fn, allow_pickle=True)
    return data["X"], data["y"]
