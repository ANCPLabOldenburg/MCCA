import numpy as np
import json, os

def get_dir_from_user(title='Select Zhang data directory'):
    import tkinter as tk
    import tkinter.filedialog
    root = tk.Tk()
    root.withdraw()
    return tk.filedialog.askdirectory(title=title) + '/'

def load_config():
    global config
    if os.path.exists('./config.json'):
        changed = False
        with open('./config.json','r') as config_file:
            config = json.load(config_file)
            if not os.path.isdir(config['zhang_data_folder']):
                changed = True
                config['zhang_data_folder'] = get_dir_from_user()
        if changed:
            with open('./config.json','w') as config_file:
                json.dump(config,config_file,indent=4)
    else:
        config = {'data_folder': '',
                  'raw_folder': '',
                  'results_folder': './results/',
                  'cachedir': './joblib/',
                  'zhang_data_folder': get_dir_from_user()}
        with open('./config.json','w') as config_file:
            json.dump(config,config_file,indent=4)
    if not os.path.exists(config['results_folder']):
        os.mkdir(config['results_folder'])

load_config()
results_folder = config['results_folder']
 
def load_zhang_data():
    from scipy.io import loadmat
    even = loadmat(config['zhang_data_folder']+'averageDataEven.mat')
    odd = loadmat(config['zhang_data_folder']+'averageDataOdd.mat')
    return even['dataEven'].T,odd['dataOdd'].T

def load_zhang_data_st():
    from scipy.io import loadmat
    data = loadmat(config['zhang_data_folder']+'averageData.mat')
    st = loadmat(config['zhang_data_folder']+'singleTrials.mat')
    return data['dataAll'].T,st['temp'][np.newaxis,:,2:]