""" Hyperparameters for MJC peg insertion policy optimization. """
import imp
import os.path
from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS

BASE_DIR = '/'.join(str.split(__file__, '/')[:-3])
default = imp.load_source('default_hyperparams', BASE_DIR+'/hyperparams.py')

EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

# Update the defaults
common = default.common.copy()
common.update({
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
})

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# Algorithm
algorithm = default.algorithm.copy()
algorithm.update({
    'type': AlgorithmMDGPS,
    'sample_on_policy': True,
    'step_rule': 'classic',
})

algorithm['policy_opt']['weights_file_prefix'] = EXP_DIR + 'policy'

config = default.config.copy()
config.update({
    'common': common,
    'algorithm': algorithm,
    'verbose_policy_trials': 1,
    'random_seed': 25,
})

common['info'] = generate_experiment_info(config)
