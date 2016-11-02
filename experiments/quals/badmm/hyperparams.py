""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import evall1l2term
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import *

NUM_SAMPLES = 5

SENSOR_DIMS = {
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 3,
}

GAINS = np.ones(SENSOR_DIMS[ACTION])

CONDITIONS = 4

xs = [-0.7, 0.7]
zs = [-0.7, 0.7]
pos_body_offset = [np.array([x, 0, z]) for x in xs for z in zs]


BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/quals_reacher/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    'train_conditions': range(CONDITIONS), 
    'test_conditions': range(CONDITIONS), 
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/arm_3link_reach.xml',
    'conditions': common['conditions'],
    'x0': np.zeros(6),
    'T': 100,
    'dt': 0.05,
    'substeps': 3,
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([1]),
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 6., 0., 0., 0., 0.]),
}

#algorithm = {
#    'type': AlgorithmTrajOpt,
#    'conditions': common['conditions'],
#    'iterations': 10,
#    'kl_step': 1.0,
#    'min_step_mult': 0.1,
#    'max_step_mult': 3.0,
#}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'sample_on_policy': True,
    'iterations': 15,
    'kl_step': 0.5,
    'min_step_mult': 0.1,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': GAINS,
    'init_acc': np.zeros_like(GAINS),
    'init_var': 5.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_costs = [{
    'type': CostAction,
    'wu': 1e-3 / GAINS,
} for i in range(common['conditions'])]

# Diff between average of end_effectors and block.
fk_costs = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0., 0., 0.]), agent['pos_body_offset'][i]]),
    'wp': np.array([0, 0, 0, 1, 1, 1]),
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 0,
    'evalnorm': evall1l2term,
    'wp_final_multiplier': 3.0,
} for i in range(common['conditions'])]

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [torque_costs[i], fk_costs[i]],
    'weights': [2.0, 1.0],
}  for i in range(common['conditions'])]

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 3*NUM_SAMPLES,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'iterations': 2000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'gui_on': True,
    'iterations': algorithm['iterations'],
    'num_samples': NUM_SAMPLES,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
