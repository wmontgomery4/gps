import sys, imp
import os.path
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Add gps/python to path so that imports work.
# (should be run from ~/gps)
sys.path.append(os.path.abspath('python'))
from gps.sample.sample_list import SampleList

expts = ['badmm', 'off_classic', 'off_global', 'on_classic', 'on_global']
labels = ['BADMM (Off Policy)', 'Off Policy, Classic Step', 'Off Policy, Global Step', 'On Policy, Classic Step', 'On Policy, Global Step']
seeds = [0, 1, 2]
colors = ['k', 'r', 'm', 'b', 'g']

SUCCESS_THRESHOLD = -0.5 + 0.06

def pickle_final_eepts(task, expt, seed, itr):
    print "Pickling task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)

    # extract samples
    dirname = "experiments/%s/%s/%s/data_files" % (task, expt, seed)
    fname = "%s/pol_sample_itr_%02d.pkl" % (dirname, itr)

    with open(fname, 'rb') as f:
        sample_lists = cPickle.load(f) # (num_samples, M)

    # magic final_eepts extraction (get first final_eepts position of eepts)
    # NOTE: There's only one sample per sample_list, since it's policy samples
    final_eepts = np.array([s[0]._data[3][-1] for s in sample_lists]) # (M, num_eepts)

    # save final_eepts for faster replotting
    fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'wb') as f:
        cPickle.dump(final_eepts, f, -1)

def unpickle_final_eepts(task, expt, seed, itr):
    print "Unpickling task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)
    dirname = "experiments/%s/%s/%s/data_files" % (task, expt, seed)
    fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'rb') as f:
        final_eepts = cPickle.load(f)
    return final_eepts
    
def get_final_eepts(task, expt, itr):
    print "Processing task %s, expt %s, itr %s" % (task, expt, itr)

    eepts = []
    for seed in seeds:
        dirname = "experiments/%s/%s/%s/data_files" % (task, expt, seed)

        # skip if the seed hasn't been run
        pol_fname = "%s/pol_sample_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(pol_fname):
            print "Skipping task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)
            continue

        # pickle if the seed hasn't been pickled
        pkl_fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(pkl_fname) or \
                os.path.getmtime(pol_fname) > os.path.getmtime(pkl_fname):
            pickle_final_eepts(task, expt, seed, itr)

        pts = unpickle_final_eepts(task, expt, seed, itr)
        if np.isnan(pts).any() or np.isinf(pts).any():
            #debug_here()
            print "Skipping, NaN/Inf, task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)
            continue
        eepts.append(pts)

    return np.array(eepts) # (num_seeds, M, num_eepts)

#########################
## Success Percentages ##
#########################
def write_success_pcts(task):
    fname = 'experiments/%s/success.txt' % task
    with open(fname, 'w') as f:
        f.write('\hline\n')
        f.write('Iteration & ' + ' & '.join(labels) + '\\\\\n')
        f.write('\hline\n')
        for i in [3, 6, 9, 12]:
            line = "%d" % i
            for expt in expts:
                eepts = get_final_eepts(task, expt, i-1)
                zs = eepts[:, :, 2]
                pct = 100*np.mean(zs < SUCCESS_THRESHOLD)
                line += ("& %.2f" % pct) #+ "~\% "
            line += "\\\\ \n"
            f.write(line)
        f.write('\hline\n')

write_success_pcts('peg4')
write_success_pcts('peg9')
write_success_pcts('peg4_blind')

####################
## Distance Plots ##
####################

def plot_task(task, iters, fn):
    """ Plot 'iters' vs. 'fn(eepts)' for a given task """
    for expt, color, label in zip(expts, colors, labels):
        means = []
        stds = []
        for itr in range(iters):
            eepts = get_final_eepts(task, expt, itr)

            # Skip if none of the seeds have been run
            # NOTE: 'break' since no future iters could be run either
            if eepts.shape[0] == 0:
                break

            # Apply plot function and average over conditions
            ys = fn(eepts) # (num_seeds, M)
            ys = ys.mean(axis=1)

            # Store mean/std over seeds
            means.append(ys.mean())
            stds.append(ys.std())

        # Skip if none of the iterations have been run
        if len(means) == 0:
            continue

        # Plot vs. iterations (1-indexed)
        xs = np.arange(len(means)) + 1
        plt.errorbar(xs, means, stds, c=color, label=label)

# peg4 (distance)
task = 'peg4'
iters = 12
tgt = np.array([0, 0.3, -0.5])
fn = lambda eepts: np.sqrt(np.sum((eepts[:, :, :3] - tgt)**2, axis=2))

plt.title("Peg Insertion (4 conditions)")
plot_task(task, 12, fn)
xs = np.arange(iters) + 1
ys = 0.1*np.ones(iters)
plt.plot(xs, ys, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((1, iters))
plt.ylabel('Distance to Target')
plt.ylim((0, 0.5))
plt.savefig('experiments/peg4/distance.png')
plt.clf()

# peg4 (z-position)
task = 'peg4'
iters = 12
tgt = np.array([0, 0.3, -0.5])
fn = lambda eepts: eepts[:, :, 2] + 0.5

plt.title("Peg Insertion (4 conditions)")
plot_task(task, 12, fn)
xs = np.arange(iters) + 1
ys = 0.1*np.ones(iters)
plt.plot(xs, ys, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((1, iters))
plt.ylabel('Z Position')
plt.ylim((0, 0.5))
plt.savefig('experiments/peg4/z.png')
plt.clf()

# peg9 (distance)
task = 'peg9'
iters = 12
tgt = np.array([0, 0.3, -0.5])
fn = lambda eepts: np.sqrt(np.sum((eepts[:, :, :3] - tgt)**2, axis=2))

plt.title("Peg Insertion (9 conditions)")
plot_task(task, 12, fn)
xs = np.arange(iters) + 1
ys = 0.1*np.ones(iters)
plt.plot(xs, ys, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((1, iters))
plt.ylabel('Distance to Target')
plt.ylim((0, 0.5))
plt.savefig('experiments/peg9/distance.png')
plt.clf()

# peg9 (z-position)
task = 'peg9'
iters = 12
tgt = np.array([0, 0.3, -0.5])
fn = lambda eepts: eepts[:, :, 2] + 0.5

plt.title("Peg Insertion (9 conditions)")
plot_task(task, 12, fn)
xs = np.arange(iters) + 1
ys = 0.1*np.ones(iters)
plt.plot(xs, ys, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((1, iters))
plt.ylabel('Z Position')
plt.ylim((0, 0.5))
plt.savefig('experiments/peg9/z.png')
plt.clf()

# peg4_blind (distance)
task = 'peg4_blind'
iters = 12
tgt = np.array([0, 0.3, -0.5])
fn = lambda eepts: np.sqrt(np.sum((eepts[:, :, :3] - tgt)**2, axis=2))

plt.title("Blind Peg Insertion (4 conditions)")
plot_task(task, 12, fn)
xs = np.arange(iters) + 1
ys = 0.1*np.ones(iters)
plt.plot(xs, ys, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((1, iters))
plt.ylabel('Distance to Target')
plt.ylim((0, 0.5))
plt.savefig('experiments/peg4_blind/distance.png')
plt.clf()

# peg9 (z-position)
task = 'peg4_blind'
iters = 12
tgt = np.array([0, 0.3, -0.5])
fn = lambda eepts: eepts[:, :, 2] + 0.5

plt.title("Blind Peg Insertion (4 conditions)")
plot_task(task, 12, fn)
xs = np.arange(iters) + 1
ys = 0.1*np.ones(iters)
plt.plot(xs, ys, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((1, iters))
plt.ylabel('Z Position')
plt.ylim((0, 0.5))
plt.savefig('experiments/peg4_blind/z.png')
plt.clf()
