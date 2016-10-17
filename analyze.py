import sys, imp
import os.path
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add gps/python to path so that imports work.
# (should be run from ~/gps)
sys.path.append(os.path.abspath('python'))
from gps.sample.sample_list import SampleList

#############
## Helpers ##
#############
def unpickle_final_eepts(task, method, seed, itr):
    dirname = "experiments/%s/%s/%s/data_files" % (task, method, seed)
    fname = "%s/test_eepts_%02d.pkl" % (dirname, itr)
    with open(fname, 'rb') as f:
        final_eepts = cPickle.load(f)
    return final_eepts
    
def get_final_eepts(task, method, seeds, itr, fn=None):
    """ fn :: (M, dim_eepts) => (M, dim_fn) """

    print "Processing task %s, method %s, itr %s" % (task, method, itr)

    vals = []
    for seed in seeds:
        dirname = "experiments/%s/%s/%s/data_files" % (task, method, seed)

        fname = "%s/test_eepts_%02d.pkl" % (dirname, itr)
        if not os.path.exists(fname):
            print "Skipping task %s, method %s, seed %s, itr %s" % (task, method, seed, itr)
            continue

        eepts = unpickle_final_eepts(task, method, seed, itr)
        eepts = eepts[:, -1, :] # Hack...
        if np.isnan(eepts).any() or np.isinf(eepts).any():
            debug_here()
            print "Skipping, NaN/Inf, task %s, method %s, seed %s, itr %s" % (task, method, seed, itr)
            continue

        if fn is not None:
            vals.append(fn(eepts))
        else:
            vals.append(eepts)

    return np.array(vals) # (num_seeds, M, dim_eepts or dim_fn)

##############
## Plotting ##
##############
def plot(task, methods, seeds, fn, fn_name, colors, labels):
    """ Plot 'iters' vs. 'fn(eepts)' for a given task """
    for method, color, label in zip(methods, colors, labels):
        fname = 'experiments/%s/%s/stats_%s.pkl' % (task, method, fn_name)
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                xs, means, stds = cPickle.load(f)
        else:
            means = []
            stds = []
            itr = 0
            while True:
                ys = get_final_eepts(task, method, seeds, itr, fn) # (num_seeds, M)

                # Stop if no seeds have run.
                if ys.shape[0] == 0:
                    break

                # Average over conditions.
                ys = ys.mean(axis=1) # (num_seeds,)

                # Store mean/std over seeds
                means.append(ys.mean())
                stds.append(ys.std())
                itr += 1

            xs = 1. + np.arange(len(means))

            # Store for easier access next time.
            with open(fname, 'wb') as f:
                data = (xs, means, stds)
                cPickle.dump(data, f, -1)

        # Plot vs. iterations (1-indexed)
#        plt.errorbar(xs, means, stds, c=color, label=label)
        plt.plot(xs, means, c=color, label=label)
        xs = np.array(xs)
        means = np.array(means)
        stds = np.array(stds)
        plt.fill_between(xs, means-stds, means+stds, color=color, alpha=0.15)

def success_pct(task, methods, seeds, iters, success):
    """ Print out the success percentages. """

    out = "%4s" % "Iter"
    out += " | "
    out += " | ".join(["%15s" % method for method in methods])
    out += "\n"

    for itr in iters:
        out += "%4d" % itr
        out += " | "
        means = []
        stds = []
        for method in methods:
            ys = get_final_eepts(task, method, seeds, itr, success) # (num_seeds, M)

            # Average over conditions.
            ys = ys.mean(axis=1) # (num_seeds,)

            # Store mean/std over seeds.
            means.append(100*ys.mean())
            stds.append(100*ys.std())

        # Add on new means/stds.
        out += " | ".join(["%.3f +/- %.3f" % meanstd for meanstd in zip(means, stds)])
        out += "\n"

    # Give all the success percentages.
    print out


###### Main section
methods = ['badmm', 'off_classic','off_global','on_classic','on_global']
labels = ['BADMM', 'Off Policy, Classic Step','Off Policy, Global Step','On Policy, Classic Step','On Policy, Global Step']
colors = ['k', 'b', 'g', 'r', 'm']
seeds = range(10)


##### Distance Plots
plt.figure(figsize=(13, 4))
gs = gridspec.GridSpec(1, 3)


##### Obstacle Course (5 Positions)
task = 'obstacle'
tgt = np.array([3.0, 0.0])
distance = lambda eepts: np.sqrt(np.sum((eepts[:, :2] - tgt)**2, axis=1))

### Distance
plt.subplot(gs[0,0])
plot(task, methods, seeds, distance, 'distance', colors, labels)
plt.title("Obstacle Course")
# Labels, etc.
plt.xlabel('Iterations')
plt.ylabel('Distance to Target')
ax = plt.gca()
ax.grid(True)
ax.set_xlim([1, 15])
ax.set_ylim([0, 3.5])


##### Peg Insertion (9 Positions)
task = 'peg'
tgt = np.array([0, 0.3, -0.5])
distance = lambda eepts: np.sqrt(np.sum((eepts[:, :3] - tgt)**2, axis=1))
success = lambda eepts: distance(eepts) < 0.06

### Distance
plt.subplot(gs[0,1])
plot(task, methods, seeds, distance, 'distance', colors, labels)
plt.title("Peg Insertion")
# Success threshold.
xs = [0, 12]
ys = [0.1, 0.1]
plt.plot(xs, ys, 'k--', label="Success Threshold")
# Labels, etc.
plt.xlabel('Iterations')
plt.ylabel('Distance to Target')
ax = plt.gca()
ax.grid(True)
ax.set_xlim([1, 12])
ax.set_ylim([0, 0.6])
ax.legend(loc='upper right', borderaxespad=0.)


##### Blind Peg Insertion (4 Positions)
task = 'peg_blind'

### Distance
plt.subplot(gs[0,2])
plot(task, methods, seeds, distance, 'distance', colors, labels)
plt.title("Blind Peg Insertion")
# Success threshold.
xs = [0, 12]
ys = [0.1, 0.1]
plt.plot(xs, ys, 'k--', label="Success Threshold")
# Labels, etc.
plt.xlabel('Iterations')
plt.ylabel('Distance to Target')
ax = plt.gca()
ax.grid(True)
ax.set_xlim([1, 12])
ax.set_ylim([0, 0.6])


##### Finalize/save
plt.tight_layout()
plt.savefig('results.png')
plt.clf()


###### Success Percentages
tgt = np.array([0, 0.3, -0.5])
distance = lambda eepts: np.sqrt(np.sum((eepts[:, :3] - tgt)**2, axis=1))
success = lambda eepts: distance(eepts) < 0.1
iters = [2, 5, 8, 11]
success_pct('peg', methods, seeds, iters, success)
success_pct('peg_blind', methods, seeds, iters, success)
