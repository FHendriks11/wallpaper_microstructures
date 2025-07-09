# %% [markdown]
# # Define things

# %%
import copy
import subprocess
import pickle
import os
import datetime
import time
import sys
import getopt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

import skgeom as sg
import meshio
import scipy.io

import helper_funcs as hf

import importlib
importlib.reload(hf)

# SKELETON GRAPH CONSTANTS:
MAX_B = 2 # maximum number of points on a boundary of the fundamental domain
MAX_I = 5 # maximum number of points inside the fundamental domain
PROB_C = 0.2 # probability of creating a point on a corner (or a point in the middle of a self-similar boundary)

# SECONDARY POLYGON CONSTANTS:
MIN_AREA = 0.1/6
MIN_THICKNESS = 0.03/np.sqrt(6)  # absolute
MAX_REL_THICKNESS = 0.7  # relative to bisector from straight skeleton
PROB_RANDOM_FILL = 0.02

# TERTIARY POLYGON CONSTANTS:
MIN_RADIUS = 0.05/np.sqrt(6)  # used to determine an absolute min. distance from secondary polygon corner to tertiary polygon corner
MIN_D_REL = 0.2  # relative min. distance
MIN_SEP_ABS = 0.05/np.sqrt(6)  # absolute separation between the two points on one secondary polygon edge
MIN_SEP_REL = 0.2 # relative separation
MIN_AREA2 = 0.05/6

# Maximum volume fraction of the material:
MAX_DENSITY = 0.75

# Maximum characteristic element size:
CLMAX = 0.1/np.sqrt(6)

def generate_material_geometry(group, shape, verbose=False, figures=1, save_dir='', rng_state_path=None):
    """Generate a new material geometry based on the given wallpaper group and shape.

    Parameters
    ----------
    group : str
        wallpaper group
    shape : str
        the shape that the fundamental or unit cell sort of resembles
    verbose : bool, optional
        whether to print lots of debugging info, by default False
    figures : int, optional
        how many figures to create and save. 0: None, 1: only the most important, 2: all (useful for debugging), by default 1
    save_dir : str, optional
        place to save outputs, by default ''
    rng_state_path : str, optional
        path to random number generator state. Allows replication of previously generated material. If None, generate new material. By default None

    Raises
    ------
    ValueError
        raised if a given input parameter value is not valid, or if something goes wrong resulting in invalid values for certain variables.
    Exception
        raised if the generated geometry is not valid, e.g. if the volume fraction is too high.
    Exception
        Raised if something goes wrong when removing the leafs or determining the faces of the skeleton graph (should not happen)
    """
    start_time = time.time()

    name = os.path.split(save_dir)[-1]

    print('Name:', name)
    print('Path:', save_dir)

    # %% [markdown]
    # ## Create random topology
    fig_nr = 0

    # %%
    fd = {'group': group, 'shape': shape}

    rng = np.random.default_rng()

    if rng_state_path is not None:
        with open(rng_state_path, 'rb') as f:
            rngstate = pickle.load(f)
        rng.bit_generator.state = rngstate

    # Save rng state
    with open(hf.new_path(os.path.join(save_dir, 'rngstate.pkl')), 'wb') as f:
        pickle.dump(rng.bit_generator.state, f)

    params = hf.wallpaper_groups[group]['fundamental domain parameters'][shape]
    fd['a'] = params['a']
    fd['b'] = params['b'] if not isinstance(params['b'], list) else rng.uniform(*params['b'])
    fd['gamma'] = params['gamma'] if not isinstance(params['gamma'], list) else rng.uniform(*params['gamma'])

    # normalize area
    area = fd['a']*fd['b']*np.sin(fd['gamma'])  # current area of fd
    if verbose:
        print(area)
    if hf.wallpaper_groups[group]['fundamental domain shape'] == 'triangle':
        area /= 2

    n_fds = len(hf.wallpaper_groups[fd["group"]]['unit cell'])
    desired_area = 1/n_fds
    fd['a'] = fd['a']/np.sqrt(area)*np.sqrt(desired_area)
    fd['b'] = fd['b']/np.sqrt(area)*np.sqrt(desired_area)

    # recalculate area, check if it's 1/n_fds
    area = fd['a']*fd['b']*np.sin(fd['gamma'])
    if hf.wallpaper_groups[group]['fundamental domain shape'] == 'triangle':
        area /= 2
    if verbose:
        print(area)

    fd['a1'] = np.array([fd["a"], 0.0])
    fd['a2'] = np.array([fd["b"]*np.cos(fd["gamma"]), fd["b"]*np.sin(fd["gamma"])])

    # define boundaries of fundamental domain
    # counterclockwise, by start point and vector pointing along edge
    if hf.wallpaper_groups[group]['fundamental domain shape'] == 'triangle':
        fd['bounds'] = [(np.array([0,0]), fd["a1"]), (fd["a1"], fd["a2"]-fd["a1"]), (fd["a2"], -fd["a2"])]
    elif hf.wallpaper_groups[group]['fundamental domain shape'] == 'parallelogram':
        fd['bounds'] = [(np.array([0,0]), fd["a1"]), (fd["a1"], fd["a2"]), (fd["a1"]+fd["a2"], -fd["a1"]), (fd["a2"], -fd["a2"])]
    else:
        raise ValueError(f'bad value of fundamental domain shape: {hf.wallpaper_groups[group]["fundamental domain shape"]}')

    fd['corners'] = np.array([b[0] for b in fd['bounds']])
    fd['linked_bounds'] = hf.wallpaper_groups[group]['boundaries']


    fd['points'], fd['bound_inds'] = hf.generate_points(fd['a1'], fd['a2'],
        fd['bounds'], fd['corners'], rng, fd['linked_bounds'], max_b=MAX_B, max_i=MAX_I, prob_c=PROB_C
        )

    fd['edges'] = hf.connect_nodes(fd['points'], fd['bound_inds'], fd['linked_bounds'], rng=rng)

    A = np.zeros((len(fd['points']), len(fd['points'])), dtype=bool)
    A[fd['edges'][:, 0], fd['edges'][:, 1]] = True
    A[fd['edges'][:, 1], fd['edges'][:, 0]] = True

    conn, component = hf.is_connected(A, fd['bound_inds'], fd['linked_bounds'], return_component=True)

    if not conn:
        raise Exception('not connected!')


    # %%
    fd['n_points'] = len(fd['points'])
    fd['n_edges'] = len(fd['edges'])

    # %% Plot fundamental domain
    if figures in [1,2]:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{fd["group"]} ({fd["shape"]})')
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        ax.fill(*fd["corners"].T, c='lightgrey', zorder=-1)  #, alpha=0.5)

        # plot original points
        ax.scatter(*fd["points"].T, label='original position', s=1, c='tab:red')
        for c in np.unique(component):
            ax.scatter(*fd["points"][component == c].T, c='black')
        x, y = np.transpose(fd["points"][fd["edges"].T], axes=[2,0,1])
        edges0 = ax.plot(x, y, c='black')

        ax.set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_fundamental_domain.png'))
        fig_nr += 1
        fd_plot_lims = (ax.get_xlim(), ax.get_ylim())
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% [markdown]
    # ## Turn fundamental domain into unit cell
    uc = {}
    uc['transforms'] = []
    p_arr = []
    uc['corners_per_fd'] = []
    uc['n_fds'] = len(hf.wallpaper_groups[fd["group"]]['unit cell'])
    uc['bounds_per_fd'] = []
    uc['bound_inds_per_fd'] = []

    for j, copy1 in enumerate(hf.wallpaper_groups[fd["group"]]['unit cell']):
        uc['transforms'].append([])

        p2 = np.copy(fd["points"])
        c2 = np.copy(fd['corners'])
        bounds = fd['bounds'].copy()
        bound_inds  = fd['bound_inds'].copy()

        for i, b in enumerate(bound_inds):
            bound_inds[i] = [bi + j*fd['n_points'] for bi in b]

        for transform in copy1:

            transform = (transform
                            .replace('a1', 'np.array('+str(fd["a1"].tolist())+')')
                            .replace('a2', 'np.array('+str(fd["a2"].tolist())+')').split(':'))

            if transform[0] == 'T':
                p2 = hf.translate_points(p2, eval(transform[1]))
                c2 = hf.translate_points(c2, eval(transform[1]))
                for i, b in enumerate(bounds):
                    bounds[i] = (hf.translate_points(b[0], eval(transform[1]))[0], b[1])

            elif transform[0] == 'R':
                degrees = np.array(eval(transform[2]))
                p2 = hf.rotate_points(p2, eval(transform[1]), degrees/360*2*np.pi)
                c2 = hf.rotate_points(c2, eval(transform[1]), degrees/360*2*np.pi)
                for i, b in enumerate(bounds):
                    bounds[i] = (hf.rotate_points(b[0], eval(transform[1]), degrees/360*2*np.pi)[0], hf.rotate_points(b[1], [0,0], degrees/360*2*np.pi)[0])
            elif transform[0] == 'M':
                p2 = hf.mirror_points(p2, eval(transform[1]), eval(transform[2]))
                c2 = hf.mirror_points(c2, eval(transform[1]), eval(transform[2]))
                for i, b in enumerate(bounds):
                    bounds[i] = (hf.mirror_points(b[0], eval(transform[1]), eval(transform[2]))[0], hf.mirror_points(b[1], [0,0], eval(transform[2]))[0])
            else:
                raise ValueError(f'transform {transform[0]} is not a valid transform, choose from ["T", "R", "M"] (translate, rotate, mirror)')

            uc['transforms'][-1].append(transform)


        p_arr.append(p2)
        uc['corners_per_fd'].append(c2)
        uc['bounds_per_fd'].append(bounds)
        uc['bound_inds_per_fd'].append(bound_inds)

    vecs = []
    for vec in hf.wallpaper_groups[group]['lattice vectors']:
        vec = (vec.replace('a1', 'np.array('+str(fd["a1"].tolist())+')')
                    .replace('a2', 'np.array('+str(fd["a2"].tolist())+')'))
        vec = eval(vec)
        vecs.append(vec)
    uc['lattice vectors'] = np.array(vecs)

    # unit cell
    corners2 = [eval(asdf[0].replace('a1', 'np.array('+str(fd["a1"].tolist())+')')
                .replace('a2', 'np.array('+str(fd["a2"].tolist())+')')) for asdf in hf.wallpaper_groups[group]['unit cell boundaries']]
    uc['corners'] = np.array(corners2)



    # %%
    uc["points"] = np.concatenate(p_arr, axis=0)
    uc["points_inds_per_fd"] = np.arange(len(uc["points"])).reshape(uc['n_fds'], fd['n_points'])
    uc["edges"] = (np.tile(fd["edges"], (len(p_arr), 1, 1)) + len(p_arr[0])*np.arange(len(p_arr)).reshape(-1, 1, 1)).reshape(-1, 2)

    # %%
    if verbose:
        print('Contents of unit cell dict:')
        for key, value in uc.items():
            try:
                print(f"{key:20} shape {value.shape}")
            except AttributeError:
                try:
                    print(f"{key:20} len {len(value)}")
                except TypeError:
                    print(f"{key:20} {value}")

    # %% Plot unit cell, to check boundaries
    if figures == 2:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-1)  # alpha=0.3,

        # background fundamental domain shape
        ax.fill(*fd['corners'].T, c='whitesmoke', zorder=-1)  #, alpha=0.5

        # loop over copies of fundamental domain
        for i, [inds, c] in enumerate(zip(uc["points_inds_per_fd"], uc['corners_per_fd'])):
            points_temp = uc["points"][inds]
            # plt.scatter(*points.T)
            ax.scatter(*points_temp.T, alpha=0.5, s=500)  #, c='tab:orange')
            x, y = np.transpose(points_temp[fd["edges"].T], axes=[2,0,1])
            edges0 = ax.plot(x, y, alpha=0.3, c='tab:red')

            if i==0:
                ax.scatter(*c.T, s=200, alpha=0.5, marker='x')

        for b in fd["bounds"]:
            plt.annotate("", xy=b[0]+b[1], xytext=b[0],
                        arrowprops=dict(#arrowstyle="->",
                                        zorder=10
                                        )
                        )

        for fd_temp in uc["bounds_per_fd"]:
            for b in fd_temp:
                plt.scatter(*b[0], s=500, c='black')
                plt.scatter(*(b[0]+b[1]), s=200, c='black')
                plt.annotate("", xy=b[0]+b[1], xytext=b[0],
                            arrowprops=dict(#arrowstyle="->",
                                            zorder=10
                                            )
                            )

        ax.set_aspect('equal')
        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell_check_boundaries.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()


    # %% Plot unit cell tiled 3×3
    if figures == 2:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        ax.fill(*uc["corners"].T, c='whitesmoke', alpha=0.5, zorder=-10)

        # background fundamental domain shape
        ax.fill(*fd['corners'].T, c='whitesmoke', alpha=0.5, zorder=-10)

        # loop over vecs[0] translations
        for i in range(3):
            # loop over vecs[1] translations
            for j in range(3):
                points_temp = hf.translate_points(uc["points"], i*vecs[0]+j*vecs[1])
                ax.scatter(*points_temp.T, alpha=0.5, s=50)  #, c='tab:orange')
                x, y = np.transpose(points_temp[uc["edges"].T], axes=[2,0,1])
                edges0 = ax.plot(x, y, alpha=0.3, c='tab:red')

        ax.set_aspect('equal')
        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell_3x3.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()


    # %% [markdown]
    # ## Remove duplicate nodes and edges

    # %%
    if verbose:
        print('Content of unit cell dict:')
        for key, value in uc.items():
            try:
                print(f"{key:20} shape {value.shape}")
            except AttributeError:
                try:
                    print(f"{key:20} len {len(value)}")
                except TypeError:
                    print(f"{key:20} {value}")

    # %%
    uc['points'], inds, inv, c = hf.uniquetol(uc["points"], tol=1e-5, return_counts=True, return_index=True, return_inverse=True, axis=0)

    for i, b in enumerate(uc["bound_inds_per_fd"]):
        for j, ind in enumerate(b):
            uc["bound_inds_per_fd"][i][j] = inv[ind].tolist()

    uc['points_inds_per_fd'] = inv[uc["points_inds_per_fd"]]
    edges_new = inv[uc["edges"]]

    # remove duplicate edges
    uc['edges'] = np.unique(np.sort(edges_new, axis=-1), axis=0)

    uc['fd_of_points'] = np.floor(inds/fd['n_points']).astype(int)

    # %% [markdown]
    # ## Find boundary nodes of unit cell

    # %%
    tol = 1e-6

    # find new boundary nodes
    uc['bounds'] = []
    b_inds2 = []
    for b_start, b_vec in hf.wallpaper_groups[group]['unit cell boundaries']:
        b_start = eval(b_start.replace('a1', 'np.array('+str(fd['a1'].tolist())+')')
                    .replace('a2', 'np.array('+str(fd['a2'].tolist())+')'))
        b_vec = eval(b_vec.replace('a1', 'np.array('+str(fd['a1'].tolist())+')')
                    .replace('a2', 'np.array('+str(fd['a2'].tolist())+')'))
        b_inds2.append([])
        bools_onbound = hf.iscollinear(uc['points'], b_start, b_start+b_vec, tol_g=1e-5)
        b_inds2[-1] = np.where(bools_onbound)[0]

        # sort them in the correct order
        if np.abs(b_vec[0]) > tol:
            temp = np.argsort(uc['points'][b_inds2[-1], 0]/b_vec[0])
        elif np.abs(b_vec[1]) > tol:
            temp = np.argsort(uc['points'][b_inds2[-1], 1]/b_vec[1])
        else:
            raise ValueError(f"bad value of b_vec: {b_vec}")
        b_inds2[-1] = b_inds2[-1][temp].tolist()

        uc['bounds'].append((b_start, b_vec))

    uc['bound_inds'] = b_inds2

    uc['bound_inds_flat'] = []
    for inds in uc['bound_inds']:
        uc['bound_inds_flat'].extend(inds)

    # %% Plot to check unit cell boundaries
    if figures == 2:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # points_temp = hf.translate_points(uc['points'], i*vecs[0]+j*vecs[1])
        # plt.scatter(*points.T)
        ax.scatter(*uc['points'].T, alpha=0.5)  #, c='tab:orange')
        ax.scatter(*uc['points'][uc['bound_inds_flat']].T, alpha=0.5)  #, c='tab:orange')

        # background fundamental domain shape
        ax.fill(*fd["corners"].T, c='whitesmoke', zorder=-1)  # , alpha=0.5

        # background unit cell shape
        ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-1)  # , alpha=0.5

        x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
        edges0 = ax.plot(x, y, alpha=0.3, c='tab:red')

        for i, point in enumerate(uc['points']):
            ax.text(*(point+0.02), i)

        for i, ind in enumerate(uc["bound_inds_flat"]):
            ax.text(*(uc['points'][ind]+0.01), i, c='tab:red')


        for bound in uc["bounds"]:
            ax.annotate("", xy=bound[0]+bound[1], xytext=bound[0],
                        arrowprops=dict(#arrowstyle="->",
                                        zorder=10
                                        )
                        )

        ax.set_aspect('equal')
        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell_check_boundaries2.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% Plot to check unit cell without boundary vectors
    if figures in [1,2]:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # points_temp = hf.translate_points(uc['points'], i*vecs[0]+j*vecs[1])
        # plt.scatter(*points.T)
        ax.scatter(*uc['points'].T, c='black')

        # background fundamental domain shape
        ax.fill(*fd["corners"].T, c='lightgrey', zorder=-1)  #, alpha=0.5

        # background unit cell shape
        ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-2)  # , alpha=0.5

        x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
        edges0 = ax.plot(x, y, c='black')

        ax.set_aspect('equal')
        uc_plot_lims = (ax.get_xlim(), ax.get_ylim())
        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # plot grid of multiple unit cells
    if figures in [2,]:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        ax.fill(*fd["corners"].T, c='whitesmoke', alpha=0.5, zorder=-1)

        # background fundamental domain shape
        ax.fill(*uc["corners"].T, c='whitesmoke', alpha=0.5, zorder=-1)

        # loop over vecs[0] translations
        for i in range(3):
            # loop over vecs[1] translations
            for j in range(3):
                points_temp = hf.translate_points(uc["points"], i*vecs[0]+j*vecs[1])
                ax.scatter(*points_temp.T, alpha=0.5)  #, c='tab:orange')
                x, y = np.transpose(points_temp[uc["edges"].T], axes=[2,0,1])
                edges0 = ax.plot(x, y, alpha=0.3, c='tab:red')

        ax.set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell_3x3.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% [markdown]
    # ## Find stubs

    # %%
    if hf.wallpaper_groups[group]['unit cell shape'] == 'parallelogram':
        uc["linked_bounds"] = [(0, 2, -1), (1, 3, -1)]
        uc["n_corners"] = 4
        uc['linked_corners'] = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
    elif hf.wallpaper_groups[group]['unit cell shape'] == 'hexagon':
        uc["linked_bounds"] = [(0, 3, -1), (1, 4, -1), (2, 5, -1)]
        uc["n_corners"] = 6
        uc['linked_corners'] = [[2,4], [3,5], [0,4], [1,5], [0,2], [1,3]]
    else:
        raise ValueError(f"{hf.wallpaper_groups[group]['unit cell shape']} is an invalid unit cell shape")

    # %%
    for i in range(1000):  # actually basically a while loop
        if verbose: print(f'Removing stubs, iteration {i}')

        # get dependent nodes
        equiv_nodes = hf.list_equivalent_nodes(uc["bound_inds"], uc["linked_bounds"], len(uc["points"]), include_self=True)
        if verbose: print(equiv_nodes)
        uc['equiv_nodes'] = equiv_nodes

        # get list of corner indices
        uc['corner_inds'] = [[] for i in range(uc["n_corners"])]
        if uc['bound_inds'][-1][-1] == uc['bound_inds'][0][0]:
            uc['corner_inds'][0].append(uc['bound_inds'][0][0])
        for i, [b1, b2] in enumerate(zip(uc['bound_inds'][:-1], uc['bound_inds'][1:])):
            if len(b1) > 0 and len(b2) > 0:
                if b1[-1] == b2[0]:
                    uc['corner_inds'][i+1].append(b1[-1])
        if verbose: print('uc["corner_inds"]:', uc['corner_inds'])

        # make flattened version of corner indices list
        uc['corner_inds_flat'] = []
        for inds in uc['corner_inds']: uc['corner_inds_flat'].extend(inds)
        uc['corner_inds_flat']

        # get list of unique edges that are not duplicates/periodic copies of other edges
        uc['periodic_edges'] = hf.list_equivalent_edges(uc["bound_inds"],
                                            uc["linked_bounds"],
                                            uc["edges"])
        if verbose: print(" uc['periodic_edges']",  uc['periodic_edges'])
        # find unique pairs of equivalent edges
        temp_pe = np.unique(np.sort(uc['periodic_edges'], axis=-1), axis=0)
        if len(temp_pe) > 0:
            edges2 = np.delete(uc["edges"], temp_pe[:, 0], axis=0)
        else:
            edges2 = np.copy(uc['edges'])

        # count nr of edges per node
        c = np.bincount(edges2.flatten(), minlength=len(uc['points']))
        # add count of equivalent nodes
        for i in range(len(uc['points'])):
            c[i] += np.sum(c[equiv_nodes[i]])
        stubs = np.where(c<=1)[0]  # indices of nodes that are the end of stubs

        if verbose:
            print('edges2', edges2)
            print('c', c)
            print('stubs', stubs)

        n_stubs = len(stubs)
        stubs = stubs.tolist()
        # also add equivalent nodes of stubs
        for i in range(n_stubs):
            stubs.extend(equiv_nodes[stubs[i]])

        if len(stubs) == 0:
            break

        if figures in [1,2]:
            # plot unit cell to check if stubs are correct
            fig, ax = plt.subplots(figsize=(7,7))
            # ax.set_title(f'{group} ({shape})')
            fig.patch.set_facecolor("None")

            ax.scatter(*uc['points'].T, c='black')  #alpha=0.5)  #, c='tab:orange')
            # ax.scatter(*uc['points'][uc['bound_inds_flat']].T, alpha=0.5)  #, c='tab:orange')
            ax.scatter(*uc['points'][stubs].T, c='magenta', zorder=10)

            # background fundamental domain shape
            ax.fill(*fd["corners"].T, c='lightgrey', zorder=-1)  #, alpha=0.5

            # background fundamental domain shape
            ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-2)  #, alpha=0.5

            x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
            edges0 = ax.plot(x, y, c='black')

            # for i, point in enumerate(uc['points']):
            #     ax.text(*(point+0.02), i)

            ax.set_aspect('equal')

            path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_stubs.png'))
            fig_nr += 1
            plt.axis('off')
            fig.savefig(path1)
            plt.close()

        # ## Remove stubs
        to_keep = np.ones(len(uc["points"]), dtype=bool)
        to_keep[stubs] = False

        # remove stubs (points and edges)
        # prod works as logical AND
        bools = np.prod(to_keep[uc["edges"]], axis=-1).astype(bool)
        edges = uc["edges"][bools]
        points = uc["points"][to_keep]

        # renumber indices
        to_keep = np.where(to_keep)[0]
        edges = np.searchsorted(to_keep, edges)

        # renumber bound_inds
        bound_inds2 = [[] for b in uc["bound_inds"]]
        for i, b in enumerate(uc["bound_inds"]):
            temp = np.searchsorted(to_keep, b)
            temp = np.clip(temp, a_min=0, a_max=len(to_keep)-1)
            bound_inds2[i] = temp[to_keep[temp] == b]

        # renumber corner_inds
        corner_inds2 = [[] for b in uc["corner_inds"]]
        for i, b in enumerate(uc["corner_inds"]):
            temp = np.searchsorted(to_keep, b)
            temp = np.clip(temp, a_min=0, a_max=len(to_keep)-1)
            corner_inds2[i] = temp[to_keep[temp] == b]

        # to do: update
        if verbose:
            print('Before:')
            print(uc["bound_inds_per_fd"])

        # bound_inds_per_fd (remove stubs & renumber)
        for i, fd_temp in enumerate(uc["bound_inds_per_fd"]):
            for j, b in enumerate(fd_temp):
                temp = np.searchsorted(to_keep, b)
                temp = np.clip(temp, a_min=0, a_max=len(to_keep)-1)
                uc["bound_inds_per_fd"][i][j] = temp[to_keep[temp] == b].tolist()

        if verbose:
            print('After:')
            print(uc["bound_inds_per_fd"])

            print('Before:')
            print(uc["points_inds_per_fd"])

        # points_inds_per_fd (remove stubs & renumber)
        temp = np.searchsorted(to_keep, uc["points_inds_per_fd"][0])
        temp = np.clip(temp, a_min=0, a_max=len(to_keep)-1)
        isintokeep = to_keep[temp] == uc["points_inds_per_fd"][0]
        uc["points_inds_per_fd"] = uc["points_inds_per_fd"][:, isintokeep]
        for i, inds in enumerate(uc["points_inds_per_fd"]):
            temp = np.searchsorted(to_keep, inds)
            temp = np.clip(temp, a_min=0, a_max=len(to_keep)-1)
            uc["points_inds_per_fd"][i] = temp

        if verbose:
            print('After:')
            print(uc["points_inds_per_fd"])

        # fd_of_points
        uc["fd_of_points"] = uc["fd_of_points"][to_keep]

        # n_points
        uc["n_points"] = len(uc["points"])

        # put in uc dict
        uc["points"] = points
        uc["edges"] = edges
        uc["bound_inds"] = bound_inds2
        uc["corner_inds"] = corner_inds2
        uc["n_points"] = len(points)

        # update flattened version of corner indices list
        uc['corner_inds_flat'] = []
        for inds in uc['corner_inds']: uc['corner_inds_flat'].extend(inds)

        # update flattened version of boundary indices list
        uc['bound_inds_flat'] = []
        for inds in uc['bound_inds']: uc['bound_inds_flat'].extend(inds)

        if figures in [1,2]:
            # plot unit cell to check if stubs are correct
            fig, ax = plt.subplots(figsize=(7,7))
            # ax.set_title(f'{group} ({shape})')
            fig.patch.set_facecolor("None")

            ax.scatter(*uc['points'].T, c='black')

            # background fundamental domain shape
            ax.fill(*fd["corners"].T, c='lightgrey', zorder=-1)  #, alpha=0.5,

            # background fundamental domain shape
            ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-2)  #, alpha=0.5

            x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
            edges0 = ax.plot(x, y, c='black')

            ax.set_aspect('equal')

            path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_stubs.png'))
            fig_nr += 1
            plt.axis('off')
            fig.savefig(path1)
            plt.close()
    else:  # no break
        raise Exception('oh no for loop too long')

    # %% [markdown]
    # ## Plot unit cell tiled 3×3
    if figures == 2:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_title(f'{group} ({shape}), stubs removed')
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        ax.fill(*uc["corners"].T, c='whitesmoke', alpha=0.5)

        # background fundamental domain shape
        ax.fill(*fd['corners'].T, c='whitesmoke', alpha=0.5)

        # loop over vecs[0] translations
        for i in range(3):
            # loop over vecs[1] translations
            for j in range(3):
                points_temp = hf.translate_points(uc["points"], i*vecs[0]+j*vecs[1])
                ax.scatter(*points_temp.T, alpha=0.5)  #, c='tab:orange')
                x, y = np.transpose(points_temp[uc["edges"].T], axes=[2,0,1])
                edges0 = ax.plot(x, y, alpha=0.3, c='tab:red')

        ax.set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell_3x3.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% [markdown]
    # ## Find cycles

    # %%
    # duplicate edges in other direction
    uc["edges_directional"] = np.concatenate((uc["edges"], np.flip(uc["edges"], axis=-1)), axis=0)

    # %%
    # %%
    uc['periodic_nodes'] = hf.list_equivalent_nodes(uc["bound_inds"],
                                            uc["linked_bounds"],
                                            len(uc["points"]))
    uc['periodic_edges'] = hf.list_equivalent_edges(uc["bound_inds"],
                                            uc["linked_bounds"],
                                            uc["edges_directional"], directional=True)
    uc['periodic_edges'] = np.array(uc['periodic_edges'])

    # %%
    triplets = []
    angles = []

    tol_g = 1e-6
    visited = np.zeros(len(uc["edges_directional"]), dtype=bool)
    holes = []
    angles_per_hole = []
    while not visited.all():
        e = np.where(~visited)[0][0]
        if verbose: print(f'================== NEW HOLE, starting with {e} ==================')
        visited[e] = True

        # equivalent edge also gets marked visited
        if len(uc['periodic_edges']) > 0:
            if e in uc['periodic_edges'][:, 0]:
                equiv_edge = uc['periodic_edges'][np.where(uc['periodic_edges'][:, 0]==e)[0][0], 1]
                if verbose:
                    print('e, equiv_edge', e, equiv_edge)
                visited[equiv_edge] = True
        done = False
        hole = [e,]
        angles_temp = []
        while not done:
            edges = uc["edges_directional"]
            edge = edges[e]
            if verbose:
                print('=============', edge, '=============')
            # vector that points along current edge
            edge_vec = -(uc["points"][edge[1]] - uc["points"][edge[0]])

            # find edges that connect to this edge
            bools = ((edges[:,0]==edges[e][1])
                            + np.isin(edges[:,0],uc['periodic_nodes'][edges[e][1]])
            )
            # only keep one copy of the connected edges
            hpe = uc['periodic_edges'][:len(uc['periodic_edges'])//2] # take half
            if len(hpe) > 0:
                bools[hpe[:, 0]] = bools[hpe[:, 1]]
                bools[hpe[:, 1]] = False
            inds = np.where(bools)[0]

            conn_edges = edges[inds]
            if verbose:
                print(conn_edges)

            # vectors pointing along connected edges
            conn_vecs = uc["points"][conn_edges[:, 1]] - uc["points"][conn_edges[:, 0]]

            # get all angles relative to x-axis
            theta1 = np.arctan2(*np.flip(edge_vec, axis=-1))
            theta2 = np.arctan2(*np.flip(conn_vecs, axis=-1).T)

            # angle of connected edges to current edge
            theta = (theta2 - theta1) % (2*np.pi)
            theta[np.abs(theta) < tol_g] = 2*np.pi
            if verbose:
                print(f'{theta/np.pi}π')
            for conn_edge, theta_temp in zip(conn_edges, theta):
                triplets.append([*edge, *conn_edge])
                angles.append(theta_temp)

            # find smallest angle (to go clockwise)
            e = inds[np.argmin(theta)]
            angles_temp.append(np.min(theta))
            if verbose:
                print('Choose:', e, edges[e])
            if visited[e] and (e == hole[0] or [e, hole[0]] in uc['periodic_edges']):
                done = True
                holes.append(hole)
                angles_per_hole.append(angles_temp)
            elif visited[e]:
                holes.append(hole)
                raise Exception(f'Next edge {e} is already visited but is not equal to the first edge {hole[0]}')
            else:
                visited[e] = True
                # equivalent edge also gets marked visited
                if len(uc['periodic_edges']) > 0:
                    if e in uc['periodic_edges'][:, 0]:
                        peri_edge = uc['periodic_edges'][np.where(uc['periodic_edges'][:, 0]==e)[0][0], 1]
                        if verbose:
                            print('e, peri_edge', e, peri_edge)
                        visited[peri_edge] = True
                hole.append(e)


    # %%
    # print equivalent nodes
    if verbose:
        print('equivalent nodes:')
        for i, asdf in enumerate(uc['periodic_nodes']):
            if len(asdf) > 0:
                print(i, asdf)

    # %%
    # print equivalent edges
    if verbose:
        print('equivalent edges:')
        for asdf in uc['periodic_edges']:
            if len(asdf) > 0:
                print(asdf[0], asdf[1], uc["edges_directional"][asdf[0]], uc["edges_directional"][asdf[1]])

    # %%
    print(f'Nr of holes: {len(holes)}')

    # %%
    # separate figure for each hole
    if figures == 2:
        for i, hole in enumerate(holes):
            fig = plt.figure()
            plt.scatter(*uc["points"].T, c='black', s=50, zorder=10)
            x, y = np.transpose(uc["points"][edges.T], axes=[2,0,1])
            edges0 = plt.plot(x, y, alpha=0.3, c='tab:red')

            x, y = np.transpose(uc["points"][uc["edges_directional"][hole].T], axes=[2,0,1])
            edges0 = plt.plot(x, y, #alpha=0.3,
                            c='tab:red', #linewidth=w
                            )
            for edge in uc["edges_directional"][hole]:
                plt.annotate("", xy=uc["points"][edge[1]], xytext=uc["points"][edge[0]],
                            arrowprops=dict(#arrowstyle="->",
                                            zorder=10
                                            )
                            )
            for i, point in enumerate(uc["points"]):
                plt.text(*(point+0.05), i)
            plt.gca().set_aspect('equal')

            path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_hole.png'))
            fig_nr += 1
            plt.axis('off')
            fig.savefig(path1)
            plt.close()

    # %% [markdown]
    # ## Adjust small angles
    # %%
    # per node, indicate which boundary it belongs to
    uc["n_points"] = len(uc["points"])
    uc["bound_per_node"] = [[] for i in range(uc["n_points"])]
    # iterate over fundamental domains
    for i, b_fd in enumerate(uc["bound_inds_per_fd"]):
        # iterate over boundaries
        for j, b in enumerate(b_fd):
            # iterate over indices of points on this boundary
            for k, ind in enumerate(b):
                # node ind is the kth node on the jth boundary of the ith fun. dom.
                uc["bound_per_node"][ind].append((i,j,k))

    # %%
    # Plot indicating nr of boundaries a point is on
    if figures == 2:
        max_c = []
        len_bs = []
        for bs in uc['bound_per_node']:
            _, c = np.unique([b[0] for b in bs], return_counts=True)
            if len(c) > 0:
                max_c.append(np.max(c))
            else:
                max_c.append(0)

            len_bs.append(len(bs))

        max_c = np.array(max_c)
        len_bs = np.array(len_bs)

        fig, ax = plt.subplots(figsize=(10,10))

        ax.scatter(*uc['points'][len_bs == 0].T, s=5, zorder=10, label='inner node')
        ax.scatter(*uc['points'][(len_bs != 0)*(max_c == 1)].T, s=10, zorder=10, label='boundary node')
        ax.scatter(*uc['points'][(len_bs != 0)*(max_c > 1)].T, s=15, zorder=10, label='corner node')

        x, y = np.transpose(uc["points"][uc["edges"].T], axes=[2,0,1])
        edges0 = ax.plot(x, y, alpha=0.3, c='black')

        # background fundamental domain shape
        ax.fill(*fd["corners"].T, c='whitesmoke', alpha=0.5, zorder=-1)

        ax.set_aspect('equal')
        ax.legend(title='count')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_nr_of_boundaries.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% shift nodes to reduce sharp angles
    SHIFT_FRAC = 0.01
    points_temp = np.copy(uc["points"])
    moved_points = []
    moved_points_b = []
    to_move_points = []

    # do multiple iterations of shifting
    nm = []
    for i in range(5):
        if verbose:
            print(f'=================== Iteration {i} of shifting ===================')
        points_temp_old = np.copy(points_temp)
        triplets, angles = hf.find_angles(uc)
        for trip, ang in zip(triplets, angles):
            vec1 = points_temp_old[trip[1]] - points_temp_old[trip[0]]
            vec2 = points_temp_old[trip[3]] - points_temp_old[trip[2]]


            # check if angle ABC is too small
            if verbose:
                print(f'{str(trip):20} {ang/np.pi:.20}π', end='')
                print(f', too small (< {1/7:.2}π)' if ang < (1/7)*np.pi else '')
            if ang < (1/7)*np.pi:

                # ======================== MOVE B inward ========================
                if not(trip[1] in nm or trip[2] in nm):
                    bs = uc["bound_per_node"][trip[1]]
                    # c: nr of boundaries of the same fundamental domain the point is on
                    _, c = np.unique([b[0] for b in bs], return_counts=True)

                    if len(bs) == 0:  # on no boundaries = an inner node
                        points_temp[trip[1]] = points_temp[trip[1]] - SHIFT_FRAC*vec1 + SHIFT_FRAC*vec2

                        moved_points.append(trip[1])
                        if verbose:
                            print(f'{trip[1]} moved!')

                    # if it's a boundary node, only move it along the boundary
                    elif np.max(c) == 1:

                        # calculate unit vector along boundary
                        fd_temp, b, _ = bs[0]
                        bound_vec = uc["bounds_per_fd"][fd_temp][b][1]
                        bound_vec_dir = bound_vec/np.linalg.norm(bound_vec)
                        if verbose:
                            print('old', points_temp[trip[1]])

                        # calculate shift along fundamental domain boundary
                        shift = (- SHIFT_FRAC*np.inner(vec1, bound_vec_dir)*bound_vec_dir
                                + SHIFT_FRAC*np.inner(vec2, bound_vec_dir)*bound_vec_dir)
                        points_temp[trip[1]] += shift

                        moved_points_b.append(trip[1])

                        # if the shifted node is a unit cell boundary node, shift its equivalent node as well
                        if len(equiv_nodes[trip[1]]) > 0:
                            for en in equiv_nodes[trip[1]]:
                                points_temp[en] += shift
                                moved_points_b.append(en)

                        if verbose:
                            print('new', points_temp[trip[1]])
                            print(f'{trip[1]} moved along boundary!')

                    # if it's in multiple boundaries of the same fundamental domain (therefore at a corner), don't move it at all
                    else:
                        to_move_points.append(trip[1])
                        if verbose:
                            print(f'{trip[1]} cannot move, to do')

                # ====================== MOVE A & C outward ======================
                    for ind_temp, vec_temp in zip([trip[0], trip[3]], [vec1, vec2]):
                        move_vec = SHIFT_FRAC*np.flip(vec_temp)  #/np.linalg.norm(vec_temp)
                        move_vec[0] *= -1

                        bs = uc["bound_per_node"][ind_temp]
                        # c: nr of boundaries of the same fundamental domain the point is on
                        _, c = np.unique([b[0] for b in bs], return_counts=True)

                        if len(bs) == 0:
                            points_temp[ind_temp] = points_temp[ind_temp] + move_vec

                            moved_points.append(ind_temp)
                            if verbose:
                                print(f'{ind_temp} moved!')

                        # if it's a boundary node, only move it along the boundary
                        elif np.max(c) == 1:

                            # calculate unit vector along boundary
                            fd_temp, b, _ = bs[0]
                            bound_vec = uc["bounds_per_fd"][fd_temp][b][1]
                            bound_vec_dir = bound_vec/np.linalg.norm(bound_vec)
                            if verbose:
                                print('old', points_temp[ind_temp])

                            # calculate shift along fundamental domain boundary
                            shift = np.inner(move_vec, bound_vec_dir)*bound_vec_dir
                            points_temp[ind_temp] += shift

                            moved_points_b.append(ind_temp)

                            # if the shifted node is a unit cell boundary node, shift its equivalent node as well
                            if len(equiv_nodes[ind_temp]) > 0:
                                for en in equiv_nodes[ind_temp]:
                                    points_temp[en] += shift
                                    moved_points_b.append(en)

                            if verbose:
                                print('new', points_temp[ind_temp])
                                print(f'{ind_temp} moved along boundary!')

                        # if it's in multiple boundaries of the same fundamental domain (therefore at a corner), don't move it at all
                        else:
                            to_move_points.append(ind_temp)
                            if verbose:
                                print(f'{ind_temp} cannot move')

        # check for crossed edges
        # crossed, cross_inds = hf.detect_crossed_edges(points_temp, uc['edges_directional'], return_indices=True)
        cross_inds = hf.crossing_edges(points_temp, uc['edges_directional'])
        # if crossed edges happen, reset to points_temp_old, and mark nodes as non-moving

        if len(cross_inds)> 1:  #crossed:
            if figures == 2:
                # Plot crossing edges
                fig, ax = plt.subplots(figsize=(7,7))
                ax.set_title(f'{group} ({shape}) Crossed edges')
                fig.patch.set_facecolor("None")

                ax.scatter(*uc["points"].T, c='tab:blue', label="old", s=100)
                ax.scatter(*points_temp.T, c='tab:orange', label="new")
                ax.scatter(*(points_temp[moved_points_b]).T, c='tab:red', label="new, moved along boundary")
                ax.scatter(*(points_temp[to_move_points]).T, c='tab:green', label="new, should move but couldn't")

                ax.fill(*uc["corners"].T, c='lightgrey', zorder=-2, alpha=0.2)
                # ax.fill(*uc["corners_per_fd"][0].T, c='whitesmoke', zorder=-1)

                x, y = np.transpose(uc["points"][uc["edges"].T], axes=[2,0,1])
                edges0 = ax.plot(x, y, alpha=0.3, c='tab:blue')

                x, y = np.transpose(points_temp[uc["edges"].T], axes=[2,0,1])
                edges0 = ax.plot(x, y, alpha=0.3, c='tab:orange')

                x, y = np.transpose(points_temp[uc["edges_directional"][cross_inds.flatten()].T], axes=[2,0,1])
                edges0 = ax.plot(x, y, alpha=0.3, c='black', linewidth=10)

                for j, point in enumerate(uc["points"]):
                    ax.text(*(point+0.02), j)

                ax.legend()
                ax.set_aspect('equal')

                path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_crossed_edges.png'))
                fig_nr += 1
                plt.axis('off')
                fig.savefig(path1)
                plt.close()

            points_temp = points_temp_old
            SHIFT_FRAC /= 2
            if verbose:
                print(f'Found crossed edges! Reducing SHIFT_FRAC to {SHIFT_FRAC} and trying again')

    # %% Plot to compare original and shifted
    if figures in [1,2]:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-2)  #, alpha=0.2)
        ax.fill(*fd["corners"].T, c='lightgrey', zorder=-1)
        # ax.fill(*uc["corners_per_fd"][0].T, c='whitesmoke', zorder=-1)

        # old points
        ax.scatter(*uc["points"].T, c='darkgrey', label="old", zorder=1)  #, s=100)  # c='tab:blue',

        # new points
        ax.scatter(*points_temp.T, c='black', label="new", zorder=2)

        # old edges
        x, y = np.transpose(uc["points"][uc["edges"].T], axes=[2,0,1])
        edges0 = ax.plot(x, y, c='darkgray', zorder=1)  #'tab:blue'), alpha=0.3,

        # new edges
        x, y = np.transpose(points_temp[uc["edges"].T], axes=[2,0,1])
        edges0 = ax.plot(x, y, c='black', zorder=2)

        # ax.legend()
        ax.set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_shifted.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()


    # %% Assign adjusted to the positions of the nodes, update angles
    uc["points"] = points_temp
    triplets, angles = hf.find_angles(uc)

    # %% [markdown]
    # ## Create list of equivalent holes

    # %%
    # get list of unique edges that are not duplicates/periodic copies of other edges
    edges2 = np.copy(uc["edges_directional"])
    for i, edge in enumerate(edges2):
        for j, node in enumerate(edge):
            if len(equiv_nodes[node]) != 0:
                # replace with lowest numbered equivalent node
                edges2[i][j] = min(node, *equiv_nodes[node])

    # %%
    # for each node, all fundamental domains it is a part of
    uc['fds_per_point'] = [[] for i in range(uc['n_points'])]
    for fd_i, inds in enumerate(uc['points_inds_per_fd']):
        for ind in inds:
            uc['fds_per_point'][ind].append(fd_i)

    # %%
    # for each edge, get which fundamental domains it is a member of
    uc['edge_fds'] = []
    for i, edge in enumerate(uc['edges_directional']):
        fds = set(uc['fds_per_point'][edge[0]]) & set(uc['fds_per_point'][edge[1]])
        uc['edge_fds'].append(list(fds))

    # %%
    # create list of 'prototype edges': edges in first fundamental domain
    # bools[i,j,k] = ith edge, whether its jth node (source or target) is equal to node k
    bools = (uc['edges_directional'][..., np.newaxis] == uc['points_inds_per_fd'][0].reshape(1, 1, -1))

    bools = bools.any(axis=-1) # check if node is equal to any of the ones in fd1
    bools = bools.all(axis=-1) # check if it's the case for both nodes of the edge

    proto_edges = uc['edges_directional'][bools]

    # %%
    equiv_proto_edges = hf.list_equivalent_edges(uc['bound_inds_per_fd'][0], fd['linked_bounds'], proto_edges, directional=True)

    # %%
    if len(equiv_proto_edges) != 0:
        # only keep one way around
        inds = np.argmin(equiv_proto_edges, axis=-1)
        equiv_proto_edges = equiv_proto_edges[np.where(inds == 0)]

    # deduplicate
    equiv_proto_edges = np.unique(equiv_proto_edges, axis=0)

    uc['proto_edges'] = proto_edges

    # %%
    # assumption which should hold:
    # second half of proto_edges are the reverse edges of the first half (i.e., edges flipped)

    # this is the case in uc['edges_directional'], so just make sure to maintain that order

    # %%
    uc["n_edges_d"] = len(uc['edges_directional'])

    # %%
    # for each edge, indicate which proto edge it corresponds to
    proto_inds = np.empty(uc["n_edges_d"], dtype=int)
    for i, edge in enumerate(uc['edges_directional']):
        fd_temp = uc['edge_fds'][i][0]
        j = np.where(uc['points_inds_per_fd'][fd_temp] == edge[0])[0][0]
        k = np.where(uc['points_inds_per_fd'][fd_temp] == edge[1])[0][0]

        bools = (uc['proto_edges'] == [j,k]).all(axis=-1)
        ind_temp = np.where(bools)[0][0]  # index of the proto edge that edge i corresponds to

        # if this proto_edge has an equivalent periodic edge, change it to that one
        if len(equiv_proto_edges) != 0:
            if ind_temp in equiv_proto_edges[:, 1]:
                ind_temp = equiv_proto_edges[np.where(equiv_proto_edges[:, 1] == ind_temp)[0][0], 0]
        proto_inds[i] = ind_temp

    if verbose:
        print('proto_inds:')
        print(proto_inds)
    uc['proto_inds'] = proto_inds

    # %%
    # plot equivalent edges in same color
    if figures == 2:
        fig = plt.figure()

        # plt.gca().set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # plot graph with same color for equivalent edges
        plt.scatter(*uc["points"].T, c='black', s=20, zorder=10)
        x, y = np.transpose(uc["points"][uc["edges_directional"].T], axes=[2,0,1])
        cmap = plt.get_cmap('tab20').colors
        for i in range(np.max(uc['proto_inds'])):
            edges0 = plt.plot(x[:,uc['proto_inds'] == i], y[:,uc['proto_inds'] == i], alpha=0.3, c=cmap[i % len(cmap)], linewidth=3)

        for i, point in enumerate(uc["points"]):
            plt.gca().text(*(point+0.02), i)

        plt.gca().fill(*uc["corners"].T, c='lightgrey', zorder=-2, alpha=0.2)

        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_equiv_edges.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()


    # %%
    tol = 1e-6
    unique_holes = []
    for j, hole in enumerate(holes):
        if verbose:
            print('========================================')
        proto_inds1 = proto_inds[hole]
        proto_inds2 = (np.flip(proto_inds1) + len(uc['proto_edges'])//2) % len(uc['proto_edges'])

        # after flipping, you have to do this again: if this proto_edge has an equivalent periodic edge, change it to that one
        if len(equiv_proto_edges) != 0:
            for i, ind_temp in enumerate(proto_inds2):
                if ind_temp in equiv_proto_edges[:, 1]:
                    ind_temp = equiv_proto_edges[np.where(equiv_proto_edges[:, 1] == ind_temp)[0][0], 0]
                proto_inds2[i] = ind_temp

        # compare hole to all previous unique holes
        for i, hole2 in enumerate(unique_holes):

            # if the lenghts of the holes are not the same then the holes are not the same, so immediately move on
            if len(hole2['edges']) != len(hole):
                continue
            if verbose:
                print(f'Compare {j} to {i}')

            proto_inds3 = proto_inds[hole2['edges']]
            if verbose:
                print(proto_inds1, proto_inds2, proto_inds3)

            proto_inds3 = np.tile(proto_inds3, 2)
            n = len(proto_inds1)

            match = False

            # search for proto_inds1 in proto_inds3
            inds = np.where(proto_inds1[0] == proto_inds3[:n])[0]
            for start_ind in inds:
                check = proto_inds3[start_ind:start_ind+n]
                if (check == proto_inds1).all():
                    angles1 = angles_per_hole[j]
                    angles3 = hole2['angles']
                    angles3 = np.tile(angles3, 2)
                    angles3 = angles3[start_ind:start_ind+n]
                    if (np.abs(angles1 - angles3) < tol).all():
                        # match!
                        if verbose:
                            print('-----------------------')
                            print('match!')
                            print(proto_inds1, proto_inds3, start_ind)
                            print(list(proto_inds1)[n-start_ind:] + list(proto_inds1)[:n-start_ind])
                            print(n-start_ind)
                        hole2['equiv_holes'].append({'edges': hole[n-start_ind:] + hole[:n-start_ind], 'reverse': False})

                        # print(proto_inds1[n-start_ind+1:] + proto_inds1[:n-start_ind+1])
                        if verbose:
                            print('matching hole:', uc['edges_directional'][hole2['equiv_holes'][-1]['edges']])
                        match = True
                        break

            if match:
                break

            # search for proto_inds2 in proto_inds3
            inds = np.where(proto_inds2[0] == proto_inds3[:n])[0]
            for start_ind in inds:
                check = proto_inds3[start_ind:start_ind+n]
                if (check == proto_inds2).all():
                    angles2 = np.flip(angles_per_hole[j]).tolist()
                    angles2 = np.array(angles2[1:] + angles2[:1])
                    angles3 = hole2['angles']
                    angles3 = np.tile(angles3, 2)
                    angles3 = angles3[start_ind:start_ind+n]
                    if (np.abs(angles2 - angles3) < tol).all():
                        # match!
                        if verbose:
                            print('-----------------------')
                            print('match with reversed!')
                            print(proto_inds2, proto_inds3, start_ind)
                            print(list(proto_inds2)[n-start_ind:] + list(proto_inds2)[:n-start_ind])
                            print(n-start_ind)
                        temp = list((np.flip(hole) + uc['n_edges_d']//2) % uc['n_edges_d'])
                        hole2['equiv_holes'].append({'edges': temp[n-start_ind:] + temp[:n-start_ind], 'reverse': True})

                        if verbose:
                            print('matching hole:', uc['edges_directional'][hole2['equiv_holes'][-1]['edges']])
                        match = True
                        break

            if match:
                break

        else: # no break -> new unique hole
            unique_holes.append({'edges': hole, 'equiv_holes': [], 'angles': angles_per_hole[j]})
            if verbose:
                print(f'unique hole {len(unique_holes)-1}: {hole}, which has proto_inds {proto_inds1}')

    # %%
    # separate figure for each unique hole
    if figures == 2:
        for hole in unique_holes:
            fig = plt.figure()

            # plot graph
            plt.scatter(*uc["points"].T, c='black', s=50, zorder=10)
            x, y = np.transpose(uc["points"][uc["edges"].T], axes=[2,0,1])
            edges0 = plt.plot(x, y, alpha=0.3, c='black')

            # plot hole
            for i, edge in enumerate(uc["edges_directional"][hole['edges']]):
                c = 'tab:blue'
                if i == 0:
                    c = 'tab:red'
                plt.annotate("", xy=uc["points"][edge[1]], xytext=uc["points"][edge[0]],
                            arrowprops=dict(#arrowstyle="->",
                                            zorder=10,
                                            color=c, alpha=1.0
                                            )
                            )

            # plot copies of holes in different color
            for hole2 in hole['equiv_holes']:
                for i, edge in enumerate(uc["edges_directional"][hole2['edges']]):
                    c = 'tab:green'
                    if i == 0:
                        c = 'tab:red'
                    plt.annotate("", xy=uc["points"][edge[1]], xytext=uc["points"][edge[0]],
                                arrowprops=dict(#arrowstyle="->",
                                                zorder=10,
                                                color=c, # 'tab:green',
                                                alpha=0.5
                                                )
                                )

            for i, point in enumerate(uc["points"]):
                plt.text(*(point+0.05), i)
            plt.gca().set_aspect('equal')

            path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_hole.png'))
            fig_nr += 1
            plt.axis('off')
            fig.savefig(path1)
        plt.close()

    # %% [markdown]
    # ## Add info for each unique hole

    # %%
    # add more info to each unique hole:
    # points for a polygon of the hole shape
    # edge vectors
    # edge lengths
    # edge orientations (angle with x-axis)
    # angles between subsequent edges

    for hole in unique_holes:
        if verbose:
            print('================================')

        # first node of first edge is starting point
        node1 = uc['edges_directional'][hole['edges'][0]][0]
        hole['poly_points'] = [uc['points'][node1]]
        hole['edge_vecs'] = []

        for i, edge_ind in enumerate(hole['edges']):
            edge = uc['edges_directional'][edge_ind]
            if verbose:
                print(edge, end='')
            prev_point = hole['poly_points'][-1]
            r = uc['points'][edge[1]] - uc['points'][edge[0]]
            hole['edge_vecs'].append(r)

            if i == len(hole['edges'])-1:
                if verbose:
                    print('\nCompare:', hole['poly_points'][0], prev_point+r)
            else:
                hole['poly_points'].append(prev_point + r)
                if verbose:
                    print(hole['poly_points'][-1])

        hole['edge_lens'] = np.linalg.norm(hole['edge_vecs'], axis=-1)

        # get all angles relative to x-axis
        phi = np.arctan2(*np.flip(hole['edge_vecs'], axis=-1).T)
        hole['edge_orientations'] = phi

        hole['n'] = len(hole['edges'])
        # angle of adjacent edges
        e2 = np.arange(hole['n']) # indices into edges_temp, e.g. [0,1,2,3]
        e1 = (e2-1) % hole['n']  # [3,0,1,2]
        # angle from e1 to e2 (but with e1 backward)
        dphi = (phi[e2] - (phi[e1]+np.pi)) % (2*np.pi)
        dphi[np.abs(dphi) < tol_g] = 2*np.pi

        if verbose:
            print(f'{dphi/np.pi}π')

        hole['angles'] = dphi

        # convert all lists to arrays (except 'equiv_holes', which would be ragged)
        for key in hole:
            if isinstance(hole[key], list):
                if key != 'equiv_holes':
                    hole[key] = np.asarray(hole[key])


    # %% [markdown]
    # ## Straight skeletons

    # %%
    if figures in [1,2]:
        # plot all straight skeletons in the same figure
        fig = plt.figure(figsize=(7,7))
        fig.patch.set_facecolor("None")

        plt.scatter(*uc["points"].T, c='black', s=50, zorder=10)
        x, y = np.transpose(uc["points"][uc["edges"].T], axes=[2,0,1])
        edges0 = plt.plot(x, y, c='black')  #, zorder=-10)  # , alpha=0.3

        pastels = plt.get_cmap('Pastel1').colors

        plt.gca().fill(*uc["corners"].T, c='whitesmoke', zorder=-2)
        plt.gca().fill(*fd["corners"].T, c='lightgrey', zorder=-1)

        for i, hole in enumerate(unique_holes):
            plt.fill(*hole['poly_points'].T, c=pastels[i % len(pastels)])

            points = [sg.Point2(*point) for point in hole['poly_points']]
            poly = sg.Polygon(points)

            # reverse list of points if necessary
            if poly.orientation().name == 'NEGATIVE':
                points = list(reversed(points))
                poly = sg.Polygon(points)
                rev = True
            elif poly.orientation().name == 'POSITIVE':
                rev = False
            else:
                raise ValueError('invalid polygon orientation')
            if verbose:
                print('\nArea', poly.area())

            # create straight skeleton
            try:
                skel = sg.skeleton.create_interior_straight_skeleton(poly)
            except RuntimeError as e:
                if verbose: print(repr(e), 'adjusting polygon and trying again')
                pp2 = np.copy(hole['poly_points'])

                # see which ones are identical
                vals, inv, c = hf.uniquetol(hole['poly_points'], tol, return_counts=True, return_inverse=True, axis=0)
                inds = np.where(c > 1)[0]
                # these ones are duplicates; adjust a little bit
                inds2 = np.where(np.isin(inv, inds))[0]

                vecs1 = hole['edge_vecs'][(inds2 - 1) % hole['n']]
                vecs2 = hole['edge_vecs'][inds2]

                pp2[inds2] = pp2[inds2] - 0.0001*vecs1 + 0.0001*vecs2

                points = [sg.Point2(*point) for point in pp2]
                poly = sg.Polygon(points)

                # reverse list of points if necessary
                if poly.orientation().name == 'NEGATIVE':
                    points = list(reversed(points))
                    poly = sg.Polygon(points)
                    hole['reverse_for_skel'] = True
                elif poly.orientation().name == 'POSITIVE':
                    hole['reverse_for_skel'] = False
                else:
                    raise ValueError('invalid polygon orientation') from e

                hole['poly'] = poly
                hole['area'] = float(poly.area())

                skel = sg.skeleton.create_interior_straight_skeleton(poly)

            for he in skel.halfedges:
                x = (float(he.vertex.point.x()), float(he.opposite.vertex.point.x()))
                y = (float(he.vertex.point.y()), float(he.opposite.vertex.point.y()))

                if he.is_bisector and he.vertex.id < len(points):
                    plt.plot(x, y, c='tab:red', linewidth=2, alpha=0.5)  #, zorder=-3)
                elif he.is_bisector and he.opposite.vertex.id < len(points):
                    # ignore, opposite of bisectors where vertex.id < n
                    pass
                elif he.is_bisector:
                    # bisectors from inner point to another inner point
                    plt.plot(x, y, c='tab:gray', linewidth=2, alpha=0.5)  #c='tab:blue') #, linewidth=10, zorder=-2) #, alpha=0.5)
                elif he.is_border:
                    pass
                    # plt.plot(x, y, c='tab:orange', linewidth=2)  #, zorder=-1) #, alpha=0.5)
                else:
                    # neither bisector nor border: ignore because are opposite of border edges
                    pass

                if verbose:
                    print('.', end='')

        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_straight_skeletons.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)

        plt.close()

    # %%
    # put all bisectors of the straight skeletons in unique_holes
    # as well as the area of each polygon/hole
    for hole in unique_holes:

        points = [sg.Point2(*point) for point in hole['poly_points']]
        poly = sg.Polygon(points)

        # reverse list of points if necessary
        if poly.orientation().name == 'NEGATIVE':
            points = list(reversed(points))
            poly = sg.Polygon(points)
            hole['reverse_for_skel'] = True
        elif poly.orientation().name == 'POSITIVE':
            hole['reverse_for_skel'] = False
            pass
        else:
            raise ValueError('invalid polygon orientation')

        poly = sg.Polygon(points)
        hole['poly'] = poly
        hole['area'] = float(poly.area())

        # create straight skeleton
        try:
            skel = sg.skeleton.create_interior_straight_skeleton(poly)
        except RuntimeError as e:
            if verbose: print(repr(e), 'adjusting polygon and trying again')
            pp2 = np.copy(hole['poly_points'])

            # see which ones are identical
            vals, inv, c = hf.uniquetol(hole['poly_points'], tol, return_counts=True, return_inverse=True, axis=0)
            inds = np.where(c > 1)[0]
            # these ones are duplicates; adjust a little bit
            inds2 = np.where(np.isin(inv, inds))[0]

            vecs1 = hole['edge_vecs'][(inds2 - 1) % hole['n']]
            vecs2 = hole['edge_vecs'][inds2]

            pp2[inds2] = pp2[inds2] - 0.0001*vecs1 + 0.0001*vecs2

            points = [sg.Point2(*point) for point in pp2]
            poly = sg.Polygon(points)

            # reverse list of points if necessary
            if poly.orientation().name == 'NEGATIVE':
                points = list(reversed(points))
                poly = sg.Polygon(points)
                hole['reverse_for_skel'] = True
            elif poly.orientation().name == 'POSITIVE':
                hole['reverse_for_skel'] = False
            else:
                raise ValueError('invalid polygon orientation') from e

            hole['poly'] = poly
            hole['area'] = float(poly.area())

            skel = sg.skeleton.create_interior_straight_skeleton(poly)

        hole['skel'] = skel

        # new nodes
        hole['bisectors_x'] = []
        hole['bisectors_r'] = []
        hole['bisectors_d'] = []
        for he in skel.halfedges:
            if he.is_bisector and he.vertex.id < len(points):
                x = float(he.opposite.vertex.point.x())
                y = float(he.opposite.vertex.point.y())
                x2 = float(he.vertex.point.x())
                y2 = float(he.vertex.point.y())
                hole['bisectors_x'].append([x,y])
                r = [x-x2, y-y2]
                hole['bisectors_r'].append(r)
                hole['bisectors_d'].append(np.linalg.norm(r))
            else:
                pass

        if hole['reverse_for_skel']:
            hole['bisectors_x'] = list(reversed(hole['bisectors_x']))
            hole['bisectors_r'] = list(reversed(hole['bisectors_r']))
            hole['bisectors_d'] = list(reversed(hole['bisectors_d']))

    # %% [markdown]
    # ## Add thickness
    # use
    # * edges2
    # * holes
    # * points

    # %%


    for k, hole in enumerate(unique_holes):
        if verbose:
            print(f'====================== hole {k} ======================')
            print(hole['edges'])
            print(uc['edges_directional'][hole['edges']])
        # decide if the hole will actually be a hole in the mesh or be filled
        # no hole if polygon too small, else 20% chance of being filled
        offset_polys = hole['skel'].offset_polygons(MIN_THICKNESS)

        if len(offset_polys) == 0:
            hole['filled'] = True
            print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because it\'s too small for an offset polygon of MIN_THICKNESS {MIN_THICKNESS}')
            continue

        if offset_polys[0].area() < MIN_AREA:
            hole['filled'] = True
            print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because area too small')
            continue
        elif (np.asarray(hole['bisectors_d']) < MIN_THICKNESS).any():
            hole['filled'] = True
            print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because one or more bisectors too short')
            continue
        elif len(offset_polys) > 1:
            hole['filled'] = True
            print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because offset polygon splits into multiple polygons')
            continue
        else:
            if rng.random() < PROB_RANDOM_FILL:
                hole['filled'] = True
                print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled randomly')
                continue

        hole['filled'] = False

        edges = np.asarray(hole['edges'])

        e2 = np.arange(hole['n']) # indices into edges_temp, e.g. [0,1,2,3]
        e1 = (e2-1) % hole['n']  # [3,0,1,2]
        hole['circ'] = np.stack((e1, e2), axis=-1) # can be either pairs of nodes or pairs of edges

        hole['thickness_vecs'] = []
        hole['inner_points'] = []
        hole['thickness'] = []

        proto_combis1 = uc['proto_inds'][hole['edges'][hole['circ']]]
        proto_combis2 = (np.flip(proto_combis1, axis=-1) + len(uc['proto_edges']) //2) % len(uc['proto_edges'])
        if verbose:
            print(proto_combis1)
            print(proto_combis2)

        # if this proto_edge has an equivalent periodic edge, change it to that one
        if len(equiv_proto_edges) != 0:
            for i, temp in enumerate(proto_combis2):
                for j, ind_temp in enumerate(temp):
                    if ind_temp in equiv_proto_edges[:, 1]:
                        proto_combis2[i][j] = equiv_proto_edges[np.where(equiv_proto_edges[:, 1] == ind_temp), 0]

        proto_combis = np.stack((proto_combis1, proto_combis2), axis=1)
        inds = np.argsort(proto_combis[:, :, 0], axis=1)
        val, inv = np.unique(np.take_along_axis(proto_combis, inds[...,np.newaxis], axis=1), return_inverse=True, axis=0)
        hole['unique_corner_ind'] = inv
        ran = rng.random(len(val)) # one random nr for each unique combination of edges

        hole['bisectors_r'] = np.array(hole['bisectors_r'])
        hole['bisectors_d'] = np.array(hole['bisectors_d'])

        for i in range(max(inv)+1): # iterate over unique corners
            new_d = np.min(hole['bisectors_d'][inv == i])
            if verbose:
                print('----------------------------------------')
                print('inv', inv)
                print(hole['bisectors_d'][inv == i])
                print('hole["bisectors_d"]', hole['bisectors_d'])
                print('new_d', new_d)
            hole['bisectors_r'][inv == i] = hole['bisectors_r'][inv == i]/hole['bisectors_d'][inv == i].reshape(-1, 1)*new_d
            hole['bisectors_d'][inv == i] = new_d
            if verbose:
                print('hole["bisectors_d"]', hole['bisectors_d'])

        # create thickness points
        for i, [e1, e2] in enumerate(hole['circ']):
            dphi_temp = hole['angles'][i]

            # middle point
            mid_point = hole['poly_points'][i]

            # construct new
            phi_new = (hole['edge_orientations'][e2] - dphi_temp/2) % (2*np.pi)
            if dphi_temp < np.pi:
                len_min = MIN_THICKNESS/np.sin(dphi_temp/2)
            else:
                len_min = MIN_THICKNESS/np.cos(dphi_temp/2-np.pi/2)

            len_max = MAX_REL_THICKNESS*hole['bisectors_d'][i]
            if len_min > len_max:
                hole['filled'] = True
                if verbose:
                    print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because len_max too small')
                break

            ra = ran[inv[i]]
            len_new = len_min + ra*(len_max - len_min)

            # len_new = rng.uniform(len_min, len_max)
            vec = np.array([len_new*np.cos(phi_new), len_new*np.sin(phi_new)])
            hole['thickness_vecs'].append(vec)
            hole['inner_points'].append(mid_point + vec)
            hole['thickness'].append(len_new)
        else:  # no break = not filled
            hole['thickness_vecs'] = np.array(hole['thickness_vecs'])
            hole['inner_points'] = np.array(hole['inner_points'])
            hole['thickness'] = np.array(hole['thickness'])

    # check if not all holes have been filled

    bools_filled = [hole['filled'] for hole in unique_holes]
    assert not all(bools_filled), 'all holes filled!'

    # %% Plot all inner boundaries
    if figures in [1,2]:
        fig = plt.figure(figsize=(7,7))
        fig.patch.set_facecolor("None")

        # plot graph
        plt.scatter(*uc['points'].T, c='black', s=50, zorder=10)
        x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
        edges0 = plt.plot(x, y, c='black')

        plt.gca().fill(*uc["corners"].T, c='whitesmoke', zorder=-2) #, alpha=0.2)
        plt.gca().fill(*fd["corners"].T, c='lightgrey', zorder=-1)  #, alpha=0.2)

        for i, hole in enumerate(unique_holes):
            if not hole['filled']:

                # plot hole in different color
                # plt.scatter(*hole['poly_points'].T)
                x, y = np.transpose(hole['poly_points'][hole['circ']], axes=[2,0,1])
                edges0 = plt.plot(x, y, #alpha=0.3,
                                c='tab:blue', linewidth=4
                                )

                # hole color
                plt.fill(*hole['poly_points'].T, c=pastels[i % len(pastels)], zorder=-0.5)

                # plot inner boundary
                plt.scatter(*hole['inner_points'].T, c='tab:orange')
                x, y = np.transpose(hole['inner_points'][hole['circ']], [2,1,0])
                plt.plot(x,y, c='tab:orange')

                bisectors = hole['bisectors_r']
                inner_points = hole['poly_points']

                x = np.stack((inner_points[:, 0], inner_points[:, 0] + bisectors[:, 0]), axis=1)
                y = np.stack((inner_points[:, 1], inner_points[:, 1] + bisectors[:, 1]), axis=1)

                plt.plot(x.T, y.T, c='tab:red', linewidth=2, zorder=-0.5, alpha=0.5)

        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_inner_boundaries.png'))


        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # Create spline points
    import __main__ as main
    if not hasattr(main, '__file__'):
        eval('%matplotlib qt')


    for hole in unique_holes:
        if verbose:
            print(f'================ HOLE {uc["edges_directional"][hole["edges"]][:,0]} ================')
        if hole['filled']:
            continue

        edges = np.asarray(hole['edges'])

        hole['spline_points'] = []

        e1 = np.arange(hole['n']) # indices into edges_temp, e.g. [0,1,2,3]
        e2 = (e1+1) % hole['n']  # [1,2,3,0]
        circ_temp = np.stack((e1, e2), axis=-1) # can be either pairs of nodes or pairs of edges

        temp_edges = hole['unique_corner_ind'][circ_temp]
        if verbose:
            print('Unique corners:', hole['unique_corner_ind'])
        temp_edges2 = np.sort(temp_edges, axis=-1)
        flip = temp_edges[:, 0] == temp_edges2[:, 0]
        symm = temp_edges[:, 0] == temp_edges[:, 1]

        # TO DO: this way of determining unique edges is not perfect: e.g. in p3, it can say that alternating identical edges are all identical (i.e., it says [0,0,0,0,0,0] instead of [0,1,0,1,0,1]). E.g., see p3_hexagonal_2024-05-22_15-27-41.446815. Use uc['proto_inds'] instead (but still take into account flipping). symm is not necessary and in fact is giving some holes mirror symmetry when they don't need it, see e.g. p4_square_2024-05-22_14-58-43.800779
        unique_edges, inv = np.unique(temp_edges2, axis=0, return_inverse=True)
        hole['unique_edges'] = inv
        hole['flip_edges'] = flip
        hole['symm_edges'] = symm
        ran = rng.random((len(unique_edges), 2)) # one random nr for each unique edge

        hole['inner_r'] = hole['inner_points'][e2] - hole['inner_points'][e1]
        hole['inner_d'] = np.linalg.norm(hole['inner_r'], axis=-1)

        # get all angles relative to x-axis
        phi = np.arctan2(*np.flip(hole['inner_r'], axis=-1).T)
        hole['inner_edge_orientations'] = phi

        # angle from e1 to e2 (but with e1 backward)
        dphi = (phi[hole['circ'][:, 1]] - (phi[hole['circ'][:, 0]]+np.pi)) % (2*np.pi)
        hole['inner_angles'] = np.copy(dphi)
        # make sure the angle is between 0 and pi (take smallest angle between the two edges)
        dphi[np.abs(dphi) > np.pi] = 2*np.pi - dphi[np.abs(dphi) > np.pi]

        if verbose:
            print('dphi:', dphi)

        hole['spline_point_dists'] = []

        # create spline points, iterating over edges of the hole
        for c1, c2 in circ_temp:
            if verbose:
                print('c1, c2:', c1, c2)
            # c1: nr of the first corner/nr of the edge (e.g. edge 0 starts at corner 0)
            # c2: nr of the second corner (E.g. second corner of edge 0 is corner 1)
            dd = hole['inner_d'][c1]

            min_d = MIN_RADIUS/np.tan(dphi[c1]/2)
            if verbose:
                print('min_d:', min_d)
            min_d2 = MIN_RADIUS/np.tan(dphi[c2]/2)
            if verbose:
                print('min_d2:', min_d2)
            max_d = dd-min_d2
            if verbose:
                print('d, min_d, max_d', dd, min_d, max_d)
            d_avail = dd - min_d - min_d2

            if not d_avail > MIN_SEP_ABS:
                hole['filled'] = True
                print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because cannot have corners with radius > {MIN_RADIUS}')
                break

            # the two random nrs for this edge
            ra = ran[inv[c1]]

            min_sep = max(MIN_SEP_ABS, MIN_SEP_REL*d_avail)
            rel_min_sep = min_sep/d_avail
            min_d = max(MIN_D_REL*dd, min_d)
            min_d2 = max(MIN_D_REL*dd, min_d2)
            max_d = dd-min_d2

            # if this edge is symmetric then only one random nr necessary
            if symm[c1]:
                if verbose:
                    print('symm edge!')
                max_d = (dd-min_sep)/2
                d = min_d + ra[0]*(max_d - min_d)
                d = np.array([d, dd - d])
            else:
                # turn the two random numbers in [0,1] into two random numbers in [0,1] with a separation of at least min_sep
                ra = np.sort(ra)
                ra2 = (1-rel_min_sep)*ra
                ra2[1] += rel_min_sep

                # flip if edge is mirrored
                if flip[c1]:
                    ra2 = 1 - np.flip(ra2)

                # turn the random nrs into a distance inbetween min_d and max_d
                d = min_d + ra2*(max_d - min_d)

            hole['spline_point_dists'].append(d.tolist())

            hole['spline_points'].append(hole['inner_points'][c1]
                                        + hole['inner_r'][c1]*d[0]/dd)
            hole['spline_points'].append(hole['inner_points'][c1]
                                        + hole['inner_r'][c1]*d[1]/dd)

        else:  # no break
            hole['spline_points'] = np.array(hole['spline_points'])

            points = [sg.Point2(*point) for point in hole['spline_points']]
            poly = sg.Polygon(points)
            if np.abs(float(poly.area())) < MIN_AREA2:
                hole['filled'] = True
                print(f'hole {uc["edges_directional"][hole["edges"]][:,0]} will be filled because area too small')

    # %%
    # Plot all unique spline polygons in the same figure
    if figures in [1,2]:
        fig = plt.figure(figsize=(7,7))
        fig.patch.set_facecolor("None")

        # plot graph
        plt.scatter(*uc['points'].T, c='black', s=50, zorder=10)
        x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
        edges0 = plt.plot(x, y, c='black')  # , alpha=0.3

        plt.gca().fill(*uc["corners"].T, c='whitesmoke', zorder=-2) #, alpha=0.2)
        plt.gca().fill(*fd["corners"].T, c='lightgrey', zorder=-1)  #, alpha=0.2)

        for hole in unique_holes:
            if not hole['filled']:
                # plot hole
                # plt.scatter(*hole['poly_points'].T)
                x, y = np.transpose(hole['poly_points'][hole['circ']], axes=[2,0,1])
                edges0 = plt.plot(x, y, #alpha=0.3,
                                c='tab:blue', linewidth=4
                                )

                # plot inner boundary
                plt.scatter(*hole['inner_points'].T, c='tab:orange')
                x, y = np.transpose(hole['inner_points'][hole['circ']], [2,1,0])
                plt.plot(x,y, c='tab:orange')

                # plot spline boundary
                plt.scatter(*hole['spline_points'].T, c='tab:green', zorder=10)
                plt.fill(*hole['spline_points'].T, c='tab:green', alpha=0.5, zorder=9)

        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_spline_polygons.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # Check volume fraction

    hole_area = 0.0
    for i, hole in enumerate(unique_holes):
        if not hole['filled']:
            points = [sg.Point2(*point) for point in hole['spline_points']]
            poly = sg.Polygon(points)
            area = np.abs(float(poly.area()))

            if verbose:
                print(f"Hole{i}: Area {area}, {len(hole['equiv_holes']) + 1} copies")
            hole_area += area*(len(hole['equiv_holes']) + 1)
    total_area = 1.0  # uc['n_fds']*1.0
    vol_frac = 1-hole_area/total_area

    if verbose:
        print('Total area:', total_area)
        print(f'Volume occupied = {vol_frac*100:.1f}%')
        print(f'Volume vacant = {(1-vol_frac)*100:.1f}%')

    if vol_frac > MAX_DENSITY:
        raise Exception(f'Volume fraction is {vol_frac}, which is too high')

    # %%
    # check again if not all holes have been filled

    bools_filled = [hole['filled'] for hole in unique_holes]
    assert not all(bools_filled), 'all holes filled!'

    # %%
    # copy holes and create a polygon for each hole
    for hole_proto in unique_holes:
        if verbose:
            print('===============================================')
        if hole_proto['filled']:
            continue
        # CREATE ALL EQUIVALENT HOLES
        all_holes = [{'edges': hole_proto['edges'], 'reverse':False}] + hole_proto['equiv_holes']

        # create polygon that touches node 1 for each equivalent hole
        for hole in all_holes:

            # first node of first edge is starting point
            node1 = uc['edges_directional'][hole['edges'][0]][0]
            hole['poly_points'] = [uc['points'][node1]]
            hole['edge_vecs'] = []

            for i, edge_ind in enumerate(hole['edges']):
                edge = uc['edges_directional'][edge_ind]
                prev_point = hole['poly_points'][-1]
                r = uc['points'][edge[1]] - uc['points'][edge[0]]
                hole['edge_vecs'].append(r)

                if i == len(hole['edges'])-1:
                    pass
                else:
                    hole['poly_points'].append(prev_point + r)

            hole['edge_lens'] = np.linalg.norm(hole['edge_vecs'], axis=-1)

        hole_proto['all_holes'] = []

        # CREATE ALL PERIODIC COPIES OF ALL EQUIVALENT HOLES
        # by creating, for each node of each equivalent hole, a polygon that touches it
        for hole in all_holes:
            # periodic copies (shifted along lattice vectors)
            for a in [-1, 0, 1]:
                for b in [-1, 0, 1]:
                    # periodic copies, if hole is not in one piece
                    for i, e in enumerate(hole['edges']):
                        h = copy.deepcopy(hole)

                        point_ind = uc['edges_directional'][e][0]
                        h['poly_points'] += (-h['poly_points'][i] +uc['points'][point_ind])

                        h['poly_points'] += uc['lattice vectors'][0]*a + uc['lattice vectors'][1]*b

                        hole_proto['all_holes'].append(h)

        # DEDUPLICATE COPIES
        app = np.stack([h['poly_points'] for h in hole_proto['all_holes']], axis=0)
        _, inds = hf.uniquetol(app, 1e-6, return_index=True, axis=0)

        hole_proto['all_holes'] = [hole_proto['all_holes'][ind] for ind in inds]

        if verbose:
            for hole in hole_proto['all_holes']:
                print('------------------------------------------')
                for key in hole:
                    print(key, hole[key])

    # %%
    # create inner boundary for all copies of holes
    for hole_proto in unique_holes:
        if hole_proto['filled']:
            continue


        # FOR EACH COPY OF THIS HOLE, CREATE THE INNER BOUNDARY
        for hole in hole_proto['all_holes']:

            # edges of current hole (indices into points)
            edges_temp = uc['edges_directional'][hole['edges']]

            # get all angles relative to x-axis
            phi = np.arctan2(*np.flip(hole['edge_vecs'], axis=-1).T)
            hole['edge_orientations'] = phi

            hole['n'] = len(hole['edges'])
            # angle of adjacent edges
            e2 = np.arange(hole['n']) # indices into edges_temp, e.g. [0,1,2,3]
            e1 = (e2-1) % hole['n']  # [3,0,1,2]
            # angle from e1 to e2 (but with e1 backward)
            dphi = (phi[e2] - (phi[e1]+np.pi)) % (2*np.pi)
            dphi[np.abs(dphi) < tol_g] = 2*np.pi

            if verbose:
                print(f'{dphi/np.pi}π')

            hole['angles'] = dphi

            # convert all lists to arrays (except 'equiv_holes', which would be ragged)
            for key in hole:
                if isinstance(hole[key], list):
                    if key != 'equiv_holes':
                        hole[key] = np.asarray(hole[key])

            hole['thickness_vecs'] = []
            hole['inner_points'] = []
            hole['thickness'] = []

            # create thickness points
            for i, [e1, e2] in enumerate(hole_proto['circ']):
                dphi_temp = hole['angles'][i]

                # middle point
                mid_point = hole['poly_points'][i]

                # construct new
                phi_new = (hole['edge_orientations'][e2] - dphi_temp/2) % (2*np.pi)

                len_new = hole_proto['thickness'][i]  # rng.uniform(len_min, len_max)
                vec = np.array([len_new*np.cos(phi_new), len_new*np.sin(phi_new)])
                if hole['reverse']:
                    vec = -vec
                hole['thickness_vecs'].append(vec)
                hole['inner_points'].append(mid_point + vec)
                hole['thickness'].append(len_new)

            hole['thickness_vecs'] = np.array(hole['thickness_vecs'])
            hole['inner_points'] = np.array(hole['inner_points'])
            hole['thickness'] = np.array(hole['thickness'])

    # %% Create spline points for all copies of all holes
    for hole_proto in unique_holes:
        if hole_proto['filled']:
            continue

        # FOR EACH COPY OF THIS HOLE, CREATE THE INNER BOUNDARY
        for hole in hole_proto['all_holes']:

            edges = np.asarray(hole['edges'])

            hole['spline_points'] = []

            e1 = np.arange(hole['n']) # indices into edges_temp, e.g. [0,1,2,3]
            e2 = (e1+1) % hole['n']  # [1,2,3,0]

            hole['inner_r'] = hole['inner_points'][e2] - hole['inner_points'][e1]
            # create spline points, iterating over edges of the hole
            for i, d in enumerate(hole_proto['spline_point_dists']):
                hole['spline_points'].append(hole['inner_points'][i]
                                            + hole['inner_r'][i]*d[0]/hole_proto['inner_d'][i])
                hole['spline_points'].append(hole['inner_points'][i]
                                            + hole['inner_r'][i]*d[1]/hole_proto['inner_d'][i])

            hole['spline_points'] = np.array(hole['spline_points'])

    # %% Plot all inner boundaries per hole, for all copies of that hole
    if figures == 2:
        for hole_proto in unique_holes:
            if hole_proto['filled']:
                continue

            fig = plt.figure(figsize=(7,7))

            # plot graph
            plt.scatter(*uc['points'].T, c='black', s=50, zorder=10)
            x, y = np.transpose(uc['points'][uc['edges'].T], axes=[2,0,1])
            edges0 = plt.plot(x, y, c='black')  # , alpha=0.3

            # FOR EACH COPY OF THIS HOLE, PLOT THE HOLE
            for hole in hole_proto['all_holes']:
                # plot hole in different color
                plt.scatter(*hole['poly_points'].T, c='gray')
                x, y = np.transpose(hole['poly_points'][hole_proto['circ']], axes=[2,0,1])
                edges0 = plt.plot(x, y, #alpha=0.3,
                                c='tab:blue', linewidth=4
                                )

                # plot inner boundary
                plt.scatter(*hole['inner_points'].T, c='tab:orange')
                x, y = np.transpose(hole['inner_points'][hole_proto['circ']], [2,1,0])

                if hole['reverse']:
                    plt.plot(x,y, c='tab:green')
                else:
                    plt.plot(x,y, c='tab:orange')

            # label points
            for i, point in enumerate(uc['points']):
                plt.text(*(point+0.03), i)
            plt.gca().set_aspect('equal')

            path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_hole_copies.png'))
            fig_nr += 1
            plt.axis('off')
            fig.savefig(path1)
            plt.close()

    # %% Plot inner boundaries and spline points, all in the same figure
    if figures in [1,2]:
        fig = plt.figure(figsize=(7,7))

        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        plt.gca().fill(*uc["corners"].T, c='whitesmoke', zorder=-2)

        # background fundamental domain shape
        plt.gca().fill(*fd['corners'].T, c='lightgrey', zorder=-1)

        for hole_proto in unique_holes:
            if hole_proto['filled']:
                continue

        plt.gca().set_aspect('equal')

        for hole_proto in unique_holes:
            if hole_proto['filled']:
                continue

            for hole in hole_proto['all_holes']:
                # plot inner boundary
                plt.scatter(*hole['inner_points'].T, c='tab:orange')
                x, y = np.transpose(hole['inner_points'][hole_proto['circ']], [2,1,0])
                plt.plot(x,y, c='tab:orange')

        for hole_proto in unique_holes:
            if hole_proto['filled']:
                continue

            for hole in hole_proto['all_holes']:
                # plot spline boundary
                plt.scatter(*hole['spline_points'].T, c='tab:green', zorder=10)
                plt.fill(*hole['spline_points'].T, c='tab:green', alpha=0.5, zorder=9)

        plt.gca().set_xlim(uc_plot_lims[0])
        plt.gca().set_ylim(uc_plot_lims[1])
        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_all_spline_polygons.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # Plot final geometry
    if figures == 2:
        fig = plt.figure(figsize=(7,7))
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        plt.fill(*uc["corners"].T, c='whitesmoke')

        # background unit cell shape
        plt.fill(*fd['corners'].T, c='lightgrey')  # c='tab:orange')

        for i, hole_proto in enumerate(unique_holes):
            if hole_proto['filled']:
                continue
            for j, hole in enumerate(hole_proto['all_holes']):
                # plt.fill(*hole['inner_points'].T, c='whitesmoke', alpha=0.3)
                plt.fill(*hole['spline_points'].T, c='tab:green', alpha=0.5)  #c='gray')
                plt.scatter(*hole['spline_points'].T, zorder=10, label=f'{i} {j} {hole["spline_points"].shape}', s=5)
                if verbose:
                    print(hole['spline_points'].shape)

        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(uc_plot_lims[0])
        plt.gca().set_ylim(uc_plot_lims[1])

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_final_unit_cell.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% [markdown]
    # ## Create mesh

    # %%
    all_points = []
    all_lines = []
    all_splines = []
    all_loops = []
    all_surfaces = []
    all_periodic_lines = []

    po = 0  # point index
    li = 0 # line index
    sp = 0 # spline index
    lo = 0 # loop index
    su = 0 # surface index

    # add boundary points, lines, loops
    for i, b in enumerate(fd['bounds']):
        all_points.append(list(b[0]))
        po += 1

        if po > 1:
            all_lines.append(['Line', [po-1, po]])
            li += 1
    all_lines.append(['Line', [po, 1]])
    li += 1

    for b, lb, flip in fd['linked_bounds']:
        if lb is not None:
            if flip == -1:
                all_periodic_lines.append([b+1, -(lb+1)])
            elif flip == +1:
                all_periodic_lines.append([b+1, lb+1])

    all_loops.append(['Line', list(range(1,li+1))])
    lo += 1

    # %% Add splines (cubic Bezier)
    # add hole points, lines, loops
    for hole_proto in unique_holes:
        if hole_proto['filled']:
            continue
        for hole in hole_proto['all_holes']:
            hole['control_points'] = hf.interpolate_bezier4(hole['spline_points'], closed=True)
            li_old = li
            po_old = po
            all_points.extend(list(hole['control_points']))
            po += len(hole['control_points'])
            for i, point in enumerate(hole['spline_points']):
                # e.g. [5, 6, 7, 8], [8, 9, 10, 11], [11, 12, 13, 5]
                if i == len(hole['spline_points'])-1:  # last point
                    all_lines.append(['Bezier', list(range(po_old+1+i*3, po_old+1+i*3+3)) + [po_old+1]])
                else:
                    all_lines.append(['Bezier', list(range(po_old+1+i*3, po_old+1+i*3+4))])
                li += 1

            all_loops.append(['Curve', list(range(li_old+1, li+1))])
            lo += 1

    # %% Check if holes are not too close to each other

    # per hole, make a list of control points of each segment
    # and a list of indices, indicating which hole the segment belongs to
    control_points = []
    inds = []
    i = 0
    for proto_hole in unique_holes:
        if not proto_hole['filled']:
            for hole in proto_hole['all_holes']:

                # 1st segment is points 0,1,2,3, 2nd segment is 3,4,5,6, etc.
                temp = hole['control_points'].reshape(-1, 3, 2)
                temp2 = np.concatenate((temp[1:, [0]], temp[[[0]], [0]]), axis=0)
                temp3 = np.concatenate((temp, temp2), axis=1)

                control_points.extend(temp3)
                inds.extend([i]*temp3.shape[0])

                i += 1

    control_points = np.array(control_points)
    inds = np.array(inds)

    segment_pairs = hf.bezier4_too_close(control_points, MIN_THICKNESS*0.9, n=64, ignore_joined_ends=True)

    hole_inds = inds[segment_pairs]
    segment_pairs = segment_pairs[np.where(hole_inds[:, 0] != hole_inds[:, 1])]

    if len(segment_pairs) > 0:
        raise Exception(f'Holes too close together')

    # %%
    if figures in [1,2]:
        # Plot final geometry with Bézier control points
        fig = plt.figure(figsize=(7,7))
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        plt.fill(*uc["corners"].T, c='whitesmoke')

        # background unit cell shape
        plt.fill(*fd['corners'].T, c='lightgrey')  # c='tab:orange')

        for i, hole_proto in enumerate(unique_holes):
            if hole_proto['filled']:
                continue
            for j, hole in enumerate(hole_proto['all_holes']):
                # plt.fill(*hole['inner_points'].T, c='whitesmoke', alpha=0.3)

                plt.fill(*hole['spline_points'].T, c='tab:green', alpha=0.4,
                         zorder=9)  # c='gray')

                plt.scatter(*hole['control_points'].T, zorder=10, c='black', s=20)  # s=5)
                plt.scatter(*hole['spline_points'].T, zorder=10, c='tab:green')

                p1 = hole['control_points'][2::3]
                p2 = hole['control_points'][1::3]
                p1 = np.concatenate((p1[[-1]], p1[:-1]), axis=0)
                plt.plot(np.stack((p1[:,0], p2[:,0]), axis=0),
                        np.stack((p1[:,1], p2[:,1]), axis=0), c='black')

                codes = (
                    [mpath.Path.MOVETO]
                    + [mpath.Path.CURVE4]*len(hole['control_points'])
                    + [mpath.Path.CLOSEPOLY]
                )
                verts = (
                    hole['control_points'].tolist()
                    + hole['control_points'][[0]].tolist()
                    + [[0,0]]
                )
                patch = mpatches.PathPatch(mpath.Path(verts, codes), facecolor='tab:green', alpha=0.4, edgecolor='none', zorder=9.5)
                plt.gca().add_patch(patch)

        plt.gca().set_aspect('equal')

        plt.gca().set_xlim(uc_plot_lims[0])
        plt.gca().set_ylim(uc_plot_lims[1])

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_beziergons.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # Write to .geo file
    # write to gmsh input file
    file = hf.new_path(os.path.join(save_dir, f'{name}.geo'))
    with open(file, 'w', encoding='utf-8') as f:
        f.write('SetFactory("OpenCASCADE");\n')

        for i, point in enumerate(all_points):
            f.write(f'Point({i+1}) = {{{point[0]}, {point[1]}, 0, {CLMAX}}};\n')

        for i, [typ, line] in enumerate(all_lines):
            if typ == 'Line':
                f.write(f'Line({i+1}) = {{')
                f.write(', '.join([str(elem) for elem in line]))
                f.write(f'}};\n')
            elif typ == 'BSpline':
                f.write(f'BSpline({i+1}) = {{')
                f.write(', '.join([str(elem) for elem in line]))
                f.write(f'}};\n')
            elif typ == 'Bezier':
                f.write(f'Bezier({i+1}) = {{')
                f.write(', '.join([str(elem) for elem in line]))
                f.write(f'}};\n')
            elif typ == 'Spline':
                f.write(f'Spline({i+1}) = {{')
                f.write(', '.join([str(elem) for elem in line]))
                f.write(f'}};\n')
            else:
                raise ValueError(f'{typ} is not a valid line type')


        for b, lb in all_periodic_lines:
            f.write(f'Periodic Line{{{b}}} = {{{lb}}};\n')

        for i, [typ, loop] in enumerate(all_loops):
            if typ == 'Line':
                f.write(f'Line Loop({i+1}) = {{')
                f.write(', '.join([str(elem) for elem in loop]))
                f.write(f'}};\n')
            elif typ == 'Curve':
                f.write(f'Curve Loop({i+1}) = {{')
                f.write(', '.join([str(elem) for elem in loop]))
                f.write(f'}};\n')
            else:
                raise ValueError(f'{typ} loop is not a valid loop type')

        # create surfaces
        for i, loop in enumerate(all_loops):
            f.write(f'Plane Surface({i + 1}) = {{{i + 1}}};\n''')

        # Boolean Difference
        f.write('f1() = BooleanDifference')
        f.write(f'{{ Surface{{{1}}}; Delete; }}{{ Surface{{')
        for j in range(len(all_loops) - 2):
            f.write(f'{j+2},')
        f.write(f'{len(all_loops)}}}; Delete; }};\n')

        f.write('Mesh.ElementOrder = 2;\n')


    # %% [markdown]
    # Run gmsh on created .geo file with
    # Command example: 'gmsh', file, '-o', file2, '-2', '-clmax', str(CLMAX)]
    # ['gmsh', 'data/myMeshes/mymesh_graph_test_p3_00.geo', '-o', 'data/myMeshes/mymesh_graph_test_p3_00.msh', '-2', '-clmax', '0.1']
    # Explanation: gmsh: program to call, should be in path
    # file: path to input file
    # -o: specifying output file
    # file2: path to output file
    # -2: apparently indicates you want to use the command line interface of gmsh, without this it will open the gui
    # -clmax: specifying max element size
    # str(CLMAX): max element size

    file2 = hf.new_path(os.path.join(save_dir, f'{name}.msh'))
    # try:
    subprocess.run(['gmsh', file, '-o', file2, '-2', '-clmax', str(CLMAX)], check=True, timeout=100)

    # %% [markdown]

    mesh = meshio.read(file2)
    if verbose:
        print(mesh)

    # %%
    # ## Import mesh


    def import_gmsh_mesh(file_path, element_type):
        # Load the Gmsh mesh from the .msh file
        mesh = meshio.read(file_path)

        # Access mesh information
        points = mesh.points[:, :2]
        cells = mesh.cells
        cell_data = mesh.cell_data

        elements = np.array([], dtype=int).reshape(0, 6)
        for cell in cells:
            if cell.type == element_type:
                elements = np.append(elements, cell.data, axis=0)

        return points, cells, cell_data, elements

    # %%
    mesh_points, _, _, elements = import_gmsh_mesh(file2, 'triangle6')

    with open(hf.new_path(os.path.join(save_dir, f'{name}_fd.pkl')), 'wb') as f:
        pickle.dump({'p': mesh_points, 't': elements}, f)


    # %%
    # Remove unused nodes
    to_keep, inv = np.unique(elements, return_inverse=True)
    mesh_points = mesh_points[to_keep]

    # renumber elements
    temp = np.full(elements.max()+1, -1, dtype=int)
    temp[to_keep] = np.arange(len(to_keep))
    elements = temp[elements]

    # %%
    # Check volume fraction again, this time of the actual mesh

    temp = mesh_points[elements]
    # temp = np.transpose(temp, axes=[0,2,1])
    temp = temp[..., [0,3,1,4,2,5], :]
    filled_area = np.sum(hf.polygon_area(temp))
    total_area = 1.0/uc['n_fds']
    vol_frac2 = filled_area/total_area

    if verbose:
        print(f'Volume occupied = {vol_frac2*100:.1f}%')
        print(f'Volume vacant = {(1-vol_frac2)*100:.1f}%')

    # %% Turn fundamental domain into unit cell
    all_mesh_points = []
    all_elements = []
    inds_per_fd = []
    n = len(mesh_points)
    for j, copy1 in enumerate(uc['transforms']):

        p2 = np.copy(mesh_points)

        for transform in copy1:

            if transform[0] == 'T':
                p2 = hf.translate_points(p2, eval(transform[1]))
            elif transform[0] == 'R':
                degrees = np.array(eval(transform[2]))
                p2 = hf.rotate_points(p2, eval(transform[1]), degrees/360*2*np.pi)
            elif transform[0] == 'M':
                p2 = hf.mirror_points(p2, eval(transform[1]), eval(transform[2]))
            else:
                raise ValueError(f'transform {transform[0]} is not a valid transform type, choose from T, R, M (translate, rotate, mirror)')

        all_mesh_points.append(np.copy(p2))
        all_elements.append(np.copy(elements) + j*n)
        inds_per_fd.append(np.arange(n) + j*n)

    all_mesh_points = np.concatenate(all_mesh_points, axis=0)
    all_elements = np.concatenate(all_elements, axis=0)
    inds_per_fd = np.array(inds_per_fd)
    if verbose:
        print('all_mesh_points.shape', all_mesh_points.shape)
        print('all_elements.shape', all_elements.shape)
        print('inds_per_fd.shape', inds_per_fd.shape)

    # %% Deduplicate points
    start_time_dedup = time.time()
    # deduplicate
    all_mesh_points, inds, inv, c = hf.uniquetol(all_mesh_points, tol=1e-4, return_counts=True, return_index=True, return_inverse=True, axis=0)
    if verbose:
        print(f'Time for deduplication: {time.time() - start_time_dedup:.4} seconds')
    all_elements = inv[all_elements]
    inds_per_fd = inv[inds_per_fd]

    # %%
    # Plot counts
    if figures == 2:
        fig = plt.figure(figsize=(10,10))
        temp = all_mesh_points[all_elements]
        temp = np.transpose(temp, axes=[0,2,1])
        temp = temp[..., [0,3,1,4,2,5]]
        temp = temp.reshape(-1, temp.shape[-1])
        plt.fill(*temp, c='whitesmoke')
        plt.title('counts per point, should be =/= 1 at and only at overlapping fundamental domain boundaries')

        for count in np.unique(c):
            plt.scatter(*all_mesh_points[c == count].T, s=count*5, zorder=10, label=count)

        plt.gca().set_aspect('equal')
        plt.legend(title='count')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_counts.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # Plot unit cell mesh
    if figures in [1,2]:
        fig = plt.figure(figsize=(10,10))
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        plt.fill(*uc["corners"].T, c='whitesmoke')

        # background unit cell shape
        plt.fill(*fd['corners'].T, c='lightgrey')  # c='tab:orange')

        # plot filled triangles
        temp = all_mesh_points[all_elements]
        temp = np.transpose(temp, axes=[0,2,1])
        temp = temp.reshape(-1, temp.shape[-1])
        temp = temp[..., [0,3,1,4,2,5]]  # order the nodes counterclockwise
        plt.fill(*temp, alpha=0.5)  # c='whitesmoke')

        # plot points
        plt.scatter(*all_mesh_points.T, s=1, zorder=10, c='black')

        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_unit_cell_mesh.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)

        plt.close()

    # %%
    # Plot unit cell 2×2
    if figures in [1,2]:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # background fundamental domain shape
        ax.fill(*uc["corners"].T, c='whitesmoke', zorder=-11)

        # background fundamental domain shape
        ax.fill(*fd['corners'].T, c='lightgrey', zorder=-10)

        # loop over vecs[0] translations
        for i in range(2):
            # loop over vecs[1] translations
            for j in range(2):
                points_temp = hf.translate_points(all_mesh_points, i*vecs[0]+j*vecs[1])

                # plot filled triangles
                temp = points_temp[all_elements]
                temp = np.transpose(temp, axes=[0,2,1])
                temp = temp.reshape(-1, temp.shape[-1])
                temp = temp[..., [0,3,1,4,2,5]]
                if i == 0 and j == 0:
                    plt.fill(*temp)  #, alpha=0.5)
                else:
                    plt.fill(*temp, c='tab:orange')

        ax.set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_RVE.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    tol = 1e-4

    # find new boundary nodes
    mesh_bounds = []
    for b_start, b_vec in uc['bounds']:
        mesh_bounds.append([])
        bools_onbound = hf.iscollinear(all_mesh_points, b_start, b_start+b_vec, tol_g=1e-4)
        mesh_bounds[-1] = np.where(bools_onbound)[0]

        # sort them in the correct order
        if np.abs(b_vec[0]) > tol:
            temp = np.argsort(all_mesh_points[mesh_bounds[-1], 0]/b_vec[0])
        elif np.abs(b_vec[1]) > tol:
            temp = np.argsort(all_mesh_points[mesh_bounds[-1], 1]/b_vec[1])
        else:
            raise ValueError(f"invalid value of b_vec: {b_vec}")
        mesh_bounds[-1] = mesh_bounds[-1][temp].tolist()

    mesh_bounds

    # %% Plot to check boundary nodes
    if figures == 2:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        plt.scatter(*all_mesh_points.T, alpha=0.3, s=5)

        temp = all_elements[:, [0,3,1,4,2,5,0]]
        x, y = np.transpose(all_mesh_points[temp.T], axes=[2,0,1])
        edges0 = plt.plot(x, y, alpha=0.1, c='tab:red', zorder=-1)

        for inds in mesh_bounds:
            if verbose:
                print(len(inds))
            plt.scatter(*all_mesh_points[inds].T, s=5)

        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_check_boundary_nodes.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %% flip mirrored elements
    temp = all_mesh_points[all_elements]
    temp = temp[..., [0,3,1,4,2,5], :]
    signed_areas = hf.polygon_area(temp, signed=True)
    print("all_elements[signed_areas < 0].shape:")
    print(all_elements[signed_areas < 0].shape)
    all_elements[signed_areas < 0] = all_elements[signed_areas < 0][:, [2,1,0,4,3,5]]

    # %% Save things
    for i in range(len(unique_holes)):
        try:
            unique_holes[i]['poly'] = unique_holes[i]['poly'].coords
        except AttributeError as e:
            if verbose:
                print(repr(e))

    for i in range(len(unique_holes)):
        try:
            del unique_holes[i]['skel']
        except KeyError as e:
            if verbose:
                print(repr(e))

    start_time_save = time.time()
    with open(hf.new_path(os.path.join(save_dir, f'{name}.pkl')), 'wb') as f:
        pickle.dump({'uc': uc, 'fd': fd,
                    'p': mesh_points, 't': elements,
                    'p_all': all_mesh_points, 't_all': all_elements,
                    'boundary_inds': mesh_bounds,
                    'unique_holes': unique_holes,
                }, f)

    if verbose:
        print(f'Time for saving: {time.time() - start_time_save:.4} seconds')

    # %% [markdown]
    # turn unit cell into RVE: tile 2×2
    all_mesh_points2 = np.concatenate((all_mesh_points,
                                all_mesh_points + uc['lattice vectors'][0],
                                all_mesh_points + uc['lattice vectors'][1],
                                all_mesh_points + uc['lattice vectors'][0] + uc['lattice vectors'][1],
                                ), axis=0)
    n = len(all_mesh_points)
    all_elements2 = np.concatenate((all_elements,
                                    all_elements + n,
                                    all_elements + 2*n,
                                    all_elements + 3*n,
                                    ), axis=0)

    inds_per_fd2 = np.tile(inds_per_fd, reps=(4, 1, 1))
    inds_per_fd2 += np.arange(4).reshape(-1, 1, 1)*n
    if verbose:
        print('inds_per_fd2.shape', inds_per_fd2.shape)

    mesh_bounds2 = [mesh_bounds,
                    [(np.array(mb) + n).tolist() for mb in mesh_bounds],
                    [(np.array(mb) + 2*n).tolist() for mb in mesh_bounds],
                    [(np.array(mb) + 3*n).tolist() for mb in mesh_bounds],]

    # %% Deduplicate (again)
    start_time_dedup = time.time()
    all_mesh_points4, inds, inv, c = hf.uniquetol(all_mesh_points2, tol=1e-4, return_counts=True, return_index=True, return_inverse=True, axis=0)
    if verbose:
        print(f'Time for deduplication: {time.time() - start_time_dedup:.4} seconds')
    all_elements4 = inv[all_elements2]
    inds_per_fd4 = inv[inds_per_fd2]

    for i in range(len(mesh_bounds2)):
        for j in range(len(mesh_bounds2[i])):
            mesh_bounds2[i][j] = inv[mesh_bounds2[i][j]]

    # %%
    # Plot RVE, with unit cell boundaries
    if figures == 2:
        fig, ax = plt.subplots(figsize=(7,7))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        # Plot after deduplication
        temp = all_mesh_points4[all_elements4]
        temp = np.transpose(temp, axes=[0,2,1])
        temp = temp[..., [0,3,1,4,2,5]]
        temp = temp.reshape(-1, temp.shape[-1])
        plt.fill(*temp, alpha=0.5)  #, c='tab:orange')

        for mb in mesh_bounds2:
            for mb2 in mb:
                plt.scatter(*all_mesh_points4[mb2].T, s=5, zorder=11)

        plt.scatter(*all_mesh_points4[inds_per_fd4[0][0]].T, s=1, zorder=10, c='black')
        plt.scatter(*all_mesh_points4[inds_per_fd4[1][0]].T, s=1, zorder=10, c='black')

        # plt.scatter(*all_mesh_points4.T, s=1, zorder=10)
        plt.gca().set_aspect('equal')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_RVE_with_uc_boundaries.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    if figures == 2:
        # Plot counts
        fig, ax = plt.subplots(figsize=(10,10))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        temp = all_mesh_points4[all_elements4]
        temp = np.transpose(temp, axes=[0,2,1])
        temp = temp[..., [0,3,1,4,2,5]]
        temp = temp.reshape(-1, temp.shape[-1])
        plt.fill(*temp, c='whitesmoke')
        plt.title('counts per point, should be =/= 1 at and only at overlapping unit cell boundaries')

        for count in np.unique(c):
            plt.scatter(*all_mesh_points4[c == count].T, s=count*5, zorder=10, label=count)

        plt.gca().set_aspect('equal')
        plt.legend(title='count', loc=1)

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_counts.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    # %%
    # meshbounds: find new boundary nodes of RVE (old boundary points and copies thereof, minus the deduplicated ones)
    mesh_bounds4 = [[] for asdf in mesh_bounds2[0]]

    # iterate over unit cells
    for i in range(len(mesh_bounds2)):
        # iterate over direction of boundary
        # oblique: bottom, right, top, left
        # hexagonal: bottom, bottomright, topright, top, topleft, bottomleft
        for j in range(len(mesh_bounds2[i])):
            # check if it's an internal boundary by checking if all of the nodes had duplicates
            if (c[mesh_bounds2[i][j]] == 1).any():
                mesh_bounds4[j].extend(mesh_bounds2[i][j])

    # %%
    # Check if number of boundary nodes is consistent with the unit cell
    for i in range(len(mesh_bounds4)):
        mesh_bounds4[i] = np.unique(mesh_bounds4[i]).tolist()
        if hf.wallpaper_groups[group]['unit cell shape'] == 'hexagon':
            len_new = len(mesh_bounds4[i])
            len_old = len(mesh_bounds2[0][i])
            assert 2*len_old-2 <= len_new <= 3*len_old, 'boundaries are the wrong size!'
        elif hf.wallpaper_groups[group]['unit cell shape'] == 'parallelogram':
            len_new = len(mesh_bounds4[i])
            len_old = len(mesh_bounds2[0][i])
            assert 2*len_old-2 <= len_new <= 2*len_old, 'boundaries are the wrong size!'
        else:
            raise NotImplementedError(f"shape {hf.wallpaper_groups[group]['unit cell shape'] } not implemented")

    # %%
    # sort boundary nodes in the correct order
    for i, mb in enumerate(mesh_bounds4):
        coords = all_mesh_points4[mb]

        # sort by whichever has a larger difference: x- or y-coordinate
        x_diff = np.max(coords[:, 0]) - np.min(coords[:,0])
        y_diff = np.max(coords[:, 1]) - np.min(coords[:,1])

        if group == 'p3' or group == 'p3m1':
            temp2 = np.einsum('i,ji->j', uc['bounds'][i][1], coords)
            mesh_bounds4[i] = np.array(mesh_bounds4[i])[np.argsort(temp2)]
            if i >= 3:
                mesh_bounds4[i] = np.flip(mesh_bounds4[i], axis=0)
        else:
            if x_diff > y_diff:
                mesh_bounds4[i] = np.array(mesh_bounds4[i])[np.argsort(coords[:, 0])]
            # else sort by y-coordinate
            else:
                mesh_bounds4[i] = np.array(mesh_bounds4[i])[np.argsort(coords[:, 1])]

    # in the case of a hexagonal unit cell, the third boundary has a different order for the source vs the image nodes (reverse order of thirds)
    if group == 'p3':
        n = len(mesh_bounds4[3])//3
        mesh_bounds4[3] = np.concatenate((mesh_bounds4[3][2*n:],
                                        mesh_bounds4[3][n:2*n],
                                        mesh_bounds4[3][:n]), axis=0)

    if group == 'p3m1':
        n = len(mesh_bounds4[5])//3
        mesh_bounds4[5] = np.concatenate((mesh_bounds4[5][2*n:],
                                        mesh_bounds4[5][n:2*n],
                                        mesh_bounds4[5][:n]), axis=0)

    # %% Plot to check boundaries
    if figures == 2:
        fig, ax = plt.subplots(figsize=(10,10))
        # ax.set_title(f'{group} ({shape})')
        fig.patch.set_facecolor("None")

        temp = all_mesh_points4[all_elements4]
        temp = np.transpose(temp, axes=[0,2,1])
        temp = temp[..., [0,3,1,4,2,5]]
        temp = temp.reshape(-1, temp.shape[-1])
        plt.fill(*temp, c='whitesmoke')

        for mb in mesh_bounds4:
            plt.scatter(*all_mesh_points4[mb].T, s=5, zorder=11, alpha=0.5)
            plt.scatter(*all_mesh_points4[mb][:10].T, s=2, zorder=11)

        plt.scatter(*all_mesh_points4.T, s=1, zorder=10, c='black')
        plt.gca().set_aspect('equal')
        plt.title('Check boundaries')

        path1 = hf.new_path(os.path.join(save_dir, f'fig_{fig_nr}_check_boundaries.png'))
        fig_nr += 1
        plt.axis('off')
        fig.savefig(path1)
        plt.close()

    t = all_elements4
    edges = np.vstack((t[:, [0, 3]],
                t[:, [3, 1]],
                t[:, [1, 4]],
                t[:, [4, 2]],
                t[:, [2, 5]],
                t[:, [5, 0]]))

    edges2 = np.sort(edges, axis=-1)
    edges3, inv, counts = np.unique(edges2, axis=0, return_inverse=True, return_counts=True)

    if np.any(counts > 2):
        raise ValueError('Some edges are used more than twice')

    if len(hf.crossing_edges(all_mesh_points4, edges3)) > 0:
        raise ValueError('Some edges cross each other')

    # %% save to matlab stuff
    # needed:
    # * nodes: xyz coordinates of the nodes, shape [nr of nodes, 3] (z-coordinate can be zero)
    # * nNodes: nr of nodes (integer)
    # elements: indices of nodes defining elements, shape [nr of elements, 6], elements 0, 1, 2 are the corner nodes in counterclockwise order, 3,4,5 are the mid-edge nodes also in counterclockwise order
    # * elemMats: material of each element (all the same, can all be 1), shape [nr of elements, 1]
    # * nelems: nr of elements
    # FE2: 1×1 struct, contains:
    #   FE2.V: volume of RVE
    #   FE2.periodicSourceNodes: list of independent boundary + corner nodes (?)
    #   FE2.periodicImageNodes: list of corresponding dependent boundary + corner nodes
    # which nodes correspond to which fundamental domain

    transforms = copy.deepcopy(uc['transforms'])
    # to do: transforms to numerical form
    for i, temp in enumerate(uc['transforms']):
        for j, temp2 in enumerate(temp):
            for k, temp3 in enumerate(temp2):
                try:
                    transforms[i][j][k] = eval(temp3)
                except NameError:
                    transforms[i][j][k] = temp3
    if verbose:
        print(transforms)

    scipy.io.savemat(hf.new_path(os.path.join(save_dir, f'{name}.mat')),
                    {'p': all_mesh_points4,
                    't': all_elements4,
                    'boundary_inds': mesh_bounds4,
                    'inds_per_fd': inds_per_fd4,
                    'lattice_vectors': uc['lattice vectors'],
                    'transforms': transforms,
                    'a1': fd['a1'],
                    'a2': fd['a2'],
                    'volume_fraction': vol_frac2,
                    })

    # %%
    print('Volume fraction based on spline points polygons:', f'{vol_frac*100:.1f}%')
    print('Volume fraction based on mesh:', f'{vol_frac2*100:.1f}%')
    print(f'Total time to generate new material: {time.time()-start_time:.4} seconds')

    with open(hf.new_path(os.path.join(save_dir, 'info.txt')), 'w') as f:
        f.write(f'Volume fraction: {vol_frac2*100:.1f}%\n')
        f.write(f'Total time: {time.time()-start_time:.4} seconds\n')

# %%
if __name__ == '__main__':

    argumentList = sys.argv[1:]

    # Options
    options = "hg:s:v:f:"

    # Long options
    long_options = ["help", "group=", "shape=", "verbose", "figures"]

    group_given = False
    shape_given = False
    verbose = False
    figures = 1
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        # print('arguments:', arguments)
        # print('values:', values)

        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--help"):
                print('''This is a script to generate a new material geometry based on a symmetry group and Bravais lattice. The generated data and figures will be saved in a new folder inside a "data" directory, which will be created if it does not exist already.
Options:
-g, --group: name of the wallpaper group
-s, --shape: Bravais lattice type (determines shape of the unit cell), ['square', 'hexagonal', 'oblique', 'rectangular', 'rhombic'], (not all options are available for all groups, and cm has 'hexagonal1' and 'hexagonal2'.)
-v, --verbose: how much output to print (0: only the most important info, 1: lots of info)
-f, --figures: how many figures to save (0: None, 1: only the most important, 2: all)
-h, --help: print help
    ''')
                sys.exit()

            elif currentArgument in ("-g", "--group"):
                group = currentValue
                group_given = True

            elif currentArgument in ("-s", "--shape"):
                shape = currentValue
                shape_given = True

            elif currentArgument in ("-v", "--verbose"):
                if currentValue == '0':
                    verbose = False
                elif currentValue == '1':
                    verbose = True
                else:
                    raise ValueError(f'{currentValue} is not an acceptable value for verbose')

            elif currentArgument in ("-f", "--figures"):
                figures = int(currentValue)
                if figures not in [0,1,2]:
                    raise ValueError(f'{figures} is not an acceptable value for figures')

    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))

    # to use the same rng state as a previously generated material (e.g., to check the influence of a change in the code or a different value of one of the constants)
    # rng_state_path = r"data\pmg_square_2024-05-22_14-40-41.449687\rngstate_00.pkl"
    rng_state_path = None

    # set parameters for graph generation
    if not group_given:
        group = 'pmg'
    if not shape_given:
        shape = 'square'

    print(f'symmetry group: {group}, shape: {shape}')

    if not os.path.isdir('data'):
        os.mkdir('data')

    # Create new folder for figures
    date_time_string = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
    name = f'{group}_{shape}_{date_time_string}'

    save_dir = hf.new_path(os.path.join('data', name), always_number=False)
    name = os.path.split(save_dir)[-1]
    os.mkdir(save_dir)

    if rng_state_path is not None:
        with(open(hf.new_path(os.path.join(save_dir, 'based_on.txt')), 'w')) as f:
            f.write(rng_state_path)

    print('Name:', name)
    print('Path:', save_dir)

    # generate new material
    generate_material_geometry(group, shape, verbose=verbose, figures=figures, save_dir=save_dir, rng_state_path=rng_state_path)
