# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import skgeom as sg
import sklearn.neighbors as skln
import scipy.sparse as sps
import matplotlib.bezier as mbez

# %%
def new_path(path, n=2, always_number=True):
    '''Create a new path for a file
    with a filename that does not exist yet,
    by appending a number to path (before the extension) with at least n digits.
    (Higher n adds extra leading zeros). If always_number=True,
    the filename will always have a number appended, even if the
    bare filename does not exist either.'''

    # initial path is path + name + n zeros + extension if always_number=True
    # and just path + filename if always_number=False
    name, ext = os.path.splitext(path)
    if always_number:
        savepath = f'{name}_{str(0).zfill(n)}{ext}'
    else:
        savepath = path

    # If filename already exists, add number to it
    i = 1
    while os.path.exists(savepath):
        savepath = f'{name}_{str(i).zfill(n)}{ext}'
        i += 1
    return savepath

def replace(arr, orig, repl, inplace=False):
    """In arr, replace the elements in orig by the elements in repl.

    Parameters
    ----------
    arr : array-like
        arr in which some values should be replaced
    orig : 1D array-like
        values in arr that should be replaced by the corresponding values in repl
    repl : 1D array-like of same length as orig
        values to replace the values of orig with
    inplace : bool, optional
        if False, return a changed copy of arr, by default False

    Returns
    -------
    array
        arr with the values in orig replaced by the values in repl
    """

    if inplace:
        arr = np.asarray(arr)
    else:
        arr = np.copy(arr)

    orig = np.asarray(orig)
    repl = np.asarray(repl)

    if not len(orig) == len(repl):
        raise ValueError(f'orig and repl should have the same length, current lengths are {len(orig)} and {len(repl)}')

    # indices that sort orig (so we can use np.searchsorted)
    inds = np.argsort(orig)
    orig_sorted = orig[inds]
    repl_sorted = repl[inds]

    # find arr values in orig
    inds2 = np.searchsorted(orig_sorted, arr) % len(orig_sorted)

    # whether the values in arr are actually in orig
    bools = orig_sorted[inds2] == arr

    # replace the values of orig with those of repl
    arr[bools] = repl_sorted[inds2[bools]]

    if not inplace:
        return arr

# %% Define transforms
def mirror_points(p, mirrorPoint, mirrorDir):
    mirrorPoint = np.transpose(mirrorPoint)
    mirrorDir = np.transpose(mirrorDir)

    # make the mirror point the origin
    p = np.copy(p) - mirrorPoint

    # length mirrorDir squared
    denom = np.dot(mirrorDir, mirrorDir)

    # calculate new p
    dot_prod = np.dot(mirrorDir, p.T).reshape(-1, 1)  # one scalar per point in p
    p = 2 * dot_prod / denom * mirrorDir - p + mirrorPoint  # turn origin back to original

    return p # Return the updated vertices p and the triangles t


def rotate_points(p, rotationPoint, angle):
    # rotate points p around rotationPoint by angle
    rotationPoint = np.asarray(rotationPoint).reshape(1, 2)

    # rotation matrix
    Rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

    # make rotationPoint the origin
    p = p - rotationPoint

    # apply rotation
    pt = np.dot(Rot, p.T).T

    # put origin back
    pt += rotationPoint

    return pt

def translate_points(p, translation_vector):
    return p + np.array(translation_vector).reshape(1,2)

# %% [markdown]
# ## Description of each group
#
# For each wallpaper group, define:
# * triangle or paralellogram as fundamental domain
# * how to construct unit cell from fundamental domain
# * how to surround fundamental domain with its neighbors (for periodicity)
# * degrees of freedom in a, b and gamma
# * which boundaries are mirror lines and which boundaries are periodic with themselves
#
# **Notation (that I made up)**
#
# Transformations:
# * Translation: T:[translation vector]
# * Mirroring: M:[coords of point on line]:[vector pointing along line]
# * Rotation: R:[center of rotation]:[rotation in degrees]
#
# Unit cell:
# *  all transformations of the fundamental domain needed to get the unit cell
#
# Rosettes:
# * describes points where only rosettes can be placed, not any arbitrary hole
# * not including (dihedral, 1,) for mirror boundaries
# * description: (location, orientation, type of rosette, [neighbors this rosette should be copied to],)
#     - orientation: vector pointing along an axis of symmetry, only applicable for dihedral
#     - type of rosette: (cyclic or dihedral, order of rotation,)
#
# Mirror boundaries:
# * describes lines along which only (dihedral, 1,)-rosettes can be placed
# * description: (transformation, [neighbors that rosettes on this line should be copied to],)
#
# Boundaries:
# * Tuples of coupled boundaries: (boundary 1, boundary it is coupled to, -1 if direction (clockwise vs counterclockwise) flips, +1 else)
# Coupled boundary and direction are None for mirror boundaries
# * E.g. self-similar boundary: (0, 0, -1)
# * E.g. left-right periodicity: (1, 3, -1)
# * E.g. mirror-boundary: (0, None, None)

# %%
wallpaper_groups = {
    'p1': {
        'fundamental domain shape': 'parallelogram',  # either parallelogram or triangle
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', 'a2'],
        'fundamental domain parameters': {
            'oblique':     {'a': 1, 'b': [0.5, 1.0], 'gamma': [np.pi/3, np.pi/2]},
            'hexagonal':   {'a': 1, 'b':  1,  'gamma': np.pi/3},
            'rectangular': {'a': 1, 'b': [0.5, 1.0], 'gamma': np.pi/2},
            'rhombic':     {'a': 1, 'b':  1, 'gamma': [np.pi/3, np.pi/2]},
            'square':      {'a': 1, 'b':  1, 'gamma':  np.pi/2},
        },
        'unit cell': [('T:[0,0]',)],
        'neighbors': [('T:-a1','T:-a2',),
                        ('T:-a1','T:[0,0]',),
                        ('T:-a1','T:a2',),
                        ('T:[0,0]','T:-a2',),
                        ('T:[0,0]','T:[0,0]',),
                        ('T:[0,0]','T:a2',),
                        ('T:a1','T:-a2',),
                        ('T:a1','T:[0,0]',),
                        ('T:a1','T:a2',),
                        ],
        'mirror boundaries': [],
        'rosettes': [],
        'boundaries': [(0, 2, -1), (1, 3, -1)],
        'unit cell boundaries': [('[0,0]', 'a1'), ('a1', 'a2'), ('a1+a2', '-a1'), ('a2', '-a2')],
        },
        # =========================== TO DO: for p2, fundamental domain can also be a triangle. More convenient?
    'p2': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', 'a2'],
        'fundamental domain parameters': {
            'oblique':     {'a': 1, 'b': [0.5, 2.0], 'gamma': [np.pi/3, np.pi/2]},
            'hexagonal':   {'a': 1, 'b':  1,  'gamma': np.pi/3},
            'rectangular': {'a': 1, 'b': [0.5, 2.0], 'gamma': np.pi/2},
            'rhombic':     {'a': 1, 'b':  1, 'gamma': [np.pi/3, np.pi/2]},
            'square':      {'a': 1, 'b':  1, 'gamma':  np.pi/2},
        },
        'unit cell': [
            ('T:[0,0]',),
            ('R:a1/2+a2/2:180',),
            ],
        'neighbors': [
                        ('R:a1/2+a2/2:180', 'T:-a1', 'T:-a2',),
                        ('R:a1/2+a2/2:180', 'T:-a1',),
                        ('R:a1/2+a2/2:180', 'T:-a1', 'T:a2',),
                        ('T:-a2',),
                        ('T:[0,0]',),
                        ('T:a2',),
                        ('R:a1/2+a2/2:180', 'T:a1', 'T:-a2',),
                        ('R:a1/2+a2/2:180', 'T:a1', 'T:[0,0]',),
                        ('R:a1/2+a2/2:180', 'T:a1', 'T:a2',),
                        ],
        'mirror boundaries': [],
        'rosettes': [   ('a1/2', None, ('cyclic',2,), [3,4,5,6,7,8],),
                        ('a2/2', None, ('cyclic',2,), [0,1,2,3,4,5],),
                        ('[0,0]',  None, ('cyclic',2,), [3,4,5,6,7,8],),
                        ('a1',   None, ('cyclic',2,), [0,1,2,3,4,5],),
                    ],
        'boundaries': [(0, 0, -1), (1, 1, -1), (2, 2, -1)],
        'unit cell boundaries': [('[0,0]', 'a1'),
                                 ('a1', 'a2'),
                                 ('a1+a2', '-a1'),
                                 ('a2', '-a2')
                                 ],
        },
    'pm': {
        'fundamental domain shape': 'parallelogram',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', '2*a2'],
        'fundamental domain parameters': {
            'rectangular':     {'a': 1, 'b': [0.25, 1], 'gamma': np.pi/2},
            'square':     {'a': 1, 'b': 0.5, 'gamma': np.pi/2},
        },
        'unit cell': [('T:[0,0]',),
                        ('M:a2:a1',),],
        'neighbors': [
                        # ('M:a2:a1', 'T:-a1',),
                        # ('M:a2:a1',),
                        # ('M:a2:a1', 'T:a1',),
                        ('T:-a2',),
                        ('T:[0,0]',),
                        ('T:a2',),
                        # ('M:[0,0]:a1', 'T:-a1',),
                        # ('M:[0,0]:a1',),
                        # ('M:[0,0]:a1', 'T:a1',),
                        ],
        'mirror boundaries': [
                              'M:[0,0]:a1',
                              'M:a2:a1',
                            ],
        'rosettes': [],
        'boundaries': [(0, None, None), (1, 3, -1), (2, None, None)],
        'unit cell boundaries': [('[0,0]', 'a1'), ('a1', '2*a2'), ('a1+2*a2', '-a1'), ('2*a2', '-2*a2')],
        },
    'pg': {
        'fundamental domain shape': 'parallelogram',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', '2*a2'],
        'fundamental domain parameters': {
            'rectangular':     {'a': 1, 'b': [0.25, 1], 'gamma': np.pi/2},
            'square':     {'a': 1, 'b': 0.5, 'gamma': np.pi/2},
        },
        'unit cell': [('T:[0,0]',),
                        ('M:[0,0]:a2', 'T:a1', 'T:a2',),],
        'neighbors': [
                        ('M:[0,0]:a2', 'T:-a2',),
                        ('M:[0,0]:a2', 'T:-a2', 'T:a1',),
                        ('M:[0,0]:a2', 'T:-a2', 'T:2*a1',),
                        ('T:-a1',),
                        ('T:[0,0]',),
                        ('T:a1',),
                        ('M:[0,0]:a2', 'T:a2',),
                        ('M:[0,0]:a2', 'T:a2', 'T:a1',),
                        ('M:[0,0]:a2', 'T:a2', 'T:2*a1',),
                        ],
        'mirror boundaries': [],
        'rosettes': [],
        'boundaries': [(0, 2, 1), (1, 3, -1)],
        'unit cell boundaries': [('[0,0]', 'a1'), ('a1', '2*a2'), ('a1+2*a2', '-a1'), ('2*a2', '-2*a2')],
        },
    'cm': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', 'a2'],
        'fundamental domain parameters': {
            'rhombic':     {'a': 1, 'b': 1, 'gamma': [np.pi/4, 3*np.pi/4]},
            'square':     {'a': 1, 'b': 1, 'gamma': np.pi/2},
            'hexagonal1':     {'a': 1, 'b': 1, 'gamma': np.pi/3},
            'hexagonal2':     {'a': 1, 'b': 1, 'gamma': 2*np.pi/3},
        },
        'unit cell': [('T:[0,0]',),
                        ('M:a1:a2-a1',),],
        'neighbors': [
                        ('T:-a1', 'T:a2',),
                        ('M:a1:a2-a1', 'T:-a1',),
                        ('T:[0,0]',),
                        ('M:a1:a2-a1', 'T:-a2',),
                        ('T:a1', 'T:-a2',),
                        ],
        'mirror boundaries': ['M:a1:a2-a1'],
        'rosettes': [],
        'boundaries': [(0, 2, 1), (1, None, None)],
        'unit cell boundaries': [('[0,0]', 'a1'), ('a1', 'a2'), ('a1+a2', '-a1'), ('a2', '-a2')],
        },
    'pmm': {
        'fundamental domain shape': 'parallelogram',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', '2*a2'],
        'fundamental domain parameters': {
            'rectangular':  {'a': 1, 'b': [0.5, 2.0], 'gamma': np.pi/2},
            'square':       {'a': 1, 'b': 1, 'gamma': np.pi/2},
        },
        'unit cell': [('T:[0,0]',),
                        ('M:a2:a1',),
                        ('M:a1:a2',),
                        ('M:a2:a1', 'M:a1:a2',),],
        'neighbors': [('T:[0,0]',),],
        'mirror boundaries': [
                                'M:[0,0]:a1',
                                'M:[0,0]:a2',
                                'M:a2:a1',
                                'M:a1:a2',
                              ],
        'rosettes': [('[0,0]',   'a1', ('dihedral',2,), [0,],),
                     ('a1',    'a1', ('dihedral',2,), [0,],),
                     ('a2',    'a1', ('dihedral',2,), [0,],),
                     ('a1+a2', 'a1', ('dihedral',2,), [0,],),
                    ],
        'boundaries': [(0, None, None), (1, None, None), (2, None, None), (3, None, None)],
        'unit cell boundaries': [('[0,0]', '2*a1'), ('2*a1', '2*a2'), ('2*a1+2*a2', '-2*a1'), ('2*a2', '-2*a2')],
        },
    'pmg': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', 'a2'],
        'fundamental domain parameters': {
            'rectangular':  {'a': 1, 'b': [1.0, 4.0], 'gamma': np.pi/2},
            'square':       {'a': 1, 'b': 2, 'gamma': np.pi/2},
        },
        'unit cell': [  ('T:[0,0]',),
                        ('R:a1/2+a2/2:180',),
                        ('M:a1:a2',),
                        ('R:a1/2+a2/2:180', 'M:a1:a2',),
                        ],
        'neighbors': [
                        None  # not implemented because when I changed to triangular domain I did not need the neighbors anymore
                        # add later if I do need them
                      ],
        'mirror boundaries': [
                                'M:[0,0]:a2',
                              ],
        'rosettes': [
                        None  # not implemented because depends on neighbors
                    ],
        'boundaries': [(0, 0, -1), (1, 1, -1), (2, None, None)],
        'unit cell boundaries': [('[0,0]', '2*a1'), ('2*a1', 'a2'), ('2*a1+a2', '-2*a1'), ('a2', '-a2')],
        },
    'pgg': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', 'a2-a1'],
        'fundamental domain parameters': {
            'rhombic':  {'a': 1, 'b': 1, 'gamma': [np.pi/6, np.pi/2]},
            'square':  {'a': 1, 'b': 1, 'gamma': np.pi/2},
        },
        'unit cell': [  ('T:[0,0]',),
                        ('R:a1/2+a2/2:180',),
                        ('M:a1:a2-a1', 'T:-a1',),
                        ('M:[0,0]:a2+a1', 'T:a1',),
                        ],
        'neighbors': [  ('T:a2+a1',),
                        ('R:a1/2+a2/2:180', 'T:a2+a1',),
                        ('M:[0,0]:a2+a1', 'T:a2',),
                        ('M:[0,0]:a2+a1', 'T:-a1',),
                        ('M:a1:a2-a1', 'T:-a1',),
                        ('T:[0,0]',),
                        ('R:a1/2+a2/2:180',),
                        ('M:[0,0]:a2+a1', 'T:a1',),
                        ('R:[0,0]:180',),
                        ('M:[0,0]:a2+a1', 'T:-a2',),
                        ('M:a1:a2-a1', 'T:-a2',),
                        ('T:a1', 'T:-a2',),
                        ('R:[0,0]:180', 'T:2*a1',),
                      ],
        'mirror boundaries': [],
        'rosettes': [('[0,0]',  None, ('cyclic',2,), [0,1,2,3,5,6,7,9,11,12],),
                     ('a1/2+a2/2',  None, ('cyclic',2,), [0,2,3,5,7,8,9,11],),
                    ],
        'boundaries': [(0, 2, 1), (1, 1, -1)],
        'unit cell boundaries': [('[0,0]', '2*a1'), ('2*a1', 'a2-a1'), ('a1+a2', '-2*a1'), ('a2-a1', '-a2+a1')],
        },
    'cmm': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', 'a1+a2'],
        'fundamental domain parameters': {
            'rhombic':  {'a': 1, 'b': [0.25, 1], 'gamma': np.pi/2},
            'square':  {'a': 1, 'b': 1, 'gamma': np.pi/2},
            'hexagonal':  {'a': 1, 'b': 1/np.sqrt(3), 'gamma': np.pi/2},
        },
        'unit cell': [  ('T:[0,0]',),
                        ('M:[0,0]:a2',),
                        ('R:a1/2+a2/2:180',),
                        ('M:[0,0]:a1', 'T:a1', 'T:a2',),
                        ],
        'neighbors': [  ('T:[0,0]',),
                        ('R:a1/2+a2/2:180',),
                      ],
        'mirror boundaries': ['M[0,0]:a1', 'M[0,0]:a2'],
        'rosettes': [('[0,0]',  'a1', ('dihedral',2,), [0,1],),
                     ('a1',  'a1', ('dihedral',2,), [0,1],),
                     ('a1/2+a2/2',  None, ('cyclic',2,), [0],),],
        'boundaries': [(0, None, None), (1, 1, -1), (2, None, None)],
        'unit cell boundaries': [('-a1', '2*a1'), ('a1', 'a2+a1'), ('2*a1+a2', '-2*a1'), ('a2', '-a2-a1')],
        },
    'p4': {
        'fundamental domain shape': 'parallelogram',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', '2*a2'],
        'fundamental domain parameters': {
            'square':  {'a': 1, 'b': 1, 'gamma': np.pi/2},
        },
        'unit cell': [  ('T:[0,0]',),
                        ('R:a1+a2:90',),
                        ('R:a1+a2:180',),
                        ('R:a1+a2:270',),
                        ],
        'neighbors': [
                        ('R:a2:180',),
                        ('R:a1+a2:270',),
                        ('R:a1+a2:180',),
                        ('R:[0,0]:90',),
                        ('T:[0,0]',),
                        ('R:a1+a2:90',),
                        ('R:[0,0]:180',),
                        ('R:[0,0]:270',),
                        ('R:a2:180',),
                      ],
        'mirror boundaries': [],
        'rosettes': [('[0,0]',  None, ('cyclic',4,), [0,2,4,5],),
                     ('a1',  None, ('cyclic',2,), [0,1,2,4,5,6,7,],),
                     ('a1/2+a2/2',  None, ('cyclic',4,), [0,4,6,7],),],
        'boundaries': [(0, 3, -1), (1, 2, -1)],
        'unit cell boundaries': [('[0,0]', '2*a1'), ('2*a1', '2*a2'), ('2*a1+2*a2', '-2*a1'), ('2*a2', '-2*a2')],
        },
    'p4m': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', '2*a2'],
        'fundamental domain parameters': {
            'square':  {'a': 1, 'b': 1, 'gamma': np.pi/2},
        },
        'unit cell': [
                        ('M:a2:a1',),
                        ('R:a2:90',),
                        ('M:a1:a2-a1', 'R:a1+a2:180',),
                        ('R:a1+a2:180',),
                        ('T:[0,0]',),
                        ('M:a1:a2-a1',),
                        ('R:a1:270',),
                        ('M:a1:a2',),
                        ],
        'neighbors': [
                        ('T:[0,0]',),
                    ],
        'mirror boundaries': ['M:a1:a2-a1', 'M:a2:a1', 'M:a1:a2'],
        'rosettes': [('[0,0]',  'a1', ('dihedral',2,), [0,],),
                     ('a1',  'a1', ('dihedral',4,), [0,],),
                     ('a2',  'a1', ('dihedral',4,), [0,],),],
        'boundaries': [(0, None, None), (1, None, None), (2, None, None)],
        'unit cell boundaries': [('[0,0]', '2*a1'), ('2*a1', '2*a2'), ('2*a1+2*a2', '-2*a1'), ('2*a2', '-2*a2')],
        },
    'p4g': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['2*a1', '2*a2'],
        'fundamental domain parameters': {
            'square':  {'a': 1, 'b': 1, 'gamma': np.pi/2},
        },
        'unit cell': [
                        ('R:a1+a2:270',),
                        ('M:a1:a2-a1', 'R:a1+a2:270',),                        ('M:a1:a2-a1', 'R:a1+a2:180',),
                        ('R:a1+a2:180',),
                        ('T:[0,0]',),
                        ('M:a1:a2-a1',),
                        ('M:a1:a2-a1', 'R:a1+a2:90',),
                        ('R:a1+a2:90',),
                        ],
        'neighbors': [
                        ('T:[0,0]',),
                        ('R:[0,0]:90',),
                        ('R:[0,0]:180',),
                        ('R:[0,0]:270',),
                    ],
        'mirror boundaries': ['M:a1:a2-a1'],
        'rosettes': [('[0,0]',  None, ('cyclic',4,), [0,],),
                     ('a1',  'a2-a1', ('dihedral',2,), [0,1,2,3],),
                    ],
        'boundaries': [(0, 2, -1), (1, None, None)],
        'unit cell boundaries': [('[0,0]', '2*a1'), ('2*a1', '2*a2'), ('2*a1+2*a2', '-2*a1'), ('2*a2', '-2*a2')],
        },
    'p3': {
        'fundamental domain shape': 'parallelogram',
        'unit cell shape': 'hexagon',
        'lattice vectors': [ '2*a1-a2', '2*a2-a1'],
        'fundamental domain parameters': {
            'hexagonal':  {'a': 1, 'b': 1, 'gamma': np.pi/3},
        },
        'unit cell': [  ('T:[0,0]',),
                        ('R:a1:120',),
                        ('R:a1:240',),
                        ],
        'neighbors': [
                        ('R:[0,0]:240',),
                        ('R:a2:240',),
                        ('R:a2:120',),
                        ('R:a1+a2:240',),
                        ('T:-a1-a2',),
                        ('T:[0,0]',),
                        ('T:a1+a2',),
                        ('R:[0,0]:120',),
                        ('R:a1:120',),
                        ('R:a1:240',),
                        ('R:a1+a2:120',),
                      ],
        'mirror boundaries': [],
        'rosettes': [
                     ('[0,0]',  None, ('cyclic',3,), [1,2,3,4,5,8,10,],),
                     ('a1',  None, ('cyclic',3,), [0,2,3,4,5,6,],),
                     ('a2',  None, ('cyclic',3,), [0,3,5,7,9],),
                    ],
        'boundaries': [(0, 1, -1), (2, 3, -1)],
        'unit cell boundaries': [('[0,0]', '-a2+a1'),
                                 ('-a2+a1', 'a1'),
                                 ('-a2+2*a1', 'a2'),
                                 ('2*a1', 'a2-a1'),
                                 ('a1+a2', '-a1'),
                                 ('a2', '-a2')],
        },
    'p3m1': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'hexagon',
        'lattice vectors': ['2*a1-a2', '2*a2-a1'],
        'fundamental domain parameters': {
            'hexagonal':  {'a': 1, 'b': 1, 'gamma': np.pi/3},
        },
        'unit cell': [
                        ('R:a2:240',),
                        ('M:a2:a1',),
                        ('R:a2:120',),
                        ('M:[0,0]:a2',),
                        ('T:[0,0]',),
                        ('M:a1:a2-a1',),
                        ],
        'neighbors': [
                        ('T:[0,0]',),
                      ],
        'mirror boundaries': [
                        'M:[0,0]:a1',
                        'M:[0,0]:a2',
                        'M:a1:a2-a1',
                        ],
        'rosettes': [
                     ('[0,0]',  'a1', ('dihedral',3,), [0,],),
                     ('a1',  'a1', ('dihedral',3,), [0,],),
                     ('a2',  'a1', ('dihedral',3,), [0,],),
                    ],
        'boundaries': [(0, None, None), (1, None, None), (2, None, None)],
        'unit cell boundaries': [('[0,0]', 'a1'),
                                 ('a1', 'a2'),
                                 ('a2+a1', 'a2-a1'),
                                 ('2*a2', '-a1'),
                                 ('2*a2-a1', '-a2'),
                                 ('a2-a1', '-a2+a1')],
        },
    'p31m': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', '3*a2-a1'],
        'fundamental domain parameters': {
            'hexagonal':  {'a': 1, 'b': 1/np.sqrt(3), 'gamma': np.pi/6},
        },
        'unit cell': [
                        ('M:[0,0]:a1', 'T:3*a2-a1',),
                        ('R:a2:240',),
                        ('R:a2:120',),
                        ('M:[0,0]:a1', 'R:a2:120',),
                        ('M:[0,0]:a1', 'R:a1:240',),
                        ('T:[0,0]',),
                        ],
        'neighbors': [
                        ('T:[0,0]',),
                        ('R:a2:240',),
                        ('R:a2:120',),
                      ],
        'mirror boundaries': [
                        'M:[0,0]:a1',
                        # 'M:[0,0]:3*a2-a1',
                        # 'M:a1:3*a2-2a1',
                        ],
        'rosettes': [
                     ('[0,0]',  'a1', ('dihedral',3,), [0,1,2],),
                     ('a2',  None, ('cyclic',3,), [0,],),
                    ],
        'boundaries': [(0, None, None), (1, 2, -1)],
        'unit cell boundaries': [('[0,0]', 'a1'),
                                 ('a1', '3*a2-a1'),
                                 ('3*a2', '-a1'),
                                 ('3*a2-a1', '-3*a2+a1')],
        },
    'p6': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1', '3*a2-a1'],
        'fundamental domain parameters': {
            'hexagonal':  {'a': 1, 'b': 1/np.sqrt(3), 'gamma': np.pi/6},
        },
        'unit cell': [
                        ('R:[0,0]:180', 'T:3*a2',),
                        ('R:a2:240',),
                        ('R:a2:120',),
                        ('R:a1:300',),
                        ('R:3*a2-a1:60',),
                        ('T:[0,0]',),
                        ],
        'neighbors': [
                        ('R:[0,0]:120',),
                        ('R:3*a2-a1:300',),
                        ('R:[0,0]:60',),
                        ('R:a2:240',),
                        ('R:a2:120',),
                        ('R:a1:300',),
                        ('R:3*a2-a1:60',),
                        ('R:a1:240',),
                        ('T:-a1',),
                        ('T:[0,0]',),
                        ('T:a1',),
                        ('R:a1/2:180', 'T:a1',),
                        ('R:a1/2:180', 'T:[0,0]',),
                        ('R:a1/2:180', 'T:-a1',),
                        ('R:a1/2:180', 'R:a1:240',),
                        ('R:a1/2:180', 'R:3*a2-a1:60',),
                        ('R:a1/2:180', 'R:a1:300',),
                        ('R:a1/2:180', 'R:a2:120',),
                        ('R:a1/2:180', 'R:a2:240',),
                        ('R:a1/2:180', 'R:[0,0]:60',),
                        ('R:a1/2:180', 'R:3*a2-a1:300',),
                        ('R:a1/2:180', 'R:[0,0]:120',),
                      ],
        'mirror boundaries': [
                        ],
        'rosettes': [
                     ('[0,0]',  None, ('cyclic',6,), [1,3,4,7,8,9,13,14,16,20],),
                     ('a2',  None, ('cyclic',3,), [0,1,5,7,9,11,12,13,15,19],),
                    ],
        'boundaries': [(0, 0, -1), (1, 2, -1)],
        'unit cell boundaries': [('[0,0]', 'a1'),
                                 ('a1', '3*a2-a1'),
                                 ('3*a2', '-a1'),
                                 ('3*a2-a1', '-3*a2+a1')],
        },
    'p6m': {
        'fundamental domain shape': 'triangle',
        'unit cell shape': 'parallelogram',
        'lattice vectors': ['a1+2*a2', '4*a2-a1'],
        'fundamental domain parameters': {
            'hexagonal':  {'a': 1, 'b': 1/2, 'gamma': np.pi/3},
        },
        'unit cell': [
                        ('M:[0,0]:a2', 'R:2*a2-a1:60',),
                        ('T:2*a2', 'R:2*a2:60',),
                        ('M:a1:a2-a1', 'R:2*a2:120',),

                        ('R:[0,0]:120',),
                        ('M:[0,0]:a2',),
                        ('R:a2:180',),
                        ('T:[0,0]',),
                        ('M:a1:a2-a1',),
                        ('R:a1:300',),

                        ('M:[0,0]:a2', 'R:[0,0]:120',),
                        ('R:[0,0]:240',),
                        ('M:[0,0]:a1',),
                        ],
        'neighbors': [
                        ('T:[0,0]',),
                      ],
        'mirror boundaries': [
                            'M:[0,0]:a1', 'M:[0,0]:a2', 'M:a1:a2-a1'
                        ],
        'rosettes': [
                     ('[0,0]',  'a1', ('dihedral',3,), [0,],),
                     ('a1',  'a1', ('dihedral',6,), [0,],),
                     ('a2',  'a2', ('dihedral',2,), [0,],),
                    ],
        'boundaries': [(0, None, None), (1, None, None), (2, None, None)],
        'unit cell boundaries': [('-2*a2', 'a1+2*a2'),
                                 ('a1', '4*a2-a1'),
                                 ('4*a2', '-a1-2*a2'),
                                 ('2*a2-a1', '-4*a2+a1')],
        },
    }

# %% [markdown]
# ## Helper funcs

# %%
def orientation(triangles):
    """For each triangle, gives its orientation:
    0: points are collinear
    1: points are clockwise
    2: points are counterclockwise

    Parameters
    ----------
    triangles : ndarray of dimension >=2, with shape (..., 3, 2)
        2D Cartesian coordinates of points making up triangles.
        The second to last dimension indexes the three points of the triangle,
        last dimension indexes the x and y coordinate,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of integers (of same shape as input with the last two dimensions removed)
        orientation of each triangle, using broadcasting.
    """
    # gives order of points of a triangle:
    # 0 : collinear
    # 1 : clockwise
    # 2 : counterclockwise

    triangles = np.asarray(triangles)

    p = triangles[..., 0, :]  # 1st point of each triangle
    q = triangles[..., 1, :]  # 2nd point of each triangle
    r = triangles[..., 2, :]  # 3rd point of each triangle
    val = (q[..., 1]-p[..., 1]) * (r[..., 0]-q[..., 0]) - (q[..., 0]-p[..., 0]) * (r[..., 1]-q[..., 1])

    val[val>0] = 1
    val[val<0] = 2

    return val.astype(int)

def between(vals):
    """"Check if value C is between values A and B.

    Parameters
    ----------
    vals : ndarray of dimension >=1, with shape (..., 3)
        The second to last dimension indexes the values A, B, C,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last dimension removed)
        Whether value C is between values A and B (not equal to either), regardless of whether A is larger than B or vice versa.
    """
    bools1 = vals[..., 2] < np.maximum(vals[..., 0], vals[..., 1])
    bools2 = vals[..., 2] > np.minimum(vals[..., 0], vals[..., 1])
    return bools1*bools2

def onSegment(points):
    """Check if a point C is on a line segment from point A to B, assuming A, B and C are collinear.

    Parameters
    ----------
    vals : ndarray of dimension >=2, with shape (..., 3, 2)
        2D Cartesian coordinates of points A, B and C.
        The second to last dimension indexes the points A, B, C,
        last dimension indexes the x and y coordinate,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last two dimensions removed)
        Whether C is on line segment AB, but not the same as either A or B.
    """
    points = np.asarray(points)

    boolsx = between(points[..., 0])
    boolsy = between(points[..., 1])

    return boolsx + boolsy

def intersect(points):
    """Check whether a line segment from point A to point B
    intersects a line segment from point C to point D (in 2D Cartesian coordinates). (Only the end points coinciding does not count.)

    Parameters
    ----------
    points : ndarray of dimension >=2, with shape (..., 4, 2)
        2D Cartesian coordinates of points defining pairs of line segments.
        The second to last dimension indexes the four points A, B, C, D,
        last dimension indexes the x and y coordinate.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last two dimensions removed)
        Whether the line segments intersect or not.
    """
    points = np.asarray(points)  # shape [..., 4, 2]

    # different combinations of points to take as triangles
    combis = [[0,1,2], [0,1,3],[2,3,0],[2,3,1]]

    # orientation of each triangle
    ori = orientation(points[..., combis, :])

    # line segments cross and nothing is collinear
    bools = (ori[..., 0] != ori[..., 1])*(ori[..., 2] != ori[..., 3]) * (ori != 0).all(axis=-1)

    # collinear combinations
    bools_coll = ori == 0

    # any one of the combinations is collinear
    bools_coll2 = bools_coll.any(axis=-1)

    # all cases where 3 or more points are collinear
    coll_points = points[bools_coll2]

    # check overlap in case of collinear points
    bools_onSeg = onSegment(coll_points[..., combis, :])

    # if collinear and onSegment, then also return True
    bools_coll[bools_coll2] *= bools_onSeg

    return bools + bools_coll.any(axis=-1)

# %%
def uniquetol(ar, tol, return_index=False, return_inverse=False, return_counts=False, axis=None, equal_nan=True):
    """Variant of np.unique that works with a tolerance to determine if two points are close enough to be identical. If the items to be compared are subarrays, all elements must be within the tolerance individually. Unlike np.unique, it does not sort the output

    Returns the sorted unique elements of an array. There are three optional outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array


    Parameters
    ----------
    ar : array_like
        Input array. Unless axis is specified, this will be flattened if it is not already 1-D.
    tol : float
        geometrical tolerance, indicates how different two values can be from each other to still count as 'the same'
    return_index : bool, optional
        If True, also return the indices of ar (along the specified axis, if provided, or in the flattened array) that result in the unique array, by default False
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified axis, if provided) that can be used to reconstruct ar, by default False
    return_counts : bool, optional
        If True, also return the number of times each unique item appears in ar, by default False
    axis : int or None, optional
        The axis to operate on. If None, ar will be flattened. If an integer, the subarrays indexed by the given axis will be flattened and treated as the elements of a 1-D array with the dimension of the given axis, by default None
    equal_nan : bool, optional
        If True, collapses multiple NaN values in the return array into one, by default True

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    Raises
    ------
    ValueError
        Raised if there is an inconsistency in the identicality of points; if the identicality is not transitive (i.e., there is a point A which is identical B (within the tolerance), and B is identical to C, but C is not identical to A)
    """

    ar = np.asarray(ar)

    if axis is None:
        ar = ar.flatten()
    else:
        ar = np.moveaxis(ar, axis, 0)

    orig_shape, orig_dtype = ar.shape, ar.dtype

    # reshape to always have shape (N, M), with N the axis operated on, and M the other axes flattened
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))

    bools_matrix = skln.radius_neighbors_graph(ar, radius=tol).astype(bool)

    if np.isnan(ar).any():
        raise NotImplementedError('dealing with NaNs not yet implemented')

    bm_copy = bools_matrix.copy()
    bm_copy.setdiag(1)

    # check if identicality is transitive
    dotprod = bm_copy.dot(bm_copy)

    if not dotprod.nnz == bm_copy.nnz:
        raise ValueError('Identicality is not transitive!')
    if not (dotprod != bm_copy).nnz == 0:
        raise ValueError('Identicality is not transitive!')

    # get unique
    # Assuming bools_matrix is a scipy sparse matrix (CSR format)
    bools_triu = sps.triu(bools_matrix, k=1, format='csr').astype(bool)
    temp = np.asarray(bools_triu.sum(axis=0))[0].astype(bool)
    to_keep = np.where(~temp)[0]
    ar_new = ar[to_keep]
    counts = bools_triu.sum(axis=1).A1[to_keep] + 1  # A1 converts the result to a dense numpy array

    bools_matrix.setdiag(True)
    # get inv
    inv = np.asarray(bools_matrix[to_keep].argmax(axis=0))
    inv = inv[0]

    if return_index or return_inverse or return_counts:
        to_return = [ar_new]
        if return_index:
            to_return.append(to_keep)
        if return_inverse:
            to_return.append(inv)
        if return_counts:
            to_return.append(counts)
        return to_return
    else:
        return ar_new

# %%
def detect_crossed_edges(pos, edges, return_indices=False):
    """Detect crossed edges in a mesh. This function is not optimized for large meshes. Deprecated and replaced by crossing_edges().

    Parameters
    ----------
    pos : (N, 2) array_like,
        coordinates of all N nodes,
    edges : (E, 2), array_like,
        indices into pos of all E edges
    return_indices : bool, optional
        return indices of pairs of edges that cross, by default False

    Returns
    -------
    bool
        whether any crossed edges were found
    indices : (C, 2) ndarray, optional
        pairs of indices into edges, indicating which pairs of edges cross
    """
    pos = np.asarray(pos)
    edges = np.asarray(edges)
    n_edges = edges.shape[0]
    temp = pos[edges]

    # create all possible pairs of boundary edges
    points1 = np.repeat(temp[np.newaxis, ...], n_edges, axis=0)
    points2 = np.repeat(temp[:, np.newaxis, ...], n_edges, axis=1)
    points = np.concatenate((points1, points2), axis=2)

    # points1.shape = points2.shape = [nr of b edges, nr of b edges, 2, 2]
    # points.shape = [nr of b edges, nr of b edges, 4, 2]

    # check if a pair of edges intersects
    bools = intersect(points)

    # exclude edges compared with themselves
    temp2 = np.arange(n_edges)
    bools[temp2, temp2] = False

    # only use half the matrix
    bools = np.triu(bools)

    if return_indices:
        return bools.any(), np.stack(np.where(bools), axis=-1)
    else:
        return bools.any()

# %%
# depth-first search'
def dfs(A, u, component, c):
    component[u] = c
    for v in np.where(A[u])[0]:
        if component[v] == -1:
            dfs(A, v, component, c)

def components(A):
    n = A.shape[0]
    component = np.full(n, -1, dtype=int)
    c = 0
    for i in range(n):
        if component[i] == -1:
            dfs(A, i, component, c)
            c += 1
    return component

# connectivity check taking into account periodicity
def is_connected(A, bound_inds, linked_bounds, return_component=False):
    A = np.copy(A)
    n = A.shape[0]

    done = False
    while not done:
        comps = components(A)
        if np.max(comps) == 0:
            break

        done = True
        for asdf in [False, True]:
            for b, lb, flip in linked_bounds:
                if asdf:
                    b, lb = lb, b
                # check if this boundary is coupled to another one
                if lb is not None and b is not None:
                    # check connectivity between nodes on the boundary b
                    n = len(bound_inds[b])
                    temp = A[np.ix_(bound_inds[b], bound_inds[b])]
                    temp[np.arange(n), np.arange(n)] = True

                    # find nodes on this boundary that are not connected (even if they can't be)
                    temp_edges = np.stack(np.where(~temp), axis=-1) # indices into temp
                    temp_edges2 = np.asarray(bound_inds[b])[temp_edges]  # indices into A

                    # check if both nodes are in the same component
                    bools = comps[temp_edges2[:, 0]] == comps[temp_edges2[:, 1]]

                    if len(temp_edges[bools]) != 0:
                        done = False

                    for edge in temp_edges[bools]:
                        # find dependent nodes (will be at the same index but on a different boundary, or will be at a flipped index if flip=-1

                        if flip == 1:
                            edge2 = np.asarray(bound_inds[lb])[edge]
                        else:
                            edge2 = np.asarray(bound_inds[lb])[n - 1 - edge]

                        # add this edge and the dependent edge
                        A[edge2[0], edge2[1]] = True
                        A[edge2[1], edge2[0]] = True
                        edge = np.asarray(bound_inds[b])[edge]
                        A[edge[0], edge[1]] = True
                        A[edge[1], edge[0]] = True

    if return_component:
        return np.max(comps)==0, comps
    else:
        return np.max(comps)==0

# %%
def generate_points(a1, a2, bounds, corners, rng, linked_bounds, max_b=2, max_i=5, prob_c=0.5):# generate points on boundaries
    """
    Generate points on the fundamental domain spanned by the vectors a1 and a2 (or the triangle defined by a1 and a2).

    Parameters
    ----------
    a1 : array_like, shape (2,)
        vector along bottom of fundamental domain
    a2 : array_like, shape (2,)
        vector along left side of fundamental domain
    bounds : list of tuples of ndarrays
        for each boundary its starting point and the vector pointing along it, going clockwise around the fundamental domain
    corners: array_like, shape (N, 2)
        coordinates of the N corner points
    linked_bounds : list of tuples of length 3
        each tuple indicates: (boundary, boundary dependent on it, whether the depedent boundary flips the direction)
    max_b : int, optional
        max nr of points on each boundary, by default 5
    max_i : int, optional
        max nr of points inside the fundamental domain, by default 10
    prob_c : float, optional
        probability of a corner point being present, by default 0.5

    Returns
    -------
    points : ndarray, shape (N, 2)
        coordinates of the generated points
    bound_inds : list of lists of ints
        indices of points on each boundary
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    # 1) GENERATE CORNER POINTS (optional)
    nb = len(bounds)
    cp_temp = [None]*nb
    for i, _ in enumerate(corners):
        if cp_temp[i] is None:
            if rng.random() < prob_c:
                cp_temp[i] = True
            else:
                cp_temp[i] = False

        # copy to dependent boundaries
        for i in range(3): # do this 3 times to make sure they get properly copied around (probably overkill)
            for b, lb, flip in linked_bounds:
                if lb is not None:
                    if cp_temp[b] is not None:
                        if flip == 1:  # not flipped
                            if cp_temp[lb] is None:
                                cp_temp[lb] = cp_temp[b]
                        elif flip == -1:
                            if cp_temp[(lb + 1) % nb] is None:
                                cp_temp[(lb + 1) % nb] = cp_temp[b]
                    if cp_temp[lb] is not None:
                        if flip == 1:
                            if cp_temp[b] is None:
                                cp_temp[b] = cp_temp[lb]
                        elif flip == -1:
                            if cp_temp[(b + 1) % nb] is None:
                                cp_temp[(b + 1) % nb] = cp_temp[lb]
                    if cp_temp[(b+1) % nb] is not None:
                        if flip == 1:
                            if cp_temp[(lb + 1) % nb] is None:
                                cp_temp[(lb + 1) % nb] = cp_temp[(b+1) % nb]
                        elif flip == -1:
                            if cp_temp[lb] is None:
                                cp_temp[lb] = cp_temp[(b+1) % nb]
                    if cp_temp[(lb+1) % nb] is not None:
                        if flip == 1:
                            if cp_temp[(b + 1) % nb] is None:
                                cp_temp[(b + 1) % nb] = cp_temp[(lb+1) % nb]
                        elif flip == -1:
                            if cp_temp[b] is None:
                                cp_temp[b] = cp_temp[(lb+1) % nb]
    corner_points = [[corners[i].tolist()] if cp_temp[i] else [] for i in range(nb)]

    # 2) GENERATE POINTS ON BOUNDARIES
    bound_points = [None]*nb
    for b, lb, flip in linked_bounds:

        # optionally add mid point on self-periodic boundary
        midpoint = False
        if b == lb:
            if rng.random() < prob_c:
                midpoint = True

        # subtract number of corner points and midpoint from max_b and min_b
        min_b = max(0, 1 - len(corner_points[b])
                         - len(corner_points[(b+1) % nb])
                         - midpoint
                    )
        max_b = (max_b - len(corner_points[b])
                       - len(corner_points[(b+1) % nb])
                       - midpoint
                )

        if b == lb:
            if max_b//2 <= min_b:
                n_nodes = min_b
            else:
                n_nodes = rng.integers(min_b, max_b//2+1)
        else:
            if max_b <= min_b:
                n_nodes = 1
            else:
                n_nodes = rng.integers(min_b, max_b+1)

        c = rng.random(n_nodes)
        c = np.sort(c) # so order of nodes will be clockwise

        #only use half of a boundary connected to itself
        if b == lb:
            c = c/2

        if midpoint:
            c = np.concatenate((c, [0.5]))

        bound_points[b] = []
        p = bounds[b][0] + c.reshape(-1,1)*bounds[b][1]
        bound_points[b].extend(p.tolist())

        # copy points to dependent boundaries
        if lb is not None:
            if flip==-1:
                c = 1-c
                c = np.flip(c, axis=0)
            # don't copy midpoint
            if midpoint:
                p = bounds[lb][0] + c[1:].reshape(-1,1)*bounds[lb][1]
            else:
                p = bounds[lb][0] + c.reshape(-1,1)*bounds[lb][1]

            if bound_points[lb] is None:
                bound_points[lb] = []
            bound_points[lb].extend(p.tolist())



    # interleave the boundary and corner points
    bp = []
    bound_inds = [None]*len(bounds)  # indicates which points are part of which boundary
    for i in range(len(bounds)):
        if bound_inds[i] is None:
            bound_inds[i] = []

        # add next corner point and include it in the previous boundary and the current
        bound_inds[i].extend(np.arange(len(bp), len(bp)+len(corner_points[i])))
        if i != 0:
            bound_inds[i-1].extend(np.arange(len(bp), len(bp)+len(corner_points[i])))
        bp.extend(corner_points[i])

        # add next boundary points and include in the current boundary
        bound_inds[i].extend(np.arange(len(bp), len(bp)+len(bound_points[i])))
        bp.extend(bound_points[i])

    if len(corner_points[0]) > 0:
        bound_inds[-1].append(0)

    bp = np.asarray(bp)

    # 3) GENERATE POINTS INSIDE FUNDAMENTAL DOMAIN
    # to do: use poisson or something; more points less likely
    n_nodes = rng.integers(1, max_i+1)
    c = rng.random((n_nodes, 2))
    if len(bounds) == 3:
        c[:, 0] /= 2
        bools = c[:, 1] > 1 - c[:, 0]
        c[bools, 0] = 1 - c[bools, 0]
        c[bools, 1] = 1 - c[bools, 1]
    inside_points = c[:,[0]]*a1 + c[:,[1]]*a2
    all_points = np.concatenate((bp, inside_points), axis=0)

    return all_points, bound_inds

# %%
def connect_nodes(all_points, bound_inds, linked_bounds, rng=None, max_nr_of_edges=None):
    if rng is None:
        rng = np.random.default_rng()

    n = all_points.shape[0]

    # 0: not an edge yet
    # -1: not allowed an edge (self-loop, or reverse already exists, or will overlap with another edge)
    # 1: edge
    A = np.zeros((n, n))

    # disallow all edges between nodes on the same boundary that are not adjacent
    for inds in bound_inds:
        # disallow edges between nodes on the same boundary
        A[np.ix_(inds, inds)] = -1
        # reallow them if they are adjacent
        A[inds[:-1], inds[1:]] = 0
        A[inds[1:], inds[:-1]] = 0

    # set lower triangular part to -1
    A_temp = np.ones((n, n))*-1
    A_temp = np.tril(A_temp)
    A = np.minimum(A_temp, A)

    connected = False

    i = 0
    while not connected:
        # add random edge
        # to do: not random but between nodes from different components
        allowed_edges = np.stack(np.where(A == 0), axis=-1)
        if len(allowed_edges) == 0:
            raise ValueError('no more edges possible')
        edge_to_add = rng.choice(allowed_edges)
        A[tuple(edge_to_add)] = 1

        current_edges = np.stack(np.where(A == 1), axis=-1)

        # check if the nodes of this edge are both in the same boundary which has another boundary dependent on it
        dep_edge = None
        for asdf in [False, True]:
            for b, lb, flip in linked_bounds:
                if asdf:
                    b, lb = lb, b
                if lb is not None and b is not None:
                    n = len(bound_inds[b])
                    if np.isin(edge_to_add, bound_inds[b]).all():
                        inds = np.where(np.isin(bound_inds[b], edge_to_add))[0]
                        if flip == 1:
                            dep_edge = np.sort(np.asarray(bound_inds[lb])[inds])
                        else:
                            dep_edge = np.sort(np.asarray(bound_inds[lb])[n-1-inds])

        # either add dependent edge too, or disallow both
        if dep_edge is not None:
            if A[tuple(dep_edge)] == -1:  # if dependent edge is not allowed, then edge_to_add is not allowed either
                A[tuple(edge_to_add)] = -1
                continue
            else:
                A[tuple(dep_edge)] = 1

        # check if this results in crossed edges
        # print('all_points', all_points)
        # print('current_edges', current_edges)
        cross_inds = crossing_edges(all_points, current_edges)
        crossed = len(cross_inds) > 0

        if crossed:
            # mark edge (and if necessary, its dependent edge) as not allowed
            A[tuple(edge_to_add)] = -1
            if dep_edge is not None:
                A[tuple(dep_edge)] = -1
        else:
            # check connectivity
            A2 = (A == 1)
            A2 = A2 + A2.T
            conn, _ = is_connected(A2, bound_inds, linked_bounds, return_component=True)

            if conn:
                break

        i += 1

        if max_nr_of_edges is not None:
            if np.sum(A==1) >= max_nr_of_edges:
                break

    return np.stack(np.where(A == 1), axis=-1)


# %%
def new_path(path, n=2, always_number=True):
    '''Create a new path for a file
    with a filename that does not exist yet,
    by appending a number to path (before the extension) with at least n digits.
    (Higher n adds extra leading zeros). If always_number=True,
    the filename will always have a number appended, even if the
    bare filename does not exist either.'''

    # initial path is path + name + n zeros + extension if always_number=True
    # and just path + filename if always_number=False
    name, ext = os.path.splitext(path)
    if always_number:
        savepath = f'{name}_{str(0).zfill(n)}{ext}'
    else:
        savepath = path

    # If filename already exists, add number to it
    i = 1
    while os.path.exists(savepath):
        savepath = f'{name}_{str(i).zfill(n)}{ext}'
        i += 1
    return savepath

# %%
def iscollinear(points, A, B, tol_g):
    """For each point in points, gives whether the points are collinear with the points A and B, within a certain tolerance

    Parameters
    ----------
    points : array-like of shape (N, 2)
    A : array_like of shape (2,)
    B : array_like of shape (2,)
    B = np.asarray(B)
    tol_g : float
        tolerance in cross product between sides of triangle to count as zero

    Returns
    -------
    ndarray of bools (N,)
        whether the point is collinear with A and B
    """

    p = np.asarray(points)
    q = np.asarray(A)
    r = np.asarray(B)

    # calculates (r-q)×(q-p)
    val = (q[..., 1]-p[..., 1]) * (r[..., 0]-q[..., 0]) - (q[..., 0]-p[..., 0]) * (r[..., 1]-q[..., 1])

    return np.abs(val) < tol_g

# Create list of equivalent nodes
def list_equivalent_nodes(bound_inds, linked_bounds, n, include_self=False):
    equiv_nodes = [[] for i in range(n)]
    for b, lb, flip in linked_bounds:
        if lb is not None:
            for rev in [False, True]:
                if rev:
                    b, lb = lb, b
                for i, ind in enumerate(bound_inds[b]):
                    n_temp = len(bound_inds[lb])
                    if flip == -1:
                        equiv_node = bound_inds[lb][n_temp-i-1]
                    else:
                        equiv_node = bound_inds[lb][i]

                    # add equivalent node if it's not already in there
                    if equiv_node not in equiv_nodes[ind]:
                        if include_self or equiv_node != ind:
                            equiv_nodes[ind].append(equiv_node)
                    # also add equivalent nodes of the equivalent node
                    equiv_nodes[ind].extend([ind2 for ind2 in equiv_nodes[equiv_node] if (ind2 not in equiv_nodes[ind]) and (ind2!=ind)])
    return equiv_nodes

# Create list of dependent edges
def list_equivalent_edges(bound_inds, linked_bounds, edges, directional=False):
    '''directional: whether the directions of the edge must also match with its equivalent edge (default False)'''
    equiv_edges = []

    # iterate over edges
    for e, edge in enumerate(edges):
        # iterate over boundaries
        for b, lb, flip in linked_bounds:
            if lb is None:
                continue
            n_temp = len(bound_inds[b])
            # check if both nodes of the edge are on this boundary
            if edge[0] in bound_inds[b] and edge[1] in bound_inds[b]:

                ind0 = np.where(bound_inds[b] == edge[0])[0][0]
                ind1 = np.where(bound_inds[b] == edge[1])[0][0]
                if flip == -1:
                    ind2 = bound_inds[lb][n_temp-1-ind0]
                    ind3 = bound_inds[lb][n_temp-1-ind1]
                else:
                    ind2 = bound_inds[lb][ind0]
                    ind3 = bound_inds[lb][ind1]

                # find the edge between the equivalent nodes
                if directional:
                    e2 = np.where((edges[:, 0] == ind2)*(edges[:, 1] == ind3))[0][0]
                else:
                    e2 = np.where((edges[:, 0] == ind2)*(edges[:, 1] == ind3) + (edges[:, 0] == ind3)*(edges[:, 1] == ind2))[0][0]
                equiv_edges.append([e, e2])
    return np.concatenate((equiv_edges, np.flip(equiv_edges, axis=-1)), axis=0)

def draw_skeleton(polygon, skeleton, show_time=False):
    sg.draw.draw(polygon)

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point
            plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=2)

    if show_time:
        for v in skeleton.vertices:
            plt.gcf().gca().add_artist(plt.Circle(
                (v.point.x(), v.point.y()),
                v.time, color='blue', fill=False))

# find angles again
def find_angles(uc):

    triplets = []
    angles = []

    tol_g = 1e-6
    visited = np.zeros(len(uc["edges_directional"]), dtype=bool)
    holes = []
    while not visited.all():
        e = np.where(~visited)[0][0]
        # print(f'================== NEW HOLE, starting with {e} ==================')
        visited[e] = True

        # equivalent edge also gets marked visited
        if len(uc['periodic_edges']) > 0:
            if e in uc['periodic_edges'][:, 0]:
                equiv_edge = uc['periodic_edges'][np.where(uc['periodic_edges'][:, 0]==e)[0][0], 1]
                # print('e, equiv_edge', e, equiv_edge)
                visited[equiv_edge] = True
        done = False
        hole = [e,]
        while not done:
            edges = uc["edges_directional"]
            edge = edges[e]
            # print('=============', edge, '=============')
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
            # print(conn_edges)

            # vectors pointing along connected edges
            conn_vecs = uc["points"][conn_edges[:, 1]] - uc["points"][conn_edges[:, 0]]

            # get all angles relative to x-axis
            theta1 = np.arctan2(*np.flip(edge_vec, axis=-1))
            theta2 = np.arctan2(*np.flip(conn_vecs, axis=-1).T)

            # angle of connected edges to current edge
            theta = (theta2 - theta1) % (2*np.pi)
            theta[np.abs(theta) < tol_g] = 2*np.pi
            # print(f'{theta/np.pi}π')
            for conn_edge, theta_temp in zip(conn_edges, theta):
                triplets.append([*edge, *conn_edge])
                angles.append(theta_temp)

            # find smallest angle (to go clockwise)
            e = inds[np.argmin(theta)]
            # print('Choose:', e, edges[e])
            if visited[e] and (e == hole[0] or [e, hole[0]] in uc['periodic_edges']):
                done = True
                holes.append(hole)
            elif visited[e]:
                holes.append(hole)
                # print('Holes:', holes)
                raise Exception(f'Next edge {e} is already visited but is not equal to the first edge {hole[0]}')
            else:
                visited[e] = True
                # equivalent edge also gets marked visited
                if len(uc['periodic_edges']) > 0:
                    if e in uc['periodic_edges'][:, 0]:
                        peri_edge = uc['periodic_edges'][np.where(uc['periodic_edges'][:, 0]==e)[0][0], 1]
                        # print('e, peri_edge', e, peri_edge)
                        visited[peri_edge] = True
                hole.append(e)

    return triplets, angles

def interpolate_bezier4(points, closed=False, return_tangents=False):
    """Create list of control points for cubic Bezier curve that goes smoothly through all given points. 'Closed' indicates if the curve should be periodic. (first point is also end point)

    Parameters
    ----------
    points : array-like, [N, D]
        array of D-dimensional coordinates of N points through which to draw a Bézier curve
    closed : bool, optional
        whether the curve is periodic (first point is also end point), by default False
    return_tangents : bool, optional
        whether to return the tangent vectors as well, by default False

    Returns
    -------
    verts : ndarray, shape [N*3-2, D] if closed=False, shape [N*3, D] if closed=True
        control for a cubic bezier curve that goes through all points
    tangents : optionally
        tangent vector of length 1 for each point
    """
    points = np.asarray(points)

    if closed:
        points = np.concatenate((points, points[[0]]), axis=0)

    n_points = len(points)
    diff = np.diff(points, axis=0)
    # vectors pointing from point to point, normalized to have length 1
    dvec = diff/np.linalg.norm(diff, axis=-1, keepdims=True)

    if closed:
        tangents = np.empty((n_points,2))
        tangents[1:-1] = dvec[:-1] + dvec[1:]
        tangents[0] = dvec[-1] + dvec[0]
        tangents[-1] = tangents[0]
    else:
        tangents = np.empty((n_points,2))
        tangents[1:-1] = dvec[:-1] + dvec[1:]
        tangents[0] = dvec[0]
        tangents[-1] = dvec[-1]
    tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)


    # control points: always [cttcttctt....ttc] with c: on-curve points, t: tangent points
    n_points = len(points)
    verts = np.empty((n_points*3-2, 2))

    # coordinates of control points that are on the curve
    verts[::3] = points

    d = np.linalg.norm(diff, axis=-1)
    V0 = tangents[:-1]
    V1 = tangents[1:]
    P0 = points[:-1]
    P1 = points[1:]

    # coordinates of control points after on-curve-points
    verts[1::3] = P0 + d.reshape(-1, 1)*V0*1/3

    # coordinates of control points before on-curve-points
    verts[2::3] = P1 - d.reshape(-1, 1)*V1*1/3

    # if closed, remove extra point & tangent that were added
    if closed:
        verts = verts[:-1]
        tangents = tangents[:-1]

    if return_tangents:
        return verts, tangents
    else:
        return verts


def polygon_area(xy, signed=False):
    """Calculate the area of a polygon defined by its corner points. Uses broadcasted einsum to calculate the area of multiple polygons at once.

    Parameters
    ----------
    xy : ndarray, shape (..., N, 2)
        x,y-coordinates of N corner points of multiple polygons
    signed : bool, optional
        whether to return a negative area if the polygon is flipped (i.e., points are specified counterclockwise), by default False

    Returns
    -------
    ndarray
        shape will be the same as xy, except the last two dimensions will be removed
    """

    # xy:
    # shape: (nr of polygons, nr of corner points, 2)
    correction = xy[:, -1, 0] * xy[:, 0, 1] - xy[:, -1, 1]* xy[:, 0, 0]
    main_area = (np.einsum('...i,...i->...', xy[:, :-1, 0], xy[:, 1:, 1])
                - np.einsum('...i,...i->...', xy[:, :-1, 1], xy[:, 1:, 0])
                )

    if signed:
        return 0.5*(main_area + correction)
    else:
        return 0.5*np.abs(main_area + correction)

def distance_bezier4(control_points1, control_points2, n=64):
    segm1 = mbez.BezierSegment(control_points1)
    segm2 = mbez.BezierSegment(control_points2)

    sample_points = segm1.point_at_t(np.linspace(0, 1, n))
    sample_points2 = segm2.point_at_t(np.linspace(0, 1, n))

    distances = np.linalg.norm(np.array(sample_points)[np.newaxis] - np.array(sample_points2)[:, np.newaxis], axis=-1)
    return np.min(distances)

def bezier4_too_close(control_points, min_distance, n=64, ignore_joined_ends=False, tol=1e-6):
    """Check if any of the quadratic Bézier segments defined by control_points are too close together, meaning the distance between two segments is less than min_distance. This is done by taking n samples of each segment and checking the distance between all pairs of samples. If the distance is smaller than min_distance, the segments are considered to overlap.

    Parameters
    ----------
    control_points : ndarray, shape (n_segments, 4, 2)
        4 control points of each segment
    min_distance : float
        minimum allowed distance between segments
    n : int, optional
        number of samples to take of each segment, by default 64
    ignore_joined_ends : bool, optional
        ignore Bézier curves that have an end point in common, by default False
    tol : float, optional
        tolerance for determining if end points match, only used if ignore_joined_ends=True, by default 1e-6

    Returns
    -------
    ndarray, shape (n_pairs, 2)
        pairs of indices of segments that are too close together
    """

    sample_points = np.array([mbez.BezierSegment(cp).point_at_t(np.linspace(0, 1, n)) for cp in control_points])

    sample_points = np.array([mbez.BezierSegment(cp).point_at_t(np.linspace(0, 1, n)) for cp in control_points])

    # matrix indicating which points are within min_distance of each other
    bools_matrix = skln.radius_neighbors_graph(sample_points.reshape(-1, 2), radius=min_distance).astype(bool)

    bools_matrix = bools_matrix.tocoo()
    row = bools_matrix.row
    col = bools_matrix.col

    pairs = np.stack((row, col), axis=1)

    # for each point, get index of its segment
    inds = pairs//n

    # check for each pair if it connects two different segments
    segment_pairs = inds[np.where(inds[:, 0] != inds[:, 1])]

    # deduplicate
    segment_pairs = np.unique(np.sort(segment_pairs, axis=-1), axis=0)

    if ignore_joined_ends:
        # check if the two segments are joined at the ends, if so, ignore them
        start_points = control_points[segment_pairs, 0, :]  # [segment, point, xy]
        end_points = control_points[segment_pairs, 3, :]  # [segment, point, xy]

        # check start against start points
        bools1 = (np.abs(start_points[:, 0] - start_points[:, 1]) > tol).any(axis=1)
        # check start against end points
        bools2 = (np.abs(start_points[:, 0] - end_points[:, 1]) > tol).any(axis=1)
        # check end against start points
        bools3 = (np.abs(end_points[:, 0] - start_points[:, 1]) > tol).any(axis=1)
        # check end against end points
        bools4 = (np.abs(end_points[:, 0] - end_points[:, 1]) > tol).any(axis=1)

        segment_pairs = segment_pairs[bools1*bools2*bools3*bools4]

    return segment_pairs

from sklearn.neighbors import NearestNeighbors

def crossing_edges(xy, edges, verbose=False):
    """
    Find the pairs of edges that cross each other.
    """
    n = len(xy)
    E = len(edges)

    xy = np.asarray(xy)
    edges = np.asarray(edges)

    # calculate the maximum edge length
    r = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    max_r = np.max(r)
    # print('max_r:', max_r)

    # calculate max nr of neighbors
    # _, counts = np.unique(edges.flatten(), return_counts=True)
    # max_n_neighbors = np.max(counts)

    # nbrs = NearestNeighbors(n_neighbors=max_n_neighbors*2+1).fit(xy)
    # bools_matrix = nbrs.kneighbors_graph(xy).astype(bool)

    # use radius_neighbors_graph to find nodes that are within 2*max_r of each other, and therefore could potentially be part of edges that cross
    bools_matrix = skln.radius_neighbors_graph(xy, radius=2*max_r).astype(bool)
    # print('bools_matrix:\n', bools_matrix)

    bools_matrix.eliminate_zeros()
    # print(f'n={n}, n^2={n**2}')
    if verbose:
        print('bools_matrix.nnz:', bools_matrix.nnz, '(nr of pairs nodes close enough that their edges could cross)')
    # print('bools_matrix:\n', bools_matrix)

    data = np.ones(E*2)
    row = np.concatenate((np.arange(E), np.arange(E)))
    col = np.concatenate((edges[:, 0], edges[:, 1]))

    # pairs of edges that have nodes that are close enough that they could cross
    edges_temp = sps.coo_matrix((data, (row, col)), shape=(E,n), dtype=int)
    edge_to_edge = edges_temp.dot(bools_matrix.astype(int)).dot(edges_temp.T)
    edge_to_edge = sps.coo_matrix(edge_to_edge)
    # eliminate edge with itself
    edge_to_edge.setdiag(0)
    edge_to_edge.eliminate_zeros()
    # print('edge_to_edge:', edge_to_edge)
    if verbose:
        print('edge_to_edge.nnz:', edge_to_edge.nnz, '(nr of pairs of edges that have nodes that are close enough that they could cross)')

    # edges AB and CD can only cross if distance AC, AD, BC, BD are all smaller than 2*max_r, so the value in edge_to_edge should be 4
    counts = edge_to_edge.data
    # print('edge_to_edge.data:', edge_to_edge.data)
    edge_to_edge = np.array(edge_to_edge.nonzero()).T
    # print('edge_to_edge:', edge_to_edge)
    edge_to_edge = edge_to_edge[(edge_to_edge[:, 0] < edge_to_edge[:, 1])*(counts==4)]  # only keep sorted pairs
    # print('edge_to_edge:', edge_to_edge)

    points = xy[edges[edge_to_edge].reshape(-1, 4)]
    if verbose:
        print('points.shape:', points.shape, '(shape of the points given to intersect function, which checks if line segments intersect)')
    bools = intersect(points)

    return edge_to_edge[bools]

if __name__ == '__main__':
    # rng = np.random.default_rng()
    # p = rng.random((10, 2))
    # print(p)
    # e = rng.integers(0, 10, (5, 2))
    # print(e)

    p = [[4.42914494e-02, 0.00000000e+00],
 [4.55708551e-01, 0.00000000e+00],
 [4.31180556e-01, 1.37638887e-01],
 [6.88194436e-02, 8.62361113e-01],
 [3.41458769e-17, 5.57644489e-01],
 [3.21166853e-01, 2.55522084e-01],
 [1.34179589e-01, 2.29057111e-01],
 [1.53190802e-01, 2.86861248e-01]]

    e = [[0, 2],
    [0, 3],
    [0, 6],
    [0, 7],
    [2, 5],
    [2, 7],
    [3, 4],
    [3, 5],
    [3, 7],
    [5, 6],
    [5, 7],
    [6, 7]]

    plt.scatter(*np.asarray(p).T)


    x, y = np.transpose(np.asarray(p)[np.asarray(e).T], axes=[2,0,1])
    edges0 = plt.plot(x, y, alpha=0.5)
    plt.show()

    crossed, cross_inds0 = detect_crossed_edges(p, e, return_indices=True)
    cross_inds1 = crossing_edges(p, e)

    print(crossed)
    print(cross_inds0)
    print(len(cross_inds1) > 0)
    print(cross_inds1)

    print(detect_crossed_edges(p, e, return_indices=True))
    print(crossing_edges(p, e))

    print('test:')
    print(np.all(cross_inds0 == cross_inds1))