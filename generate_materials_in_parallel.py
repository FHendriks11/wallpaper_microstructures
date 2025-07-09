# %%
import concurrent.futures
import os
import datetime

from data_myMeshes import generate_material_geometry
from helper_funcs import wallpaper_groups
from helper_funcs import new_path

# directory to save the generated materials
# if it does not exist, it will be created
data_dir = 'data'

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

figures = 1
verbose = False

# Define a function to generate one material geometry
# This function will be called multiple times in parallel
# It will create a new folder for each geometry and save the generated material there
def _generate_material_geometry(group, shape):

    # keep trying until succesful
    for i in range(100):
        # Create new folder for figures
        date_time_string = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
        name = f'{group}_{shape}_{date_time_string}'

        save_dir = new_path(os.path.join(data_dir, name), always_number=False)
        # safely create new folder
        while(True):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                break
            else:
                save_dir = new_path(os.path.join(data_dir, name), always_number=False)

        try:
            # generate new material
            generate_material_geometry(group, shape, verbose=verbose, figures=figures, save_dir=save_dir)
        except Exception as e:
            with(open(new_path(os.path.join(save_dir, 'error.txt')), 'w')) as f:
                f.write(repr(e))
            print(repr(e))
        else:   # if no exception -> successful! -> break loop
            break
    else:  # no break -> failed 100 times
        print(f'Failed to generate {group} {shape} 100 times!')

# %%
# Print all options
for group in wallpaper_groups:
    print(group)
    for shape in wallpaper_groups[group]['fundamental domain parameters']:
        print('  -', shape)

# %%
# Define the arguments to pass to the function
args1 = []
args2 = []
n = 60
for group in wallpaper_groups:
    print(group)
    args1.extend([group]*n)
    shapes = wallpaper_groups[group]['fundamental domain parameters'].keys()
    shapes = list(shapes)
    for i in range(n):
        args2.append(shapes[i % len(shapes)])  # cycle through shapes

assert len(args1) == len(args2), 'Length of args1 and args2 should be the same'

# %%
def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for results in executor.map(_generate_material_geometry, args1, args2):
            pass

    error_dir = os.path.join(data_dir, 'error_geometries')
    if not os.path.exists(error_dir):
        os.mkdir(error_dir)

    # move all folders with an error_00.txt file to error_geometries folder
    print('Failed:')
    for folder in os.listdir(data_dir):
        if folder == 'error_geometries':
            continue
        if not os.path.isdir(os.path.join(data_dir, folder)):
            continue
        if not os.path.exists(os.path.join(data_dir, folder, 'error_00.txt')):
            continue

        print(folder)

        # move folder to error_geometries folder
        os.rename(os.path.join(data_dir, folder), os.path.join(data_dir, 'error_geometries', folder))

if __name__ == '__main__':
    main()
