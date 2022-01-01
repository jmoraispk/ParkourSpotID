# -*- coding: utf-8 -*-
"""
Created on Nov 29 2021

@author: Joao

-------------------------------------------------
Phase A:
    1- Get 1 coordinate and radius (side of square)
    2- Get area of interest
    3- See which points of the uniform grid are inside the area of interest
    4- Run loop for these points:

Phase B: LOOP
    1- 

Phase C: Display results
-------------------------------------------------

Problems for future work:
    1- The GOOGLE API does not maintain uniformity when zoom levels change. 
       Therefore, we need to take all pictures with the maximum zoom to build
       the big pictures.
    2- Latitude scales (not Longitude) change depending on the coordinates.
    Since the scale is not uniform, it is difficult to build a uniform grid.
    We built a function to test scales: 
    # test_scale(coords=coords, scale=2.5, n_pics=20, lat_or_long='lon', zoom=21)



    3- Currently, the top-view model is not used. But we can filter tremendously
    which locations might be worth fetching street view images from based on 
    top-view images with different zooms. The ASU campus area is 2.7 km2. 
    Say we fetched around 3 km2 and it costed 30$

"""

import os
import utm
import time
import shutil
import requests
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

# Folders for data storage
top_view_folder = r'./Database images/Top view/'
street_view_folder = r'./Database images/Street view/'
street_view_results_folder = r'./Street view predictions/'

if not os.path.isdir(top_view_folder):
    os.makedirs(top_view_folder)
if not os.path.isdir(street_view_folder):
    os.makedirs(street_view_folder)
if not os.path.isdir(street_view_results_folder):
    os.makedirs(street_view_results_folder)
     
# Relevant constants and parameters
SAT_ZOOM = 21
N_PIXELS = 640
ORIGINAL_GOOGLE_PIX_PER_IMG = 256
LON_SCALE = N_PIXELS / ORIGINAL_GOOGLE_PIX_PER_IMG # = 2.5
LAT_SCALE = 2.067 # use test_scale() to find this number
DELTA_LAT = 360/(2**SAT_ZOOM) * LAT_SCALE # Lat. diff of adjacent requests [º]
DELTA_LON = 360/(2**SAT_ZOOM) * LON_SCALE # Lon. diff of adjacent requests [º]

# API related functions
API_KEY = "BIz3SyC5upoHhYy5fMFMJxYHHYIU70wAAs1LXXj" 
# -> this api key is just an example and IT WILL NOT WORK.
# go to https://developers.google.com/maps/documentation/embed/get-api-key
# to create your own API key. 
# Pricing: 7$ / 1k requests (street view) | 2$ / 1k requests (satellite view)

def dl_sat_img(params):
    """ 
    Params = {'lat': ..., 'long': ..., 'idx': ..., 'path': ..., 'zoom': ...}    
    Downloads the satellite image of a given coordinate and saves to the path.
    Zoom is normally set to 21. Idx is just for logging and debugging purposes.
    """
    headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
        }    
    
    url = "https://maps.googleapis.com/maps/api/staticmap"
    uri = url+"?center="+f'{params["lat"]},{params["long"]}'+\
                "&zoom="+str(params["zoom"])+\
                "&size="+params["size"]+"&maptype="+"satellite"+\
                "&key="+API_KEY+"&format="+"png"+"&visual_refresh=true"
    
    print(f"Downloading satellite image for {params['path']}...")
    
    try:
        with requests.get(url=uri, headers=headers, stream=True) as response:
            if response.status_code == 200:
                response.raw.decode_content = True
                with open(params['path'], "wb") as f:
                    shutil.copyfileobj(response.raw, f)

        return (params["idx"], response.status_code)
    
    except Exception as e:
        print(e)
        return (params["idx"], 000)

def dl_street_img(params):
    """ 
    Params = {'lat': ..., 'long': ..., 'idx': ..., 'path': ..., 
              'heading': ..., 'tilt': ...}    
    Downloads the satellite image of a given coordinate and saves to the path.
    Zoom is normally set to 21. Idx is just for logging and debugging purposes.
    """
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/70.0.3538.77 "
                             "Safari/537.36",
        }    
    
    url = "https://maps.googleapis.com/maps/api/streetview"
    uri = (url + f"?size={params['size']}&location={params['lat']},{params['long']}" + 
           f"&heading={params['heading']}&pitch={params['tilt']}&"
           f"fov={params['fov']}&key={API_KEY}")
    
    print(f"Downloading street vew image for {params['path']}...")
    
    try:
        with requests.get(url=uri, headers=headers, stream=True) as response:
            if response.status_code == 200:
                response.raw.decode_content = True
                with open(params['path'], "wb") as f:
                    shutil.copyfileobj(response.raw, f)
        return (params["idx"], response.status_code)
    except Exception as e:
        print(e)
        return (params["idx"], 000)

def get_meters_per_pixel(lat=33, zoom=21):
    """ Returns the number of meters that corresponds a pixel in the requested 
    image. """
    if lat < -90 or lat > 90:
        raise Exception('You probably inputed the longitude...')
    
    R_EARTH = 6378137 # [m]
    perimeter_earth = 2 * R_EARTH * np.pi
    perimeter_at_lat = np.cos(lat * np.pi/180) * perimeter_earth
    meters_per_pixel = perimeter_at_lat / (2 ** zoom) / 256 
    return meters_per_pixel 

def get_diameter_of_sat_img(lat=33, zoom=21):
    """ Gets the meters from left to right (or top to bottom) across a requested
    image. If a circle is inscribed in the square image, this is its diameter.
    """
    return get_meters_per_pixel(lat,zoom) * N_PIXELS

def xy_from_latlong(lat_long):
    """ Assumes lat and long along row. Returns same row vec/matrix on 
    cartesian coords."""
    # if there's just one coordinate
    if len(lat_long.shape) == 1:
        lat_long = np.reshape(lat_long, (1, lat_long.shape[0]))
    
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)

def dist_between_coords(coord1, coord2):
    " More precise than the other methods of obtaining distances."
    coord1_cartesian = xy_from_latlong(coord1)
    coord2_cartesian = xy_from_latlong(coord2)
    # sqrt(diff[0]**2 + diff[1]**2)
    return np.linalg.norm(coord2_cartesian - coord1_cartesian)

def test_scale(coords, scale, n_pics=5, lat_or_long='lat', zoom=21):
    """
    Downloads and appends 5 pictures that differ either in latitude or longitude
    an angle delta, that we derive from the scale and zoom. 
    The images are supposed to fit together continuously, and that's how we 
    assess if the scale parameter is appropriate.
    """
    if lat_or_long != 'lat':
        print('Note: longitude scales do not change with coordinates.'
              'Thus, the current formula will compute the correct scale.')
    
    delta = 360 / (2**zoom) * scale
    
    test_scale_folder = r'./test_scales/'
    if not os.path.isdir(test_scale_folder):
        os.makedirs(test_scale_folder)
        
    img_filenames = []
    for i in range(n_pics):
        lat = coords[0]
        long = coords[1]
        if lat_or_long == 'lat':
            lat -= delta * i
        else:
            long += delta * i
        img_filename = \
            os.path.join(test_scale_folder, (f'{coords[0]},{coords[1]},zoom={zoom},'
                                             f'scale={scale},{lat_or_long}_{i}.png'))
        params_sat = {'lat': lat,
                      'long':long,
                      'idx': i,
                      'path': img_filename,
                      'zoom': zoom,
                      'size': '1000x1000'}
        img_filenames.append(img_filename)
        if os.path.exists(img_filename):
           print(f'{img_filename} already exists.') 
        else:
            dl_sat_img(params_sat)
        
    separator_len = 10
    if lat_or_long == 'lat':
        final_img = np.zeros((1,N_PIXELS,3))
        separator = np.ones((separator_len,N_PIXELS,3))
        concat_axis = 0
    else:
        final_img = np.zeros((N_PIXELS,1,3))
        separator = np.ones((N_PIXELS,separator_len,3))
        concat_axis = 1
    
    for i in range(n_pics):
        img = plt.imread(img_filenames[i])
        final_img = np.concatenate((final_img, img[:,:,:3]), axis=concat_axis)
        final_img = np.concatenate((final_img, separator), axis=concat_axis)
    
    plt.imshow(final_img)
    plt.savefig('test_scale_final_figure.png', dpi=300)

def all_comb(list_of_arrays):
    """ Returns all combinations of elements of numpy arrays. 
    [[1,2,3], [4,5]] -> [[1,4], [1,5], [2,4], [2,5], [3,4], [3,5]]"""
    return np.stack(np.meshgrid(*list_of_arrays), -1).reshape(-1, len(list_of_arrays))

def get_angle_of_dist(origin_coord, dist, lat_or_long='lat'):
    """
    Returns the angle correspondent to a certain distance from the origin
    coordinate, along latitude or longitude.
    """
    small_angle = 0.001
    
    if 'lat' in lat_or_long:
        new_coord = coord + [small_angle,0]
    elif 'lon' in lat_or_long:
        new_coord = coord + [0,small_angle]
    else:
        raise Exception('Invalid "lat_or_lon". Input "lat" or "lon".')
    
    # Measure the distance caused by that angle
    dist_small_ang = dist_between_coords(coord, new_coord)
    
    # if small_angle causes dist x... then dist y causes so much angle
    # small_angle/dist_small ang = UNKNOWN ANGLE/ dist
    
    return small_angle / dist_small_ang * dist

def get_coord_at_dist(coord, dist, direction='N'):
    """
    Returns a coordinate at <dist> meters from <coord> in the <direction>.
    Works 'well' for 'small' angles.  
    """
    
    # If 'N' or 'S' -> lat
    # If 'E' or 'W' -> lon
    if 'N' in direction or 'S' in direction:
        lat_or_lon = 'lat'
    elif 'E' in direction or 'W' in direction:
        lat_or_lon = 'lon'
    else:
        raise Exception('Invalid direction.')
    
    diff_angle = get_angle_of_dist(coord, dist, lat_or_lon)
    
    if lat_or_lon == 'lat':
        coord_at_dist = coord + [diff_angle, 0]
    else: #long
        coord_at_dist = coord + [0, diff_angle]
    
    return coord_at_dist
    
#%% Street View Model Loading procedures

import sys
sys.path.insert(0,'/mrcnn/')
import mrcnn, mrcnn.config, mrcnn.model, mrcnn.visualize, cv2

WEIGHTS_PATH = "./final models/mask_rcnn_object_0057.h5"  

class CustomConfig(mrcnn.config.Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3  
    STEPS_PER_EPOCH = 15
    VALIDATION_STEPS= 1
    BATCH_SIZE=1
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.76
    
# config = CustomConfig()
# config.display()

def get_ax(rows=1, cols=1, size=16):
    """ Return an Axes to plot."""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

model = mrcnn.model.MaskRCNN(mode="inference", model_dir='./', 
                             config=CustomConfig())
weights_path = WEIGHTS_PATH

print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# seperate_thresholds = [0.0,0.72,0.85,0.6]
# The class ids are:- short wall:1 stairs:2 railings:3 (BG is BackGround)
CLASS_NAMES = ['BG','Short wall','Stairs','Railings']

def evaluate_streetview_img(file, output_img_path=None):
    """
    Loads image file, runs the Mask-RCNN inference on it, outputs the 
    annotated image to the output_img_path, and returns the number 
    of occurences of each class. 
    """
    img = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
    
    start_time = time.time() 
    # add verbose=1 to display some tensor and image related stats 
    results1 = model.detect([img])
    print(f"--- Prediction time in seconds {time.time() - start_time:2f} ---")
    ax = get_ax(1)
    r1 = results1[0]  
    
    # Count of each class
    sw = np.sum(r1['class_ids'] == 1)
    stairs = np.sum(r1['class_ids'] == 2)
    rail = np.sum(r1['class_ids'] == 3)
    
    pred_str = (f"Short walls = {sw}; Railings = {rail}; Stairs = {stairs}")
    print(pred_str)
    mrcnn.visualize.display_instances(img, r1['rois'], r1['masks'], 
                                      r1['class_ids'], CLASS_NAMES, 
                                      r1['scores'], ax=ax, title=pred_str,
                                      output_path=output_img_path)
    return (sw, stairs, rail)

# Test if it's working:
    
# import glob
# for file in glob.glob("./MRCNN training/test/*.png"):
#     print(file)
#     per_class_hits, annotaded_img = evaluate_streetview_img(file)
#     n_class_hits = sum(per_class_hits)
#     break

#%%

coords = np.array([33.42025897, -111.935378]) # Latitude, Longitude (theta, phi)

# Note: a way to avoid always downloading images is to use coordinates that 
# are an integer delta away from a certain coordinate. 
coords += [DELTA_LAT * -5, DELTA_LON * 6] # ASU campus coordinate

#%%
r_search = 500
r_search_lat = r_search 
r_search_lon = r_search 

# 1- Compute how many coordinates to adquire in each direction
#    First calc the real distance between screenshots to find how many are needed 
delta_lat_dist = dist_between_coords(coords, coords + [DELTA_LAT, 0])
delta_lon_dist = dist_between_coords(coords, coords + [0, DELTA_LON])

n_coords_in_lat = 0
while r_search_lat > (n_coords_in_lat+0.5) * delta_lat_dist:
    n_coords_in_lat += 1
n_coords_in_lon = 0
while r_search_lon > (n_coords_in_lon+0.5) * delta_lon_dist:
    n_coords_in_lon += 1

# 2- List coordinates
n_lats = n_coords_in_lat * 2 + 1
n_lons = n_coords_in_lon * 2 + 1
lats = np.array([coords[0] + DELTA_LAT * (-n_coords_in_lat + i) for i in range(n_lats)])
lons = np.array([coords[1] + DELTA_LON * (-n_coords_in_lon + i) for i in range(n_lons)])

all_coords = all_comb([lats, lons])

#%%
# 3- Download all pictures
filenames = []
for coord in all_coords:
    img_file = os.path.join(top_view_folder, f'[{coord[0]},{coord[1]}].png')
    filenames.append(img_file)
    params_sat = {'lat':  coord[0],
                  'long': coord[1],
                  'idx': 0,
                  'path': img_file,
                  'zoom': SAT_ZOOM,
                  'size': '1000x1000'}
    
    if not os.path.exists(params_sat['path']):
        dl_sat_img(params_sat)
    
    # img = plt.imread(params_sat['path'])
    # plt.imshow(img)
    # plt.show()
#%%
# 4- Put pictures back together in big image

# Always starts with the smallest latitude, and smallest longitude.
# Then sweeps all longitudes before switching latitude.
final_img = np.zeros((n_lats * N_PIXELS, n_lons * N_PIXELS, 3))

all_idxs = all_comb([np.arange(n_lats), np.arange(n_lons)])

for idxs in tqdm(all_idxs):
    j = idxs[0]
    i = idxs[1]
    img = plt.imread(filenames[j + i*n_lons])
    final_img[j*N_PIXELS:(j+1)*N_PIXELS, i*N_PIXELS:(i+1)*N_PIXELS] = np.flipud(img[:,:,:3])

final_img = np.flipud(final_img)

# print('Displaying concatenated image.')
# im_ax=plt.imshow(final_img)
# plt.savefig(f'composed_fig_{time.time()}.png', dpi=500)
#%%
# 5- Enable coordinate plotting on big picture + plot search radius

lat_to_corner = DELTA_LAT * (n_coords_in_lat+.5)
lon_to_corner = DELTA_LON * (n_coords_in_lon+.5)

def get_corners(center, lat_delta, lon_delta):
    """
    Returns (latitude,longitude) of each corner 
    In order: top right, bot right, bot left, top left.
    """
    return np.array([[center[0] + lat_delta, center[1] + lon_delta],
                     [center[0] - lat_delta, center[1] + lon_delta],
                     [center[0] - lat_delta, center[1] - lon_delta],
                     [center[0] + lat_delta, center[1] - lon_delta]])

corner_coords = get_corners(coords, lat_to_corner, lon_to_corner)

# x is longitude, y is latitude
xy_lims = [min(corner_coords[:,1]), max(corner_coords[:,1]), 
           min(corner_coords[:,0]), max(corner_coords[:,0])]
#%%
im_ax = plt.imshow(final_img, extent=xy_lims)
im_ax.axes.get_xaxis().set_visible(False)
im_ax.axes.get_yaxis().set_visible(False)

# A) Plot only the central coordinate and the radius
plt.scatter(coords[1], coords[0], marker='*', s=200,
            facecolor='w', edgecolor='k', linewidths=1, zorder=2)

lat_diff = get_angle_of_dist(coords, r_search_lat, lat_or_long='lat')
lon_diff = get_angle_of_dist(coords, r_search_lon, lat_or_long='lon')
r_search_corners = get_corners(coords, lat_diff, lon_diff)

pol = plt.fill(r_search_corners[:,1], r_search_corners[:,0], 
                facecolor=(0,0,0,.2), edgecolor=(0,0,0,1), linewidth=1, zorder=1)
# separate facecolor from edge color to have different alphas

# plt.savefig('1-input_map_empty.png', dpi=500)
# plt.savefig('2-input_map_coord.png', dpi=500)
# plt.savefig('3-input_map_coord+radius.png', dpi=500)
#%%
# B) PARTITION: Plot all other coordnates with 'black squares' symbolizing cells
im_ax = plt.imshow(final_img, extent=xy_lims)
im_ax.axes.get_xaxis().set_visible(False)
im_ax.axes.get_yaxis().set_visible(False)
plt.scatter(all_coords[:,1], all_coords[:,0], facecolor='r', edgecolor='k', linewidths=0, s=4)


# pol = plt.fill(r_search_corners[:,1], r_search_corners[:,0], 
#                 facecolor=(0,0,0,.2), edgecolor=(0,0,0,1), linewidth=1, zorder=1)
# for c in all_coords:
#     c_corners = get_corners(c, DELTA_LAT/2, DELTA_LON/2)
#     pol = plt.fill(c_corners[:,1], c_corners[:,0], 
#                     facecolor=(0,0,.5,.15), edgecolor=(0,0,0,1), linewidth=1, zorder=1)

# Find distance to the side and to the top. Compute the deltas to create the 
# polygon corners.
plt.savefig('asu_campus.png', dpi=500)
# plt.savefig('4-input_map_partial_coords_on_search_range.png', dpi=500)
# plt.savefig('5-input_map_partial_coords.png', dpi=500)

#%%

# C) Plot one image of the top view model
idx = 2

corner_coords = get_corners(all_coords[idx], DELTA_LAT/2, DELTA_LON/2)

# x is longitude, y is latitude
xy_lims = [min(corner_coords[:,1]), max(corner_coords[:,1]), 
           min(corner_coords[:,0]), max(corner_coords[:,0])]

im_ax = plt.imshow(plt.imread(filenames[idx]))
im_ax.axes.get_xaxis().set_visible(False)
im_ax.axes.get_yaxis().set_visible(False)
plt.savefig('6-top_corner_topview.png', dpi=500)

#%%

        
###################################################################
#%%

# Things to display:
    # Top-view: Image, + classification
    # 

STREET_VIEW_FOV = 90
ANG_OFFSET = 45
N_ORI = int(360/STREET_VIEW_FOV) # number of street view imgs per coordinate
N_CLASSES = len(CLASS_NAMES) - 1 # one is for the background
all_top_view_results = np.zeros((len(all_coords), 4))
all_street_view_results = np.zeros((len(all_coords), N_ORI, N_CLASSES))

IGNORE_SAVED_FILES = False

for coord_idx, coord in enumerate(all_coords):
    print(coord_idx, coord)
    
    ########## TOP VIEW MODEL ###########
    # Top view files
    filenames[coord_idx]
    all_top_view_results[coord_idx] = 0
    ######### STREET VIEW MODEL #########
    
    # Get one street view image for each subcardinal direction (NE, SE, SW, NW)
    # PHASE 1 - download street view images
    street_filenames = []
    for i in range(N_ORI):
        heading_angle = ANG_OFFSET + i*STREET_VIEW_FOV 
        img_name = f'[{coord[0]},{coord[1]}],heading={heading_angle}.jpg'
        street_filename = os.path.join(street_view_folder, img_name)
        params_street = {'lat': coord[0],
                         'long': coord[1],
                         'heading': heading_angle, 
                         'tilt': -5,
                         'idx': i,
                         'path': street_filename,
                         'size': '1000x1000',
                         'fov': STREET_VIEW_FOV}
        street_filenames.append(street_filename)
        
        if not os.path.exists(street_filename):
            dl_street_img(params_street)
        
    # PHASE 2 - run inferences in the images
    for i in range(4):
        fname = street_filenames[i]
        annotaded_output_img_path = \
            fname.replace('.jpg', '_prediction.jpg').replace(street_view_folder, 
                                                             street_view_results_folder)
        output_txt_filename = annotaded_output_img_path.replace('jpg', 'txt')
        
        # plt.imshow(plt.imread(annotaded_output_img_path))
        # if image/results already exists, skip the calculation
        if os.path.exists(output_txt_filename) and not IGNORE_SAVED_FILES:
            # read the written results.
            all_street_view_results[coord_idx, i] = np.loadtxt(output_txt_filename)
            continue
        
        per_class_hits = evaluate_streetview_img(fname, annotaded_output_img_path)
        
        # Save hits in text file
        np.savetxt(output_txt_filename, np.array(per_class_hits))
        
        all_street_view_results[coord_idx, i] = per_class_hits

# np.save(f'all_top_view_results_{time.time()}', all_top_view_results)
# np.save(f'all_street_view_results_{time.time()}', all_street_view_results)
#%% Analyze results

# Rank order top street_view_results

sum_street_view_per_arrow = np.sum(all_street_view_results, axis=2)
sum_street_view_per_coord = np.sum(sum_street_view_per_arrow, axis=1)

plt.hist(sum_street_view_per_coord, bins=50)
plt.grid(which='major')
plt.ylabel('Number of Coordinates')
plt.xlabel('Rating')
plt.savefig('hist.png', dpi=300)
#%%
argsorted = np.flip(np.argsort(sum_street_view_per_coord))

top_results_folder = 'top_results/'

if not os.path.isdir(top_results_folder):
    os.makedirs(top_results_folder)
     

for i in range(100):
    idx = argsorted[i]
    print(f'Index {idx} got {sum_street_view_per_coord[idx]} hits.')
    # img = plt.imread(filenames[idx])
    # aaa = plt.imshow(img)
    # aaa.axes.get_xaxis().set_visible(False)
    # aaa.axes.get_yaxis().set_visible(False)
    # plt.show()
    
    # Load all street view images and append them to one
    for ori_idx in range(N_ORI):
        heading_angle = ANG_OFFSET + ori_idx*STREET_VIEW_FOV 
        img_name = f'[{all_coords[idx,0]},{all_coords[idx,1]}],heading={heading_angle}.jpg'
        img_name = os.path.join(street_view_results_folder,
                                img_name.replace('.jpg', '_prediction.jpg'))
        img = plt.imread(img_name)[20:-20] # TODO:test trim white sides 
        if ori_idx == 0:
            img_concatenated = img
        else:
            img_concatenated = np.concatenate((img_concatenated, img), axis=1)
    
    plt.imshow(img_concatenated)
    plt.title(f'coord: {all_coords[idx]}; # hits = {sum_street_view_per_coord[idx]}')
    plt.tight_layout()
    plt.savefig(os.path.join(top_results_folder, f'top-{i+1}.png'), dpi=300)
    # break

#%% Fill the positives
im_ax = plt.imshow(final_img, extent=xy_lims)
im_ax.axes.get_xaxis().set_visible(False)
im_ax.axes.get_yaxis().set_visible(False)

# CRETERIA: 
# if there are more than X parkour elements in this coordinate, mark as PK spot


plt.scatter(all_coords[:,1], all_coords[:,0], facecolor='w', edgecolor='k', linewidths=0, s=4)

for c_idx, c in enumerate(all_coords):
    c_corners = get_corners(c, DELTA_LAT/2, DELTA_LON/2)
    
    # Plot color according to threshold
    # face_col = (0,0,.5,.15)
    val = sum_street_view_per_coord[c_idx]
    if val > 20:
        face_col = (0,0.6,0,0.2) # green
    else:
        face_col = (0.6,0,0,0.25) # red
    
    # Plot color according to colormap.
    # val = (sum_street_view_per_coord[c_idx]/sum_street_view_per_coord.max()) **2
    # face_col = list(plt.cm.get_cmap('viridis')(val))
    # face_col[-1] = 0.2
    
    
    pol = plt.fill(c_corners[:,1], c_corners[:,0], facecolor=face_col, 
                   edgecolor=(0,0,0,1), linewidth=1, zorder=1)
plt.savefig('ASU_CAMPUS_results_filled.png', dpi=500)
#plt.savefig('7-results.png', dpi=500)

#%% Scatter the positives with a different color

im_ax = plt.imshow(final_img, extent=xy_lims)
im_ax.axes.get_xaxis().set_visible(False)
im_ax.axes.get_yaxis().set_visible(False)

# CRETERIA: 
# if there are more than X parkour elements in this coordinate, mark as PK spot

for c_idx, c in enumerate(all_coords):
    c_corners = get_corners(c, DELTA_LAT/2, DELTA_LON/2)
    
    # Plot color according to threshold
    # face_col = (0,0,.5,.15)
    val = sum_street_view_per_coord[c_idx]
    if val > 20:
# TODO: ....
        plt.scatter(all_coords[:,1], all_coords[:,0], facecolor='k', edgecolor='k', linewidths=0, s=4)

    # Plot color according to threshold
    # face_col = (0,0,.5,.15)
    val = sum_street_view_per_coord[c_idx]
    if val > 20:
        face_col = (0,0.6,0,0.2) # green
    else:
        face_col = (0.6,0,0,0.25) # red
    
    pol = plt.fill(c_corners[:,1], c_corners[:,0], facecolor=face_col, 
                   edgecolor=(0,0,0,1), linewidth=1, zorder=1)
    
plt.savefig('ASU_CAMPUS_results.png', dpi=500)
#plt.savefig('7-results.png', dpi=500)



#%% See if any spot is inside our area of analysis (not area of interest!)

# comm_db_csv = 'community_database.csv'
# comm_db = pd.read_csv(comm_db_csv)
tic = time.time()

for filename in filenames[:2]:
    
    image_path = filename
    img = cv2.imread(image_path)[:-20, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256*2,256*2))
    # cv2.imshow(img)
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    
    # Split in 4
    h,w,c = img.shape
    div = [0, 1/2, 1]
    
    sub_images = []
    for i in range(2):
      for j in range(2):
        im = img[round(h*div[i]):round(h*div[i+1]), round(w*div[j]):round(w*div[j+1])]
        sub_images.append(im)
        
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 2
    resized_images = []
    for i in range(1, columns*rows +1):
        # sub_images[i-1] = cv2.resize(sub_images[i-1], (256,256))
        fig.add_subplot(rows, columns, i)
        plt.imshow(sub_images[i-1])
        plt.xticks([])
        plt.yticks([])
        # resized_images.append(cv2.resize(sub_images[i-1], (256,256)))
        resized_images.append(sub_images[i-1])
    plt.show()
    
    # Normalize
    
    resized_images = np.array(resized_images)
    resized_images = resized_images / 255
    
    
    import tensorflow as tf
    
    model = tf.keras.models.load_model("./topviewMODEL/best_top_view_model_100_rescalede.h5")
    
    preds = model.predict(resized_images)

    pred_binary = [1 if pred > 0.8 else 0 for pred in preds]
    
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 2
    top,bottom,left,right = [5] * 4 # THICKNESS
    color_pos = [255,0,0] # RED
    color_neg = [0,0,0] # BLACK
    
    if 0 in pred_binary:
      to_display = []
      for i in range(4):
        if pred_binary[i] == 1:
          color = color_pos
        else:
          color = color_neg
        bordered = cv2.copyMakeBorder(sub_images[i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        to_display.append(bordered)
      t = cv2.hconcat([to_display[0],to_display[1]]) # top
      b = cv2.hconcat([to_display[2],to_display[3]]) # btm'
      whole = cv2.vconcat([t,b])
      plt.xticks([])
      plt.yticks([])
      plt.imshow(whole)
      
    else:
      bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      plt.imshow(bordered)
      plt.xticks([])
      plt.yticks([])
    plt.show()

print(f'time passed = {time.time() - tic:2.1f}')
#%%

    
import tensorflow as tf 
topview_model_path = "./topviewMODEL/best_top_view_model_100_rescalede.h5"
model_topview = tf.keras.models.load_model(topview_model_path)
tic = time.time()

resized_array = np.zeros((4*len(filenames), 256, 256, 3))
for idx, filename in enumerate(filenames):
    
    image_path = filename
    img = cv2.imread(image_path)[:-20, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256*2,256*2))
    # cv2.imshow(img)
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    
    # Split in 4
    h,w,c = img.shape
    div = [0, 1/2, 1]
    
    sub_images = []
    for i in range(2):
      for j in range(2):
        im = img[round(h*div[i]):round(h*div[i+1]), round(w*div[j]):round(w*div[j+1])]
        sub_images.append(im)
        
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 2
    resized_images = []
    for i in range(1, columns*rows +1):
        # sub_images[i-1] = cv2.resize(sub_images[i-1], (256,256))
        fig.add_subplot(rows, columns, i)
        plt.imshow(sub_images[i-1])
        plt.xticks([])
        plt.yticks([])
        # resized_images.append(cv2.resize(sub_images[i-1], (256,256)))
        resized_images.append(sub_images[i-1])
    plt.show()
    
    # Normalize
    
    resized_images = np.array(resized_images)
    resized_images = resized_images / 255
    
    
    resized_array[idx*4:(idx+1)*4] = resized_images
    
    

#%%
preds_list = model.predict(resized_array)

for preds in preds_list:
    pred_binary = [1 if pred > 0.8 else 0 for pred in preds]
    
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 2
    top,bottom,left,right = [5] * 4 # THICKNESS
    color_pos = [255,0,0] # RED
    color_neg = [0,0,0] # BLACK
    
    if 0 in pred_binary:
      to_display = []
      for i in range(4):
        if pred_binary[i] == 1:
          color = color_pos
        else:
          color = color_neg
        bordered = cv2.copyMakeBorder(sub_images[i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        to_display.append(bordered)
      t = cv2.hconcat([to_display[0],to_display[1]]) # top
      b = cv2.hconcat([to_display[2],to_display[3]]) # btm'
      whole = cv2.vconcat([t,b])
      plt.xticks([])
      plt.yticks([])
      plt.imshow(whole)
      
    else:
      bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      plt.imshow(bordered)
      plt.xticks([])
      plt.yticks([])
    plt.show()
print(f'time passed = {time.time() - tic:2.1f}')