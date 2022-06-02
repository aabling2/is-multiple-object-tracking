import numpy as np
from typing import List

"""camera_offset_from_zero_coordinate = {
cam_id_1 : (x , y ) ,
cam_id_2 : (x , y ) ,
...
cam_id_n : (x , y )
}"""


def find_possibilities_of_same_ids ( features : np . ndarray ,
    confirmed_tracks_indexes : List ,
    cam_ids : List
    ) :
    """
    Defines a cost matrix and checks
    if there are some ids to associate them
    """
    # find cosine similarity
    cs_cost_matrix = calculate_cosine_similarity ( features , features )
    # calculate the central point of each track
    central_points = [
    get_track (i ) . get_central_point () for i in
    confirmed_tracks_indexes
    ]
    # convert camera relative positions to absolute positions
    for i , track_index in enumerate ( confirmed_tracks_indexes ) :
    cam_id = cam_ids [ i]
    x , y = central_points [ i ]
    # find the offset of each camera by cam_id
    offset_x , offset_y = camera_offset_from_zero_coordinate [ cam_id ]
    central_points [ i ] = np . array ([ offset_x + x , offset_y + y ])
    # calculate euclidean distance between absolute positioned points
    distances = np . zeros ( cs_cost_matrix . shape )
    for i , p1 in enumerate ( central_points ):
    for j , p2 in enumerate ( central_points ):
    distances [i , j ] = np . linalg . norm ( p1 - p2 )
    # set those entries that are higher than threshold to 0
    distances [ distances > multi_camera_maximum_distance ] = 0

    # apply min -max normalization to convert distances to (0 to 1)
    scale
    distances =
    ( distances - distances .min () ) /
    ( distances . min () + distances .max () ) ,
    # after normalization distances that are close to each other
    # are close to 0 , while those that are relatively distant are close
    to 1
    # change this it vice - versa
    distances = np . where ( distances == 0 , 0 , 1 - distances )

    # calculate relation to the same camera
    cam_coefficients = np . ones ( cs_cost_matrix . shape )
    for i in cam_ids :
    for j in cam_ids :
    # if tracks have same_cam ids they are definitely different
    # set cam coefficient to zero
    if i == j :
    cam_coefficients [i , j ] = 0

    # compute the probability of setting same ids
    result = cam_coefficients * \
    np . add (
    multi_camera_association_coefficient * cs_cost_matrix ,
    (1 - multi_camera_association_coefficient ) * distances ,
    )
    return result


def replace_ids ( probs : np . ndarray ) :
    # all values that are less than threshold set to 0
    probs [ probs < multi_camera_association_threshold ] = 0
    max_val_indexes = np . argmax ( probs , axis =0)
    replace_ids_of_tracks_that_left ( max_val_indexes )


# start reading here !
def update_framework_step ( self ,
    features : np . ndarray ,
    confirmed_tracks_indexes : List [ int] ,
    cam_ids : List [int ]) :
    """
    : param self : Tracker
    : param features : feature vectors of currently existing tracks
    : param confirmed_tracks_indexes : indexes of confirmed tracks only
    : param cam_ids : cam_ids of tracks
    : return :
    """

    # do usual Sort update

    # try to find similar tracks and assign same ids to them
    if self . frame_index > self . n_init and len( features ) > 0:
    # method below returns probability matrix
    # of two tracks tracking the same person
    probabilities = find_probs_of_same_ids (
    features ,
    targets ,
    confirmed_tracks_indexes ,
    cam_ids
    )
    # replace ids of the
    replace_ids ( probabilities )