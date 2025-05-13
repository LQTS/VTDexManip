import os
import pickle

obj_height_file =  '../urdf/obj_info/obj_init_height.pickle'
hand_height_file = '../urdf/obj_info/hand_init_height.pickle'
with open(obj_height_file, "rb") as fh:
    object_init_height = pickle.load(fh)
with open(hand_height_file, "rb") as fh:
    hand_init_height = pickle.load(fh)

# new_obj_init_height = dict()
# new_hand_init_height = dict()
#
# for k, v in object_init_height:
#     if
print('a')