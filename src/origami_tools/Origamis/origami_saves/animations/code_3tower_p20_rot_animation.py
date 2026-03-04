from origami_tools.Utils import deg2rad, multi_point_linspace
from origami_tools.Origamis import TDK, MultiStoriesKresling, general_U_rot, general_U_trac


tdk1 = TDK.load("tdk_p20")
# tdk2 = TDK.load("tdk_p20")
# tdk3 = TDK.load("tdk_p20")

# tdk_tower = MultiStoriesKresling([tdk1, tdk2, tdk3], name="3tower_p20")

tdk_tower = MultiStoriesKresling.load("3tower_p20")
k_rot_base = 0.00017779 

tdk_tower.towers[0].set_U_raid(general_U_rot(k_rot_base), general_U_trac(1))
tdk_tower.towers[1].set_U_raid(general_U_rot(k_rot_base), general_U_trac(3))
tdk_tower.towers[2].set_U_raid (general_U_rot(k_rot_base), general_U_trac(9))

path, segment = multi_point_linspace([tdk_tower.min_phi_stable() - deg2rad(10), tdk_tower.max_phi_stable() + deg2rad(10), tdk_tower.towers[0].phi_1 + tdk_tower.towers[1].phi_2 + tdk_tower.towers[2].phi_2 - deg2rad(10), tdk_tower.towers[0].phi_1 + tdk_tower.towers[1].phi_1 + tdk_tower.towers[2].phi_1 - deg2rad(10), tdk_tower.min_phi_stable() - deg2rad(10)], 400, segmented=True)

# Delicate memory structure of origami switches de THeo Jules pour la questions des positions ateignables seulement avec un certain chemin 
tdk_tower.movement3D_rot(path, save=False, segments=segment)

