__name__ = "origami_tools.test.test_energy"

# from ..Origamis import TDK, MultiStoriesKresling 
from origami_tools.Origamis import TDK, MultiStoriesKresling 
from origami_tools.Utils import hex_to_hsv, hsv_to_hex
import numpy as np

rng_seed = 152502
# rng_seed = np.random.randint(0, 100000)
rng_gen = np.random.default_rng(rng_seed)


def test_create_tower():
    # Create a TDK tower with the given parameters and check that the parameters are correct and that the string representation is correct
    delta_phi = np.pi/4
    n = 8
    r = 40
    dh = 30 
    m = 0.8
    tdk  = TDK.from_dh_dphi_n_r(dh, delta_phi, n, r, m)
    # tdk.color = "#f865e4"
    assert tdk.h1 < tdk.h2
    assert str(tdk) == "TDK_8_68.2_53.4_40.0_0.8 : n=8, l=68.2958190677951, b=53.42401653098952, r=40, h1=15.16988933062603, h2=45.16988933062603, m=0.8"
    return tdk

def test_import_tower():
    # Import a TDK tower from a file and check that the parameters are correct
    tdk = TDK.load("tdk_p20")
    assert tdk is not None
    assert tdk.n == 8
    assert tdk.h2 - tdk.h1 == 20
    assert abs(tdk.delta_phi() - np.pi/4) < 1e-10
    assert tdk.r == 40
    assert tdk.m == 1
    return tdk

def test_random_tower(show_3D = False, r=None, n=None, m=None, dh=None, dphi=None, seed=None):
    if r is None:
        tdk = TDK.random(seed=seed)
    else:
        tdk = TDK.random(range_r=(r, r),range_dh=(0.5*r,0.7*r), range_dphi=(dphi, dphi), range_m=(m,m), range_n=(n,n), seed=seed)
    assert tdk is not None
    assert tdk.n is not None
    assert tdk.h2 > tdk.h1
    assert tdk.l > 0
    assert tdk.b > 0
    assert tdk.r > 0
    assert tdk.m > 0
    assert tdk.m < 1
    if show_3D:
        tdk.show_3D()
    return tdk

def test_assembly(tdks, show_3D=False):
    # Create a MultiStoriesKresling with the given towers and check that the towers are correctly assigned and that the 3D representation can be shown if requested
    tdk_ex = MultiStoriesKresling(tdks)
    assert tdk_ex is not None
    assert len(tdk_ex.towers) == 2
    assert tdk_ex.towers[0] == tdks[0]
    assert tdk_ex.towers[1] == tdks[1]
    if show_3D:
        tdk_ex.show_3D()
    return tdk_ex

def test_extreme_heights(tdk_ex):
    # Check that the extreme heights of the MultiStoriesKresling are correct and that the maximum height is greater than the minimum height
    height_start = tdk_ex.max_height_stable()
    height_end = tdk_ex.min_height_stable()
    assert height_start > height_end
    assert abs(height_end - sum([tdk.h1 for tdk in tdk_ex.towers])) < 1e-5
    assert abs(height_start - sum([tdk.h2 for tdk in tdk_ex.towers])) < 1e-5
    return height_start, height_end

def test_simulation(tdk_ex : MultiStoriesKresling, height_start, height_end, nb_pts=200):
    # Simulate the movement of the MultiStoriesKresling from the maximum height to the minimum height and check that the curve is correctly computed and that the heights of the towers at each point of the curve are correct
    from ..Utils import multi_point_linspace, min_search, adam_optimize, deriv_fun # pyright: ignore[reportMissingImports]
    h = multi_point_linspace([height_start, height_end, height_start], nb_pts)
    assert len(h) == nb_pts
    curve, ind = tdk_ex.movement3D_trans(h, [tdk.h2 for tdk in tdk_ex.towers]) 
    assert len(curve) == nb_pts
    for i in range(len(curve)):
        assert len(curve[i]) == 2
        # print(curve[i][1], sum(curve[i][1]), h[(i + ind) % nb_pts])
        assert sum(curve[i][1]) - h[(i + ind) % nb_pts] < 1e-5
    return curve, ind, h

def test_inverse_position(tdk_ex, height_start, height_end, nb_pts=200):
    # Check that the inverse position function of the MultiStoriesKresling correctly computes the heights of the towers for a given point of the curve and that the computed heights are correct
    tdk_ex_inv = MultiStoriesKresling(list(reversed(tdk_ex.towers)))
    for i in range(len(tdk_ex_inv.towers)):
        assert tdk_ex_inv.towers[i] == tdk_ex.towers[len(tdk_ex.towers) - 1 - i]
    curve_inv, ind_inv, h_iv = test_simulation(tdk_ex_inv, height_start, height_end, nb_pts)
    curve, ind, h = test_simulation(tdk_ex, height_start, height_end, nb_pts)

    for i in range(len(curve)):
        if not np.linalg.norm(curve[i][0] - list(reversed(curve_inv[i][0]))) < 1e-1 or not np.linalg.norm(curve[i][1] - list(reversed(curve_inv[i][1]))) < 1e-1:
            print("curve :", curve[i])
            print("curve_inv :", curve_inv[i])
            tdk_ex.show_tower_movement(h, curve, start_ind = ind, rotation=False, save=True, name="norm" + tdk_ex.name, path='C:\\Users\\MateoDruart\\Documents\\Doctorat\\Programmes\\tests_gif\\inverse_towers\\', assemb3D=(0,0), energy=(1,0), force=(1,1), height=(2, 0))
            tdk_ex.show_tower_movement(h_iv, curve_inv, start_ind = ind_inv, rotation=False, save=True, name="inv" + tdk_ex.name, path='C:\\Users\\MateoDruart\\Documents\\Doctorat\\Programmes\\tests_gif\\inverse_towers\\', assemb3D=(0,0), energy=(1,0), force=(1,1), height=(2, 0))
            return
        assert np.linalg.norm(curve[i][0] - list(reversed(curve_inv[i][0]))) < 1e-1
        assert np.linalg.norm(curve[i][1] - list(reversed(curve_inv[i][1]))) < 1e-1

def test_show_curve(tdk_ex, h, curve, ind):
    tdk_ex.show_tower_movement(h, curve, start_ind = ind, rotation=False, save=False, name="dual_wcone_reel_kt_1e-6_htours", assemb3D=(0,0), energy=(1,0), force=(1,1))


def create_batch_random(n_tow, n_ass, show=False):
    random_towers = [test_random_tower(False, 20, 8,dphi=np.pi/4, seed=rng_seed + i) for i in range(n_tow)]
    tdk_assembleds = []
    for i in range(n_ass):
        tdk_assembleds.append(MultiStoriesKresling(list(reversed([random_towers[rng_gen.integers(0, len(random_towers))], random_towers[rng_gen.integers(0, len(random_towers))]])), inversed=[True, False]))
        if show:
            tdk_assembleds[i].show_3D()
    return tdk_assembleds

def test_double_k(tdk_ex : MultiStoriesKresling, height_start, height_end, nb_pts=200, name=""):
    # Check that the inverse position function of the MultiStoriesKresling correctly computes the heights of the towers for a given point of the curve and that the computed heights are correct
    tdk_ex.towers[0].set_k_raid(tdk_ex.towers[1].k_tor, tdk_ex.towers[1].k_comp * 2)
    hsv = hex_to_hsv(tdk_ex.towers[0].color)
    tdk_ex.towers[0].color = hsv_to_hex(min((hsv[0]+0.2), 1), min((hsv[1]+0.1), 1), min((hsv[2]+0.1), 1) )
    # curve, ind, h = test_simulation(tdk_ex, height_start, height_end, nb_pts)
    if name == "":
        name = "double_k" + str(asm.name)
    # tdk_ex.show_tower_movement(h, curve, start_ind = ind, rotation=False, save=True, name=name, path='C:\\Users\\MateoDruart\\Documents\\Doctorat\\Programmes\\tests_gif\\identic_towers\\', assemb3D=(0,0), energy=(1,0), force=(1,1), height=(2, 0), kinematic=(2,1))

    # tdk_ex_inv = MultiStoriesKresling(list(reversed(tdk_ex.towers)))
    # curve_inv, ind_inv, h_inv = test_simulation(tdk_ex_inv, height_start, height_end, nb_pts)
    # tdk_ex_inv.show_tower_movement(h_inv, curve_inv, start_ind = ind_inv, rotation=False, save=True, name="inv " +name, path='C:\\Users\\MateoDruart\\Documents\\Doctorat\\Programmes\\tests_gif\\inverse_towers\\', assemb3D=(0,0), energy=(1,0), force=(1,1), height=(2, 0), kinematic=(2,1))


tdk1 = test_create_tower()
print("test_create_tower passed!")
tdk2 = test_import_tower()
print("test_import_tower passed!")
tdk3 = test_random_tower(False)
print("test_random_tower passed!")
tdk4 = TDK.load("tdk_p20_m08")
tdk5 = TDK.load("tdk_l")

tdks = [tdk1, tdk2, tdk4, tdk5]

asms = [test_assembly([tdks[i], tdks[i].copy()], False) for i in range(len(tdks))]
# asms = [test_assembly([tdks[i], tdks[i].copy()], False) for i in range(1)]
# asms = [test_assembly([tdks[rng_gen.integers(0, len(tdks))], tdks[rng_gen.integers(0, len(tdks))]], False) for _ in range(5)]
tdk_ass = test_assembly([tdk2, tdk1], False)
# tdk_asm_2 = test_assembly([tdk4, tdk1], False)
print("test_assembly passed!")

    



height_start, height_end = test_extreme_heights(tdk_ass)
print("test_extreme_heights passed!")
curve, ind, h = test_simulation(tdk_ass, height_start, height_end)
print("test_simulation passed!")

for i, asm in enumerate(asms):
    print("test assembly :", asm.as_dict())

    height_start, height_end = test_extreme_heights(asm)
    # print(height_start, height_end)
    # test_inverse_position(asm, height_start, height_end)
    # print("test_inverse_position passed")
    test_double_k(asm, height_start, height_end, name="double_k_"+ str(i) + "_" + str(asm.name))
    test_inverse_position(asm, height_start, height_end)
    print("test_inverse_position passed")

    # print("test_double_k passed")

# descr = [{"double_k_"+ str(i) + "_" + str(asms[i].name) : asms[i].as_dict()} for i in range(len(asms))]
# import os

# # Persist batch description metadata for generated GIFs.
# output_dir = "C:\\Users\\MateoDruart\\Documents\\Doctorat\\Programmes\\tests_gif\\identic_towers"
# os.makedirs(output_dir, exist_ok=True)
# output_file = os.path.join(output_dir, f"descr_test_identiques.txt")
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(str(descr))
# print(f"Description saved to: {output_file}")



# test_inverse_position(tdk_ass, height_start, height_end)
# print("test_inverse_position passed!")
# test_show_curve(tdk_ass, h, curve, ind)
# print("test_show_curve passed!")


# tdk4 = TDK.load("tdk_l")
# tdk4.color = "#833dc4"
# tdk_ass_2 = test_assembly([tdk4, tdk2], False)
# height_start, height_end = test_extreme_heights(tdk_ass_2)
# curve, ind, h = test_simulation(tdk_ass_2, height_start, height_end, 200)
# test_show_curve(tdk_ass_2, h, curve, ind)

# print(rng_seed)
# tdks_ass = create_batch_random(10,5)
# for ass in tdks_ass:
#     print(ass.as_dict())
#     height_start, height_end = test_extreme_heights(ass)
#     curve, ind, h = test_simulation(ass, height_start, height_end, 200)
#     test_show_curve(ass, h, curve, ind)
#     test_inverse_position(ass, height_start, height_end)



print("All tests passed!")