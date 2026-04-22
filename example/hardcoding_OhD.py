from lattice.symmetry.hardcoded_rep import *
from lattice.symmetry.gen_hardcoded_rep import *
from lattice.symmetry.group_generator import *
from lattice.symmetry.utils import multiplicationTable

# edit the generator of fermion representation "generator.Fermion_generator"
# generate the matrix representation of the group OHD with "genMatrixGroupOhD"

# groupOhD = genMatrixGroupOhD(
#     c4y=Fermion_generator["c4y"],
#     c4z=Fermion_generator["c4z"],
#     inv=Fermion_generator["inviden"],
# )
# for key in groupOhD.keys():
#     if (groupOhD[key] - Fermion_rep[key]).norm() > 1e-10:
#         print(key)
#         print(groupOhD[key])
#         print(Fermion_rep[key])
#         print((groupOhD[key] - Fermion_rep[key]).norm())
#         print("--------------------------------")
#         break

# print(groupOhD)

# generate the multiplication table of the group OHD with "multiplicationTable"

# groupOhD = Fermion_rep
# print(groupOhD.keys())
# print(multiplicationTable(groupOhD))

# edit the generators of all little groups "OhD_generator","Dic4_generator","Dic3_generator","Dic2_generator","C4_generator1","C4_generator2" in generator.py
# generate irreps of Oh with "genIrrepOhD"

# OhD_irreps_Dict = {}
# for key in ["A_1", "A_2", "E", "T_1", "T_2", "G_1", "G_2", "H"]:
#     if not key == "G_2":
#         continue
#     print(key)
#     OhD_irreps_Dict[key] = genIrrepOhD(key, is_hardcoded=False)
# print(OhD_irreps_Dict)

# generate irreps of all little groups with "genLittleGroupIrrep"

# Dic4_irreps_dict={}
# for key in ["A_1", "A_2", "E", "B_1", "B_2", "G_1", "G_2"]:
#     Dic4_irreps_dict[key] = genLittleGroupIrrep([0, 0, 1], key,is_hardcoded=False)
# print(Dic4_irreps_dict)

# Dic2_irreps_dict = {}
# for key in ["A_1", "A_2", "B_1", "B_2", "G_1", "G_2"]:
#     Dic2_irreps_dict[key] = genLittleGroupIrrep([0, 1, 1], key, is_hardcoded=False)
# print(Dic2_irreps_dict)

# Dic3_irreps_dict = {}
# for key in ["A_1", "A_2", "F_1", "F_2",'E', "G"]:
#     Dic3_irreps_dict[key] = genLittleGroupIrrep([1, 1, 1], key, is_hardcoded=False)
# for key in ["F_2"]:
#     Dic3_irreps_dict[key] = genLittleGroupIrrep([1, 1, 1], key, is_hardcoded=False)
# print(Dic3_irreps_dict)


# generate the connection between different rows of the same little group irreps.
# result={}
# for key in refRotateDict.keys():
#     if key=="0,1,2" or key=="2,1,1":
#         continue
#     result[key]={}
#     for key2 in irrep_generators[key].keys():
#         result[key][key2]=gen_connection(list(map(int,key.split(","))),key2)
# print(result)


# reduce the irreps of OHD to the irreps of the little group
# little_group_reduction_map_Dic4 = {}
# for key in OD_irreps.keys():
#     # if not key == "T_1":
#     #     continue
#     for parity in [1, -1]:
#         irrep_parity = f"{key}{'g' if parity == 1 else 'u'}"
#         little_group_reduction_map_Dic4[irrep_parity] = {}
#         ndim = OD_irreps[key]["iden"].shape[0]
#         for little_group_key in Dic2_irreps.keys():
#             reduction_matrix = reductionToLittleGroup([0, 1, 1], key, parity, little_group_key, is_hardcoded=False)
#             if reduction_matrix is None:
#                 continue
#             little_group_reduction_map_Dic4[irrep_parity][little_group_key] = reduction_matrix
#             ndim -= len(little_group_reduction_map_Dic4[irrep_parity][little_group_key]) * len(
#                 little_group_reduction_map_Dic4[irrep_parity][little_group_key][0]
#             )
#             if ndim == 0:
#                 break
# print(little_group_reduction_map_Dic4)


pass
