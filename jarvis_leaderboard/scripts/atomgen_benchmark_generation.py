from jarvis.db.jsonutils import loadjson,dumpjson
from jarvis.core.lattice import Lattice
from jarvis.core.atoms import Atoms
import numpy as np
from jarvis.io.vasp.inputs import Poscar
def text2atoms(response):
    print(response)
    tmp_atoms_array = response.split("\n")
    # tmp_atoms_array= [element for element in tmp_atoms_array  if element != '']
    # print("tmp_atoms_array", tmp_atoms_array)
    lat_lengths = np.array(tmp_atoms_array[0].split(), dtype="float")
    lat_angles = np.array(tmp_atoms_array[1].split(), dtype="float")
    lat = Lattice.from_parameters(
        lat_lengths[0],
        lat_lengths[1],
        lat_lengths[2],
        lat_angles[0],
        lat_angles[1],
        lat_angles[2],
    )
    elements = []
    coords = []
    for ii, i in enumerate(tmp_atoms_array):
        if ii > 1 and ii < len(tmp_atoms_array):
            # if ii>2 and ii<len(tmp_atoms_array)-1:
            tmp = i.split()
            elements.append(tmp[0])
            coords.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
    atoms = Atoms(
        coords=coords,
        elements=elements,
        lattice_mat=lat.lattice(),
        cartesian=False,
    )
    return atoms

mem={}
train=loadjson("alpaca_prop_train.json")
info={}
for ii,i in enumerate(train):
    atoms=text2atoms(i['output'])
    jid='na_'+str(ii) #i['id'].split('.vasp')[0]
    info[jid]=Poscar(atoms).to_string().replace("\n","\\n")
mem['train']=info
test=loadjson("alpaca_prop_test.json")
info={}
for i in test:
    atoms=text2atoms(i['output'])
    jid=i['id'].split('.vasp')[0]
    info[jid]=Poscar(atoms).to_string().replace("\n","\\n")
    #print(jid,atoms,info)
    #break
#
mem['test']=info
print(len(mem['train']),len(mem['test']))
dumpjson(data=mem,filename='dft_3d_Tc_supercon.json')
