from jarvis.io.vasp.inputs import Poscar
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
import pandas as pd
from jarvis.core.atoms import Atoms
d=data('dft_3d')

df=pd.read_csv('AI-AtomGen-Tc_supercon-dft_3d-test-rmse.csv')
mem={}
train_mem={}
test_ids=[]

for ii,i in df.iterrows():
  mem[i.id]=i.target
  test_ids.append(i.id)
for i in d:
    if i['jid'] not in test_ids and i['Tc_supercon']!='na':
        train_mem[i['jid']]=Poscar(Atoms.from_dict(i['atoms'])).to_string().replace("\n","\\n")
info={}
info['test']=mem
info['train']=train_mem
print('train',len(train_mem))
dumpjson(data=info,filename='dft_3d_Tc_supercon.json')
