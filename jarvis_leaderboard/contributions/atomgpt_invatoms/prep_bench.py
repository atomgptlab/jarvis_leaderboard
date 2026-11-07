from jarvis.io.vasp.inputs import Poscar
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
import pandas as pd
from jarvis.core.atoms import Atoms
d=data('alex_supercon')

df=pd.read_csv('AI-AtomGen-Tc-alex_supercon-test-rmse.csv')
mem={}
train_mem={}
test_ids=[]

for ii,i in df.iterrows():
  mem[i.id]=i.target
  test_ids.append(i.id)
for i in d:
    if i['id'] not in test_ids:
        train_mem[i['id']]=Poscar(Atoms.from_dict(i['atoms'])).to_string().replace("\n","\\n")
info={}
info['test']=mem
info['train']=train_mem
dumpjson(data=info,filename='alex_supercon_Tc.json')
