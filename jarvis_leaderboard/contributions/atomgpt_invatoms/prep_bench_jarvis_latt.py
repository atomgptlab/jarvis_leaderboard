from jarvis.io.vasp.inputs import Poscar
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
import pandas as pd
from jarvis.core.atoms import Atoms
d=data('dft_3d')

df=pd.read_csv('AI-AtomGen-Tc_supercon-dft_3d-test-rmse.csv.zip')
mem={}
test_ids=[]
f_a=open('AI-AtomGen-Tc_supercon_a-dft_3d-test-mae.csv','w')
f_b=open('AI-AtomGen-Tc_supercon_b-dft_3d-test-mae.csv','w')
f_c=open('AI-AtomGen-Tc_supercon_c-dft_3d-test-mae.csv','w')
f_alpha=open('AI-AtomGen-Tc_supercon_alpha-dft_3d-test-mae.csv','w')
f_beta=open('AI-AtomGen-Tc_supercon_beta-dft_3d-test-mae.csv','w')
f_gamma=open('AI-AtomGen-Tc_supercon_gamma-dft_3d-test-mae.csv','w')
f_a.write('id,target,prediction\n')
f_b.write('id,target,prediction\n')
f_c.write('id,target,prediction\n')
f_alpha.write('id,target,prediction\n')
f_beta.write('id,target,prediction\n')
f_gamma.write('id,target,prediction\n')
mem_a={}
mem_b={}
mem_c={}
mem_alpha={}
mem_beta={}
mem_gamma={}

train_alpha={}
train_beta={}
train_gamma={}
train_a={}
train_b={}
train_c={}
for ii,i in df.iterrows():
  jid=i.id
  target_atoms=Poscar.from_string(i.target.replace("\\n","\n")).atoms
  pred_atoms=Poscar.from_string(i.prediction.replace("\\n","\n")).atoms
  test_ids.append(jid)
  line=jid+','+str(target_atoms.lattice.abc[0])+','+str(pred_atoms.lattice.abc[0])+'\n'
  f_a.write(line)
  line=jid+','+str(target_atoms.lattice.abc[1])+','+str(pred_atoms.lattice.abc[1])+'\n'
  f_b.write(line)
  line=jid+','+str(target_atoms.lattice.abc[2])+','+str(pred_atoms.lattice.abc[2])+'\n'
  f_c.write(line)
  line=jid+','+str(target_atoms.lattice.angles[0])+','+str(pred_atoms.lattice.angles[0])+'\n'
  f_alpha.write(line)
  line=jid+','+str(target_atoms.lattice.angles[1])+','+str(pred_atoms.lattice.angles[1])+'\n'
  f_beta.write(line)
  line=jid+','+str(target_atoms.lattice.angles[2])+','+str(pred_atoms.lattice.angles[2])+'\n'
  f_gamma.write(line)
  mem_a[jid]=target_atoms.lattice.abc[0]
  mem_b[jid]=target_atoms.lattice.abc[1]
  mem_c[jid]=target_atoms.lattice.abc[2]
  mem_alpha[jid]=target_atoms.lattice.angles[0]
  mem_beta[jid]=target_atoms.lattice.angles[1]
  mem_gamma[jid]=target_atoms.lattice.angles[2]
  test_ids.append(jid)

f_a.close()
f_b.close()
f_c.close()
f_alpha.close()
f_beta.close()
f_gamma.close()

for i in d:
    if i['jid'] not in test_ids and i['Tc_supercon']!='na':
        train_a[i['jid']]=Atoms.from_dict(i['atoms']).lattice.abc[0]
        train_b[i['jid']]=Atoms.from_dict(i['atoms']).lattice.abc[1]
        train_c[i['jid']]=Atoms.from_dict(i['atoms']).lattice.abc[2]
        train_alpha[i['jid']]=Atoms.from_dict(i['atoms']).lattice.angles[0]
        train_beta[i['jid']]=Atoms.from_dict(i['atoms']).lattice.angles[1]
        train_gamma[i['jid']]=Atoms.from_dict(i['atoms']).lattice.angles[2]
info={}
info['test']=mem_a
info['train']=train_a
print('train',len(train_a))
dumpjson(data=info,filename='dft_3d_Tc_supercon_a.json')
info={}
info['test']=mem_b
info['train']=train_b
print('train',len(train_b))
dumpjson(data=info,filename='dft_3d_Tc_supercon_b.json')
info={}
info['test']=mem_c
info['train']=train_c
print('train',len(train_c))
dumpjson(data=info,filename='dft_3d_Tc_supercon_c.json')
info={}
info['test']=mem_alpha
info['train']=train_alpha
print('train',len(train_alpha))
dumpjson(data=info,filename='dft_3d_Tc_supercon_alpha.json')
info={}
info['test']=mem_beta
info['train']=train_beta
print('train',len(train_beta))
dumpjson(data=info,filename='dft_3d_Tc_supercon_beta.json')
info={}
info['test']=mem_gamma
info['train']=train_gamma
print('train',len(train_gamma))
dumpjson(data=info,filename='dft_3d_Tc_supercon_gamma.json')
