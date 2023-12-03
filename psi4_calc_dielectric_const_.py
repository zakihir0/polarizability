import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit.Chem.Draw import IPythonConsole
import psi4

import datetime
import time


# ハードウェア側の設定（計算に用いるCPUのスレッド数とメモリ設定）
psi4.set_num_threads(nthread=1)
psi4.set_memory("1GB")

# 入力する分子（アセチルサリチル酸）
smiles = 'CC(=O)Oc1ccccc1C(=O)O'

# ファイル名を決める
t = datetime.datetime.fromtimestamp(time.time())
psi4.set_output_file("{}{}{}_{}{}_{}.log".format(t.year,
                                              t.month,
                                              t.day,
                                              t.hour,
                                              t.minute,
                                                smiles))


# SMILES から三次元構造を発生させて、粗3D構造最適化
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
params = ETKDGv3()
params.randomSeed = 1
EmbedMolecule(mol, params)

# MMFF（Merck Molecular Force Field） で構造最適化する
MMFFOptimizeMolecule(mol)
#UFF（Universal Force Field）普遍力場で構造最適化したい場合は
#UFFOptimizeMolecule(mol)

conf = mol.GetConformer()


# Psi4 に入力可能な形式に変換する。
# 電荷とスピン多重度を設定（下は、電荷０、スピン多重度1)
mol_input = "0 1"

#各々の原子の座標をXYZフォーマットで記述
for atom in mol.GetAtoms():
    mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x)\
    + " " + str(conf.GetAtomPosition(atom.GetIdx()).y)\
    + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

molecule = psi4.geometry(mol_input)

# 計算手法（汎関数）、基底関数を設定
level = "b3lyp/6-31G*"

# 計算手法（汎関数）、基底関数の例
#theory = ['hf', 'b3lyp']
#basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

# 構造最適化計算を実行
energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)

'''output
Optimizer: Optimization complete!
CPU times: user 15min 50s, sys: 27.5 s, total: 16min 17s
Wall time: 17min 18s
'''





'''
energy
'''
#単位は原子単位（a.u. もしくはHartrees)

#エネルギー
print(round(energy,3),'a.u.')
# >>> -648.689 a.u.

# HOMO を表示（単位： au = Hartree）
LUMO_idx = wave_function.nalpha()
HOMO_idx = LUMO_idx - 1

homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]
lumo = wave_function.epsilon_a_subset("AO", "ALL").np[LUMO_idx]

print('homo:',round(homo,5),' a.u.')
print('lumo:',round(lumo,5),' a.u.')
# >>> homo: -0.26022 a.u.
# >>> lumo: -0.04309 a.u.



dipole_x, dipole_y, dipole_z = psi4.variable('SCF DIPOLE X'), psi4.variable('SCF DIPOLE Y'), psi4.variable('SCF DIPOLE Z') 
dipole_moment = np.sqrt(dipole_x ** 2 + dipole_y ** 2 + dipole_z ** 2)

#単位はデバイ(D,debye)
print(round(dipole_moment,3),'D')
# >>> 5.175 D
