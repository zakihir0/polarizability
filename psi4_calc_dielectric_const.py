#%%
## ライブラリのインポート
import psi4
import numpy as np
import pandas as pd

import time

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.Draw import IPythonConsole

print('rdkit version: ', rdkit.__version__) # rdkit version:  2023.03.1
print('psi4 version: ', psi4.__version__) # psi4 version:  1.8
print('numpy version: ', np.__version__) # numpy version:  1.24.3

#%%
## SMILESをxyz形式に変換
def sm2xyz(smiles: str) -> str:
    """
    SMILESで入力された分子をXYZ形式に変換する
    Args:
        smiles: 分子のSMILES

    Returns:
        str: 分子のXYZ形式

    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer(-1)
    
    xyz = '0 1'
    for atom, (x,y,z) in zip(mol.GetAtoms(), conf.GetPositions()):
        xyz += '\n'
        xyz += f'{atom.GetSymbol()}\t{x}\t{y}\t{z}'
    
    return xyz


## psi4の設定：各自の環境に合わせてください
psi4.set_num_threads(4)
psi4.set_memory('1GB')


#%%
smiles = 'CCCCO'
methane = psi4.geometry(sm2xyz(smiles))
psi4.set_output_file(f'{smiles}.log')
_, wfn = psi4.energy('B3LYP/6-31G(d)',
                     molecule=methane,
                     return_wfn=True)

for key in wfn.variables():
    print(key, '\t', wfn.variable(key))




#%%
import psi4

def calculate_polarizability_volume(molecule_geometry, basis_set='6-31G', method='HF'):
    # ジョブの設定
    psi4.set_options({'basis': basis_set, 'reference': 'rhf'})

    # 分子の設定
    mol = psi4.geometry(molecule_geometry)

    # 計算の実行
    energy, wfn = psi4.energy(method, molecule=mol, return_wfn=True)

    # 分極率テンソルの計算
    polarizability_tensor = psi4.driver.polar_polarizability(wfn)

    # 分子の体積の計算
    molecular_volume = mol.volume()

    # 分極率体積の規格化
    normalized_polarizability = polarizability_tensor / molecular_volume

    # 結果の表示
    print(f"分極率テンソル:\n{polarizability_tensor}")
    print(f"分子の体積: {molecular_volume}")
    print(f"規格化された分極率テンソル:\n{normalized_polarizability}")

# %%
smiles = 'CCCCO'
calculate_polarizability_volume(sm2xyz(smiles))
# %%
import psi4

# PSI4の初期化
psi4.core.set_output_file('output.dat', False)
psi4.core.be_quiet()

# 水分子の座標と基底関数の設定
molecule_geometry = """
O
H 1 0.96
H 1 0.96 2 104.5
"""

# PSI4での計算設定
psi4.set_options({
    'basis': 'cc-pvdz',  # 基底関数
    'scf_type': 'df',    # SCF計算の方法
    'e_convergence': 1e-8,  # 収束条件
})

# 水分子の計算
mol = psi4.geometry(molecule_geometry)
energy, wfn = psi4.optimize('scf', return_wfn=True)

# 最適化された分子の情報取得
molecule = wfn.molecule()

# 最適化された分子の体積の計算
volume = molecule.volume()

# 結果の表示
print(f"水分子の体積: {volume} Å^3")

# %%
import psi4
import numpy as np

# PSI4の初期化
psi4.core.set_output_file('output.dat', False)
psi4.core.be_quiet()

# 水分子の座標と基底関数の設定
molecule_geometry = """
O
H 1 0.96
H 1 0.96 2 104.5
"""

# PSI4での計算設定
psi4.set_options({
    'basis': 'cc-pvdz',  # 基底関数
    'scf_type': 'df',    # SCF計算の方法
    'e_convergence': 1e-8,  # 収束条件
})

# 水分子の計算
mol = psi4.geometry(molecule_geometry)
energy, wfn = psi4.optimize('scf', return_wfn=True)

# 最適化された分子の座標を取得
xyz_coordinates = mol.geometry().np

# 分子の体積を計算
volume = np.linalg.det(xyz_coordinates)

# 結果の表示
print(f"水分子の体積: {volume} Å^3")

# %%
