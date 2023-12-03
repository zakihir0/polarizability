#%%
'''Define calculater'''

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

def calculate_coordinates(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    return mol

def calculate_molecular_weight(mol):
    # 分子の分子量を計算
    molecular_weight = Descriptors.ExactMolWt(mol)

    return molecular_weight

def calculate_molecular_molMR(mol):
    # 分子のモル屈折率を計算
    molecular_molMR = Chem.Crippen.MolMR(mol)

    return molecular_molMR

def calculate_molecular_volume(mol):
    # 分子の体積を計算 (単位: Å^3)
    volume_angstrom3 = AllChem.ComputeMolVolume(mol)

    # Å^3 を cm^3 に変換
    # volume_cm3 = volume_angstrom3 * 1e-24

    return volume_angstrom3


#%%
import math

# MolMR算出
smiles_list = [
    'C', 
    "Cl", 
    "Br", 
    "O", 
    "CCO", 
    "CC(=O)O", 
    'CCCCCCCCCCCCCCCCCCCCCCCCC', 
    'CF', 
    'c1ccccc1', 
    'CCC1(COC1)COCC2(COC2)CC',
    ]

results = []

for smiles in smiles_list:
    
    mol                       = calculate_coordinates(smiles) 
    molecular_mol_MR_compuond = calculate_molecular_molMR(mol)
    molecular_volume_compound = calculate_molecular_volume(mol)
    molecular_weight_compound = calculate_molecular_weight(mol)

    # MR = 4/3 * phi * rho/molwt * N * alpha　のローレンツローレンツの式より算出
    Avogadro_const          = 6.022*10**23
    curve                   = (4/3)*math.pi
    vdw_vol                 = molecular_volume_compound
    molwt                   = molecular_weight_compound
    molvol                  = molwt*Avogadro_const/vdw_vol
    polarizability          = molecular_mol_MR_compuond/(curve*Avogadro_const)*10**25

    result = {
        'smiles': smiles, 
        'vdw_vol':vdw_vol, 
        'molMR':molecular_mol_MR_compuond, 
        'polarizability':polarizability,
        'polarizability/vdwvol':polarizability/vdw_vol, 
        'mol': mol,
        }
    results.append(result)

for result in results:
    print('-'*30)
    for key, val in result.items():
        if key is not 'mol':
            print(key, val)
        else:
            display(val)


#%%
from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_polarizability(mol):
    mol = Chem.AddHs(mol)
    
    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Embed the molecule to generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)

    # Calculate the molecular polarizability
    AllChem.MMFFOptimizeMolecule(mol)
    mol_polarizability = mol.GetProp('_MolPol')

    return float(mol_polarizability)

def calculate_dipole_moment(mol):
    # Calculate the molecular polarizability
    mol_polarizability = calculate_polarizability(mol)

    # Get 3D coordinates and atomic charges
    conf = mol.GetConformer()
    atom_data = [(conf.GetAtomPosition(atom_idx), float(mol.GetAtomWithIdx(atom_idx).GetProp('_GasteigerCharge'))) for atom_idx in range(mol.GetNumAtoms())]

    # Calculate the dipole moment
    dipole_moment = sum(charge * np.array([pt.x, pt.y, pt.z]) for pt, charge in atom_data)

    return dipole_moment

# SMILES string for water (H2O)
smiles_h2o = "O"

# Create the molecule
mol_h2o = Chem.MolFromSmiles(smiles_h2o)

# Calculate the dipole moment for water
dipole_moment_h2o = calculate_dipole_moment(mol_h2o)

print(f"Dipole Moment for H2O: {dipole_moment_h2o}")

#%%
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
import numpy as np

def calculate_gasteiger_charges(mol):
    # Gasteiger電荷を計算
    rdPartialCharges.ComputeGasteigerCharges(mol)

def get_sum_of_atom_vectors(smiles_list):
    results = []
    for idx, smiles in enumerate(smiles_list, 1):
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # 明示的な水素を追加
            mol = Chem.AddHs(mol)

            # 3D座標を計算
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)  # ランダムシードを指定して3D座標を生成

            # Gasteiger電荷を計算
            calculate_gasteiger_charges(mol)

            # 分子の3D座標と電荷を取得
            conf = mol.GetConformer()
            atom_data = [(conf.GetAtomPosition(atom_idx), float(mol.GetAtomWithIdx(atom_idx).GetProp("_GasteigerCharge"))) for atom_idx in range(mol.GetNumAtoms())]

            # 各原子のベクトルを計算
            atom_vectors = np.array([(charge * pt.x, charge * pt.y, charge * pt.z) for (pt, charge) in atom_data])

            # 縦方向に足し合わせる
            sum_of_vectors = np.sum(atom_vectors, axis=0)

            # 結果にSMILESを追加
            result = {'index': idx, 'smiles': smiles, 'vector': sum_of_vectors}
            results.append(result)
        else:
            results.append({'index': idx, 'error': 'Unable to generate molecule from SMILES'})

    return results

# 複数のSMILES文字列をリスト形式で指定
smiles_list = ["O", "CCO", "CC(=O)O", 'CCCCCCCCCCCCCCCCCCCCCCCCC', 'CF']  # 例として水、エタノール、酢酸のSMILES

# 各分子のベクトルを取得し、縦方向に足し合わせる
results = get_sum_of_atom_vectors(smiles_list)

# 結果を表示
for result in results:
    if 'vector' in result:
        print(f"{result['smiles']: <40} {result['vector']} {result['vector'].sum()/3}")
    elif 'error' in result:
        print(result['error'])
