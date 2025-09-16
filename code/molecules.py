from pennylane import numpy as np

angstrom_to_bohr_unit = 1.8897259886

def get_molecule_N2(d, unit='bohr'):
    # assuming d is in angstrom
    # argument unit means the output unit
    bond_length = d if unit == 'angstrom' else d * angstrom_to_bohr_unit

    symbols = ['N', 'N']
    geometry = np.array([
        [0.0, 0.0, 0],
        [0.0, 0.0, bond_length]
    ])


    electrons = 6
    orbitals = 6 # spatial orbitals
    charge = 0

    return symbols, geometry, electrons, orbitals, charge


def get_molecule_H4(d, unit='bohr'):
    bond_length = d if unit == 'angstrom' else d * angstrom_to_bohr_unit

    symbols = ['H', 'H', 'H', 'H']

    geometry = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.0, bond_length],
                    [0.0, 0.0, bond_length*2.0],
                    [0.0, 0.0, bond_length*3.0]], requires_grad = False)

    electrons = 4
    orbitals = 4
    charge = 0

    return symbols, geometry, electrons, orbitals, charge

def get_molecule_H2(d, unit='bohr'):
    """H2 molecule: 2 hydrogen atoms"""
    bond_length = d if unit == 'angstrom' else d * angstrom_to_bohr_unit

    symbols = ['H', 'H']
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, bond_length]
    ], requires_grad=False)
    return symbols, geometry, 2, 2, 0

def get_molecule_H6(d, unit='bohr'):
    """H6 molecule: linear chain of 6 hydrogen atoms"""
    bond_length = d if unit == 'angstrom' else d * angstrom_to_bohr_unit

    symbols = ['H'] * 6
    geometry = np.array([
        [0.0, 0.0, i * bond_length] for i in range(6)
    ], requires_grad=False)
    return symbols, geometry, 6, 6, 0

def get_molecule_H8(d, unit='bohr'):
    """H8 molecule: linear chain of 8 hydrogen atoms"""
    bond_length = d if unit == 'angstrom' else d * angstrom_to_bohr_unit

    symbols = ['H'] * 8
    geometry = np.array([
        [0.0, 0.0, i * bond_length] for i in range(8)
    ], requires_grad=False)
    return symbols, geometry, 8, 8, 0

def get_molecule_BeH2(d, unit='bohr'):
    """BeH2 molecule: linear geometry H - Be - H"""
    bond_length = d if unit == 'angstrom' else d * angstrom_to_bohr_unit

    symbols = ['H', 'Be', 'H']
    geometry = np.array([
        [0.0, 0.0, -bond_length],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, bond_length]
    ], requires_grad=False)
    return symbols, geometry, 6, 7, 0