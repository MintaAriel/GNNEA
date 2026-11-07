"""Build / read the parent-offspring ASE database from an USPEX (python, 2026) results folder.

Layout of the ASE SQLite database (one row per USPEX individual):
    atoms                 relaxed structure from gatheredPOSCARS
    row.energy            USPEX energy (stored through a SinglePointCalculator)
    key_value_pairs       uspex_id, generation, operator, n_parents, offspring_e,
                          p1_id, e1 (if >= 1 parent), p2_id, e2 (if 2 parents), is_best,
                          volume, symmetry, e_above_hull
                          + label / limit once labels.write_labels_to_db has been run
Parents are referenced by their uspex_id, so every structure is stored only once and the
offspring table (p1, p2, e1, e2, offspring_e, operator, generation, p1_id, p2_id) is
re-assembled by `read_offspring_df`.
"""
import os

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect



def build_offspring_db(results_dir, db_path, uspex_ver='2126', overwrite=True):
    # imported here so that reading the db / training does not require the ea package
    from ea.io.uspex_io import indiv_to_df, get_structure_from_id, poscar_to_atoms

    indiv = indiv_to_df(os.path.join(results_dir, 'Individuals'), uspex_ver=uspex_ver)
    best = indiv_to_df(os.path.join(results_dir, 'BESTIndividuals'), uspex_ver=uspex_ver)
    poscars = get_structure_from_id(os.path.join(results_dir, 'gatheredPOSCARS'), [],
                                    all=True, uspex_ver=uspex_ver)

    best_ids = set(best['id'])
    energy = dict(zip(indiv['id'], indiv['energy']))

    if overwrite and os.path.exists(db_path):
        os.remove(db_path)

    with connect(db_path) as db:
        for row in indiv.itertuples(index=False):
            atoms = poscar_to_atoms(poscars[row.id])
            atoms.calc = SinglePointCalculator(atoms, energy=row.energy)

            parents = [int(p) for p in row.parents]
            kv = dict(uspex_id=int(row.id), generation=int(row.generation), operator=row.operator,
                      n_parents=len(parents), offspring_e=float(row.energy),
                      is_best=row.id in best_ids, volume=float(row.volume),
                      symmetry=int(row.symmetry), e_above_hull=float(row.e_above_hull))
            for k, p in enumerate(parents[:2], start=1):
                kv[f'p{k}_id'] = p
                kv[f'e{k}'] = float(energy[p])
            db.write(atoms, **kv)

        db.metadata = {'source': os.path.abspath(results_dir), 'uspex_ver': str(uspex_ver),
                       'n_individuals': len(indiv), 'n_best': len(best_ids)}
    return db_path


def read_individuals_df(db_path):
    """All individuals as a DataFrame (one row per structure) with the ase.Atoms in 'atoms'."""
    rows = []
    for r in connect(db_path).select():
        d = dict(r.key_value_pairs)
        d['natoms'] = r.natoms
        d['atoms'] = r.toatoms()
        rows.append(d)
    return pd.DataFrame(rows).sort_values('uspex_id').reset_index(drop=True)


def read_offspring_df(db_path, operators=None, min_parents=1, structures=True):
    """Re-assemble the offspring table from the ASE database.

    Columns: p1, p2, e1, e2, offspring_e, operator, generation, p1_id, p2_id, offspring_id,
    natoms_p1, natoms_p2, natoms_offspring (+ offspring, label, limit when available).
    p1/p2/offspring are ase.Atoms (None for a missing second parent) if structures=True.
    operators: optional list, e.g. ['Heredity'] to keep only two-parent crossover children.
    """
    ind = read_individuals_df(db_path).set_index('uspex_id')
    child = ind[ind['n_parents'] >= min_parents]
    if operators is not None:
        child = child[child['operator'].isin(operators)]

    def lookup(ids, col):
        return [ind.at[int(i), col] if not pd.isna(i) else None for i in ids]

    df = pd.DataFrame({
        'offspring_id': child.index.values,
        'generation': child['generation'].values,
        'operator': child['operator'].values,
        'p1_id': child['p1_id'].astype('Int64').values,
        'p2_id': child.get('p2_id', pd.Series(np.nan, index=child.index)).astype('Int64').values,
        'e1': child['e1'].values,
        'e2': child.get('e2', pd.Series(np.nan, index=child.index)).values,
        'offspring_e': child['offspring_e'].values,
        'natoms_offspring': child['natoms'].values,
    })
    df['natoms_p1'] = pd.array(lookup(df['p1_id'], 'natoms'), dtype='Int64')
    df['natoms_p2'] = pd.array(lookup(df['p2_id'], 'natoms'), dtype='Int64')
    for col in ('label', 'limit'):
        if col in child:
            df[col] = child[col].values
    if structures:
        df['p1'] = lookup(df['p1_id'], 'atoms')
        df['p2'] = lookup(df['p2_id'], 'atoms')
        df['offspring'] = child['atoms'].values

    first = ['p1', 'p2', 'e1', 'e2', 'offspring_e', 'operator', 'generation', 'p1_id', 'p2_id']
    first = [c for c in first if c in df]
    return df[first + [c for c in df if c not in first]]
