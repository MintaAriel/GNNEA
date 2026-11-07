"""Build / read the parent-offspring ASE database from an USPEX (python, 2026) results folder.

Layout of the ASE SQLite database (one row per USPEX individual):
    atoms                 relaxed structure from gatheredPOSCARS
    row.energy            USPEX energy (stored through a SinglePointCalculator)
    key_value_pairs       uspex_id, generation, operator, n_parents, offspring_e,
                          p1_id, e1 (if >= 1 parent), p2_id, e2 (if 2 parents), is_best,
                          volume, symmetry, e_above_hull
                          + label / limit once labels.write_labels_to_db has been run
Parents are referenced by their uspex_id. Schema 3 stores calculator-free UNRELAXED
child geometry in row.data.offspring_atoms, loaded exclusively from the configured
unrelaxed POSCAR collection. Relaxed row atoms remain the parent inputs; relaxed
energies define the labels. Legacy databases expose no child geometry instead of
silently substituting a relaxed structure.
"""
import os
from pathlib import Path
import tempfile
import hashlib

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect



def build_offspring_db(results_dir, db_path, uspex_ver='2126', overwrite=True, project_name='GNNEA',
                       parent_structures='gatheredPOSCARS', offspring_structures='gatheredPOSCARS_unrelaxed'):
    # imported here so that reading the db / training does not require the ea package
    from ea.io.uspex_io import indiv_to_df, get_structure_from_id, poscar_to_atoms

    indiv = indiv_to_df(os.path.join(results_dir, 'Individuals'), uspex_ver=uspex_ver)
    best = indiv_to_df(os.path.join(results_dir, 'BESTIndividuals'), uspex_ver=uspex_ver)
    parent_path = Path(results_dir) / parent_structures
    child_path = Path(results_dir) / offspring_structures
    if not child_path.is_file():
        raise FileNotFoundError(f'Unrelaxed offspring source is required: {child_path}')
    if parent_path.resolve() == child_path.resolve():
        raise ValueError('Parent and unrelaxed offspring sources must be different files.')
    poscars = get_structure_from_id(str(parent_path), [],
                                    all=True, uspex_ver=uspex_ver)
    unrelaxed = get_structure_from_id(str(child_path), [], all=True, uspex_ver=uspex_ver)
    parented_ids = {int(row.id) for row in indiv.itertuples(index=False) if len(row.parents)}
    missing = parented_ids - unrelaxed.keys()
    if missing:
        raise ValueError(f'{len(missing)} offspring IDs missing from {child_path}: {sorted(missing)[:10]}')

    best_ids = set(best['id'])
    energy = dict(zip(indiv['id'], indiv['energy']))

    db_path = Path(db_path)
    if db_path.exists() and not overwrite:
        raise FileExistsError(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Validate/build before replacing an existing database, including if a POSCAR is corrupt.
    descriptor, temporary_path = tempfile.mkstemp(suffix='.db', prefix='.offspring-', dir=db_path.parent)
    os.close(descriptor)
    try:
        _write_offspring_db(temporary_path, indiv, best_ids, energy, poscars, unrelaxed,
                            parent_path, child_path, results_dir, uspex_ver, project_name, poscar_to_atoms)
        os.replace(temporary_path, db_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return db_path


def _write_offspring_db(db_path, indiv, best_ids, energy, poscars, unrelaxed, parent_path, child_path,
                        results_dir, uspex_ver, project_name, poscar_to_atoms):

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
            payload = {}
            if parents:
                offspring = poscar_to_atoms(unrelaxed[int(row.id)])
                if not np.array_equal(np.sort(offspring.numbers), np.sort(atoms.numbers)):
                    raise ValueError(f'Unrelaxed/relaxed compositions differ for offspring {row.id}')
                if not np.isfinite(offspring.positions).all() or not np.isfinite(offspring.cell.array).all():
                    raise ValueError(f'Nonfinite unrelaxed geometry for offspring {row.id}')
                if abs(np.linalg.det(offspring.cell.array)) < np.finfo(float).eps:
                    raise ValueError(f'Singular unrelaxed cell for offspring {row.id}')
                offspring.calc = None
                payload['offspring_atoms'] = offspring.todict()
                payload['offspring_geometry'] = 'unrelaxed'
            db.write(atoms, data=payload, **kv)

        db.metadata = {'source': os.path.abspath(results_dir), 'uspex_ver': str(uspex_ver),
                       'n_individuals': len(indiv), 'n_best': len(best_ids),
                       'schema_version': 3, 'project': project_name,
                       'parent_geometry': 'relaxed', 'offspring_geometry': 'unrelaxed',
                       'parent_structures_source': str(parent_path.resolve()),
                       'offspring_structures_source': str(child_path.resolve()),
                       'parent_structures_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                       'offspring_structures_sha256': hashlib.sha256(child_path.read_bytes()).hexdigest(),
                       'offspring_atoms_storage': 'row.data.offspring_atoms (ASE Atoms.todict)'}


def read_individuals_df(db_path):
    """All individuals as a DataFrame (one row per structure) with the ase.Atoms in 'atoms'."""
    rows = []
    db = connect(db_path)
    geometry = db.metadata.get('offspring_geometry')
    for r in db.select():
        d = dict(r.key_value_pairs)
        d['natoms'] = r.natoms
        d['atoms'] = r.toatoms()
        payload = r.data
        d['offspring_atoms'] = None
        if geometry == 'unrelaxed' and payload.get('offspring_geometry') == 'unrelaxed':
            d['offspring_atoms'] = Atoms.fromdict(payload['offspring_atoms'])
            d['offspring_atoms'].calc = None
        rows.append(d)
    return pd.DataFrame(rows).sort_values('uspex_id').reset_index(drop=True)


def read_offspring_df(db_path, operators=None, min_parents=1, structures=True):
    """Re-assemble the offspring table from the ASE database.

    Columns: p1, p2, e1, e2, offspring_e, operator, generation, p1_id, p2_id, offspring_id,
    natoms_p1, natoms_p2, natoms_offspring (+ offspring, label, limit when available).
    p1/p2 are relaxed ase.Atoms. offspring is strictly unrelaxed, or None for legacy
    databases lacking unrelaxed input; relaxed row atoms are never substituted.
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
        df['offspring'] = child['offspring_atoms'].values

    first = ['p1', 'p2', 'e1', 'e2', 'offspring_e', 'operator', 'generation', 'p1_id', 'p2_id']
    first = [c for c in first if c in df]
    return df[first + [c for c in df if c not in first]]
