"""Check child persistence, graph batching, parent symmetry and child gradient paths."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ase import Atoms
from ase.db import connect
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from offspring.db import build_offspring_db, read_offspring_df
from offspring.graphs import make_pair_dataset
from offspring.model import PairCGCNN
from offspring.mxtal import make_pair_cluster_dataset, PairMXtal
from src.config import load_config, project_path


class OffspringInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.a = Atoms('OH2', positions=[[1, 1, 1], [1.8, 1, 1], [0.8, 1.8, 1]], cell=[5, 5, 5], pbc=True)
        cls.b = cls.a.copy()
        cls.b.set_cell([5.2, 5, 5])
        cls.c = cls.a.copy()
        extra_molecule = cls.a.copy()
        extra_molecule.translate([2, 2, 2])
        cls.c.extend(extra_molecule)  # child/parent node counts differ, exposing incorrect batch offsets
        cls.c.set_cell([4.8, 5.3, 5])
        cls.df = pd.DataFrame(dict(p1=[cls.a, cls.b], p2=[cls.b, cls.a], offspring=[cls.c, cls.c],
                                   p1_id=[1, 2], p2_id=[2, 1], offspring_id=[3, 4], label=[1, 0]))

    def test_database_round_trip_and_legacy_read(self):
        individuals = pd.DataFrame(dict(id=[1, 2, 3, 4], energy=[-2., -1., -3., -1.5], generation=[1, 1, 2, 2],
            parents=[[], [], [1, 2], [2, 1]], operator=['random', 'random', 'Heredity', 'Heredity'],
            volume=[125.] * 4, symmetry=[1] * 4, e_above_hull=[0.] * 4))
        io = SimpleNamespace(indiv_to_df=lambda *args, **kwargs: individuals,
                             get_structure_from_id=lambda *args, **kwargs: {1: self.a, 2: self.b, 3: self.c, 4: self.c},
                             poscar_to_atoms=lambda a: a.copy())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'ea.io.uspex_io': io}):
            path = Path(tmp) / 'offspring.db'
            build_offspring_db(tmp, path)
            db = connect(path)
            self.assertEqual(db.metadata['offspring_geometry'], 'relaxed')
            self.assertIn('offspring_atoms', db.get(uspex_id=3).data)
            df = read_offspring_df(path, operators=['Heredity'])
            for child in df.offspring:
                np.testing.assert_allclose(child.positions, self.c.positions)
                np.testing.assert_allclose(child.cell, self.c.cell)
                self.assertIsNone(child.calc)
            # Old row-atoms-only databases are still readable.
            legacy = Path(tmp) / 'legacy.db'
            with connect(legacy) as old:
                for row in db.select():
                    old.write(row.toatoms(), **row.key_value_pairs)
            self.assertEqual(len(read_offspring_df(legacy)), 2)

    def check_graphs_and_model(self, encoder, fusion):
        torch.manual_seed(11)
        if encoder == 'cgcnn':
            data = make_pair_dataset(self.df, offspring_structure=True)
            follow = ['numbers_1', 'numbers_2', 'numbers_offspring']
            model = PairCGCNN(node_fea_len=8, edge_fea_len=8, h_fea_len=8, n_conv=1,
                              dropout=0, fusion=fusion, offspring_structure=True)
            prefix, coordinate = 'numbers', 'edge_length_offspring'
        else:
            data, _ = make_pair_cluster_dataset(self.df, offspring_structure=True, verbose=False)
            follow = ['z_1', 'z_2', 'z_offspring']
            model = PairMXtal(h_fea_len=8, dropout=0, fusion=fusion, offspring_structure=True)
            prefix, coordinate = 'z', 'pos_offspring'
        batch = next(iter(DataLoader(data, batch_size=2, follow_batch=follow)))
        for suffix in ('1', '2', 'offspring'):
            edges = batch[f'edge_index_{suffix}']
            nodes_batch = batch[f'{prefix}_{suffix}_batch']
            torch.testing.assert_close(nodes_batch[edges[0]], nodes_batch[edges[1]])
            if encoder == 'mxtal':
                src = batch[f'image_src_{suffix}']
                torch.testing.assert_close(nodes_batch, nodes_batch[src])
        model.eval()
        single = torch.cat([model(next(iter(DataLoader([d], follow_batch=follow)))) for d in data])
        torch.testing.assert_close(model(batch), single, rtol=2e-4, atol=2e-5)
        swapped = batch.clone()
        for key in list(batch.keys()):
            if key.endswith('_1') or key.endswith('_1_batch') or key.endswith('_1_ptr'):
                other = key.replace('_1', '_2')
                swapped[key], swapped[other] = batch[other], batch[key]
        torch.testing.assert_close(model(batch), model(swapped), rtol=2e-4, atol=2e-5)
        batch[coordinate].requires_grad_(True)
        output = model(batch)
        output.sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(float(batch[coordinate].grad.abs().sum()), 0)

    def test_cgcnn_concat_pool(self):
        self.check_graphs_and_model('cgcnn', 'concat_pool')

    def test_cgcnn_pool_concat(self):
        self.check_graphs_and_model('cgcnn', 'pool_concat')

    def test_mxtal_concat_pool(self):
        self.check_graphs_and_model('mxtal', 'concat_pool')

    def test_mxtal_pool_concat(self):
        self.check_graphs_and_model('mxtal', 'pool_concat')

    def test_configuration_paths_and_project(self):
        config = load_config()
        self.assertEqual(config['project']['name'], 'GNNEA')
        self.assertEqual(config['env']['conda_env'], 'brian_ase_env')
        self.assertEqual(config['runtime']['gpu'], 0)
        self.assertTrue(project_path(config['dataset']['db']).is_absolute())


if __name__ == '__main__':
    unittest.main()
