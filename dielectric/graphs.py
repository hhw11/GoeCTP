"""Implementation based on the template of ALIGNN."""
from multiprocessing.context import ForkContext
from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes

# from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new ** 2) ** 0.5) * (sum(b_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0

def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec ** 2) ** 0.5) * (sum(c_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag
def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag
    

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        # id_tag="material_id",
    ):
        self.df = df
        self.graphs = graphs
        self.target = target

        # self.ids = self.df[id_tag]
        self.labels = [torch.tensor(itm).type(torch.get_default_dtype()) for itm in self.df[target]]
        self.feat_mask = [torch.tensor(itm).type(torch.get_default_dtype()) for itm in self.df["feature_mask"]]
        self.rot_list = [torch.tensor(itm).type(torch.get_default_dtype()) for itm in self.df["rot_list"]]
        self.equality = [itm for itm in self.df["matrix_equal"]]
        # self.wignerD_num = [torch.Tensor(itm).float().type(torch.get_default_dtype()) for itm in self.df["wigner_D_num"]]
        tmp_labels = torch.stack(self.labels).view(-1, 9)
        print('dataset mean value ', tmp_labels.view(-1).mean())
        diagonal = [0, 4, 8]
        off_diagonal = [1, 2, 3, 5, 6, 7]
        print('diagonal mean, std, max', tmp_labels[:,diagonal].mean(), tmp_labels[:,diagonal].std(), tmp_labels[:,diagonal].max())
        print('off diagonal mean, std, max ', tmp_labels[:,off_diagonal].mean(), tmp_labels[:,off_diagonal].std(), tmp_labels[:,off_diagonal].max())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        g.wignerD_num = None
        label = self.labels[idx]
        mask = self.feat_mask[idx]
        equality = self.equality[idx]
        rot_list = self.rot_list[idx]
        return g, mask, equality, label, rot_list

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        graphs, masks, equalitys, labels, rot_list = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.stack(masks), torch.stack(equalitys), torch.stack(labels), rot_list


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def E3Graph(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    reduce=False,
    equivalent_atoms=None,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return E3Graph(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            reduce=reduce,
            equivalent_atoms=equivalent_atoms,
        )

    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors_now):
        # if reduce:
        #     if equivalent_atoms[site_idx] != site_idx:
        #         continue
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        for dst, image in zip(ids, images):
            src_id, dst_id, _, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))
    return edges




def E3Graph_r(
    atoms=None,
    cutoff=6.0,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)

    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors_now):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        for dst, image in zip(ids, images):
            src_id, dst_id, _, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))
    return edges



def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            use_lattice=use_lattice,
        )
    
    edges = defaultdict(set)
    # lattice correction process
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    
    #print("r_cut=================================")
    #print(r_cut)
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    #print(all_neighbors)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    #print("ids====================================")
    
    #print(ids)
    
    images = np.array([nbr[3] for nbr in neighborlist])
    #print(images)
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1,lat2,lat.matrix) > 90.0:
        lat2 = - lat2
    if angle_from_array(lat1,lat3,lat.matrix) > 90.0:
        lat3 = - lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = - lat1
        lat2 = - lat2
        lat3 = - lat3
        
    # if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
    #     print(lat1, lat2, lat3)
    # lattice correction end
    for site_idx, neighborlist in enumerate(all_neighbors_now):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(lat1))
            edges[(site_idx, site_idx)].add(tuple(lat2))
            edges[(site_idx, site_idx)].add(tuple(lat3))
            
    return edges, lat1, lat2, lat3


def equivalent_decrease(equivalent_atoms):
    # here decrease the number of atoms in the crystal structure according to equivalent_atoms
    new_id_maps = np.ones([len(equivalent_atoms)]) * -1
    nid = 0
    for cur_id in range(len(equivalent_atoms)):
        if equivalent_atoms[cur_id] == cur_id: # map cur_id to new_id and new_id += 1
            new_id_maps[cur_id] = nid
            nid += 1
    new_equivalent_list = [int(new_id_maps[itm]) for itm in equivalent_atoms]
    assert np.min(new_equivalent_list) > -0.5
    return new_equivalent_list


def build_undirected_edgedata_ic(
    atoms=None,
    edges={},
    a=None,
    b=None,
    c=None,
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r, l, nei, angle, atom_lat = [], [], [], [], [], [], []
    v1, v2, v3 = atoms.lattice.cart_coords(a), atoms.lattice.cart_coords(b), atoms.lattice.cart_coords(c)
    # atom_lat.append([v1, v2, v3, -v1, -v2, -v3])
    atom_lat.append([v1, v2, v3])
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                # nei.append([v1, v2, v3, -v1, -v2, -v3])
                nei.append([v1, v2, v3])
                # angle.append([compute_bond_cosine(dd, v1), compute_bond_cosine(dd, v2), compute_bond_cosine(dd, v3)])

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    l = torch.tensor(l).type(torch.int)
    nei = torch.tensor(np.array(nei)).type(torch.get_default_dtype())
    atom_lat = torch.tensor(np.array(atom_lat)).type(torch.get_default_dtype())
    # nei_angles = torch.tensor(angle).type(torch.get_default_dtype())
    return u, v, r, l, nei, atom_lat
    
    
def build_undirected_edgedata(
    atoms=None,
    edges={},
    reduce=False,
    equivalent_atoms=None,
):  
    u = []
    v = []
    r = []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if reduce:
            #     new_equivalent_list = equivalent_decrease(equivalent_atoms)
            #     dst_id = new_equivalent_list[dst_id]
            #     src_id = new_equivalent_list[src_id]
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    return u, v, r

def build_undirected_edgedata_r(
    atoms=None,
    edges={},
    reduce=False,
    equivalent_atoms=None,
):  
    u = []
    v = []
    r = []
    cell_offsets = []
    offsets = []
    pos = []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
            cell_offsets.append(np.array(dst_image))
            cell_offsets.append(-np.array(dst_image))
            offsets.append(atoms.lattice.cart_coords(dst_image))
            offsets.append(-atoms.lattice.cart_coords(dst_image))
            

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    cell_offsets = torch.tensor(cell_offsets)
    offsets = torch.tensor(offsets)
    pos = torch.tensor(atoms.lattice.cart_coords(atoms.frac_coords))
    return u, v, r, cell_offsets, offsets, pos




def atoms2graphs_ic(
    atoms=None,
    cutoff=4.0, 
    max_neighbors=12,
    atom_features="cgcnn",
    id=None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    edges, a, b, c = nearest_neighbor_edges_submit(
                  atoms=atoms,
                  cutoff=cutoff,
                  max_neighbors=max_neighbors,
                  id=id,
                  use_canonize=use_canonize,
                  use_lattice=use_lattice,
                  use_angle=use_angle,
              )
    u, v, r, l, nei, atom_lat = build_undirected_edgedata_ic(atoms, edges, a, b, c)
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features=atom_features))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    g = Data(x=node_features, edge_index=edge_index, edge_attr=r, edge_type=l, edge_nei=nei, atom_lat=atom_lat)
    
    return g

'''
def atoms2graphs(
    atoms=None,
    cutoff=4.0, 
    max_neighbors=12,
    atom_features="cgcnn",
    id=None,
    use_canonize=True,
    reduce=False,
    equivalent_atoms=None,
):
    edges = E3Graph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        id=id,
        use_canonize=use_canonize,
        reduce=reduce,
        equivalent_atoms=equivalent_atoms,
    )
    u, v, r = build_undirected_edgedata(atoms, edges, reduce=reduce, equivalent_atoms=equivalent_atoms)
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features=atom_features))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    g = Data(x=node_features, edge_index=edge_index, edge_attr=r)
    
    return g
'''

def atoms2graphs(
    atoms=None,
    cutoff=4.0, 
    max_neighbors=12,
    atom_features="cgcnn",
    id=None,
    use_canonize=True,
    reduce=False,
    equivalent_atoms=None,
):
    edges = E3Graph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        id=id,
        use_canonize=use_canonize,
        reduce=reduce,
        equivalent_atoms=equivalent_atoms,
    )
    u, v, r = build_undirected_edgedata(atoms, edges, reduce=reduce, equivalent_atoms=equivalent_atoms)
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features=atom_features))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    
    
    atom_pos = torch.tensor(atoms.cart_coords, dtype=torch.float)
    
    g = Data(x=node_features, edge_index=edge_index, edge_attr=r, pos=atom_pos)
    
    
    g.trans_vec = torch.tensor(atoms.lattice.matrix, dtype=torch.float)[None]
    #g.material_id = id
    g.sizes = torch.tensor([atom_pos.shape[0]], dtype=torch.long)
    return g



def atoms2graphs_etgnn(
    atoms=None,
    cutoff=6.0,
    use_canonize=True,
):
    edges = E3Graph_r(
        atoms=atoms,
        cutoff=cutoff,
        use_canonize=use_canonize,
    )
    u, v, r, cell_offsets, offsets, pos = build_undirected_edgedata_r(atoms, edges)
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features="atomic_number"))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features)
    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    g = Data(atomic_numbers=node_features, edge_index=edge_index, edge_attr=r, cell_offsets=cell_offsets, offsets=offsets, pos=pos)
    
    return g


