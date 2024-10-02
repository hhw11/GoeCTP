import argparse
import torch
from torch import nn
import numpy as np
from data import get_dataset
import pandas as pd
import pickle as pk
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader
from tqdm import tqdm
#import wandb
from e3nn.io import CartesianTensor
from pandarallel import pandarallel
from data import get_symmetry_dataset
pandarallel.initialize(progress_bar=False)

from graphs import atoms2graphs,atoms2graphs_ic, GraphDataset
from utils import get_id_train_val_test
from ecomformer_new import Ecomformer_new

from ecomformer import iComformer


from e3nn import o3
import pdb
# torch config
torch.set_default_dtype(torch.float32)
import torch
import numpy as np
import random
import os
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Set the random seed for Python, NumPy, and PyTorch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# Ensuring CUDA's determinism
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
    # Configure PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

adptor = JarvisAtomsAdaptor()

diagonal = [0, 4, 8]
off_diagonal = [1, 2, 3, 5, 6, 7]
converter = CartesianTensor("ij")
irreps_output = o3.Irreps('1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o')



def structure_to_graphs(
    df: pd.DataFrame,
    use_corrected_structure: bool = False,
    reduce_cell: bool = False,
    cutoff: float = 4.0,
    max_neighbors: int = 16
):
    def atoms_to_graph(p_input):
        """Convert structure dict to DGLGraph."""
        structure = adptor.get_atoms(p_input["structure"])
        return atoms2graphs(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            reduce=reduce_cell,
            equivalent_atoms=p_input['equivalent_atoms'],
            use_canonize=True,
        )
    graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    # graphs = df["p_input"].apply(atoms_to_graph).values
    return graphs


'''
def structure_to_graphs(
    df: pd.DataFrame,
    use_corrected_structure: bool = False,
    reduce_cell: bool = False,
    cutoff: float = 4.0,
    max_neighbors: int = 16
):
    def atoms_to_graph(p_input):
        """Convert structure dict to DGLGraph."""
        structure = adptor.get_atoms(p_input["structure"])
        return atoms2graphs_ic(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_canonize=True,
            use_lattice = True,
            use_angle = False,
        )
    graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    # graphs = df["p_input"].apply(atoms_to_graph).values
    return graphs
'''




def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total size: {total_params}")
    print("Total trainable parameter number", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total_params



class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) * 
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr 
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def get_pyg_dataset(data, target, reduce_cell=False):
    df_dataset = pd.DataFrame(data)
    g_dataset = structure_to_graphs(df_dataset, reduce_cell=reduce_cell)
    pyg_dataset = GraphDataset(df=df_dataset,graphs=g_dataset, target=target)
    return pyg_dataset



def upper_to_full_symmetric(x):

    matrix = torch.zeros(x.size(0), 3, 3, device=x.device)

    indices = torch.triu_indices(3, 3)


    matrix[:, indices[0], indices[1]] = x


    matrix = matrix + matrix.transpose(1, 2)


    matrix[:, torch.arange(3), torch.arange(3)] /= 2

    return matrix
    
def full_symmetric_to_upper(x):

    indices = torch.triu_indices(3, 3)

    upper = x[:, indices[0], indices[1]]

    return upper

def train(model, args):

    

    
        # load the dataset
    if args.load_preprocessed:
        print("load preprocessed dataset ...")

    dataset_sym = get_dataset(dataset_name=args.target,use_corrected_structure=args.use_corrected_structure,load_preprocessed=args.load_preprocessed)

    # preprocess the dataset and random split
    id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dataset_sym),
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            keep_data_order=False,
        )
        
    np.save('id_train',id_train)
    np.save('id_val',id_val)
    np.save('id_test',id_test)
    
    #id_train = np.load('id_train.npy')
    #id_val = np.load('id_val.npy')
    #id_test = np.load('id_test.npy')
    dataset_train = [dataset_sym[x] for x in id_train]
    

    
    dataset_val = [dataset_sym[x] for x in id_val]
    dataset_test = [dataset_sym[x] for x in id_test]
    
    with open("../preprocessed_%s_dataset_elec_train.pkl"%args.target, 'wb') as f:
        pk.dump(dataset_train, f)
    with open("../preprocessed_%s_dataset_elec_val.pkl"%args.target, 'wb') as f:
        pk.dump(dataset_val, f)    
    with open("../preprocessed_%s_dataset_elec_test.pkl"%args.target, 'wb') as f:
        pk.dump(dataset_test, f)
    
    
    pyg_dataset_train = get_pyg_dataset(dataset_train, args.target, args.reduce_cell)
    pyg_dataset_val = get_pyg_dataset(dataset_val, args.target, args.reduce_cell)
    pyg_dataset_test = get_pyg_dataset(dataset_test, args.target, args.reduce_cell)

    # form dataloaders
    collate_fn = pyg_dataset_train.collate
    train_loader = DataLoader(
        pyg_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        pyg_dataset_val,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        pyg_dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    

    
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    count_parameters(model)
    # set up training configs
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_iter = steps_per_epoch * args.epochs
    scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=args.learning_rate, end_lr=0.00001, power=1)
    from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "huber": nn.HuberLoss(),
    }
    criterion = criteria[args.loss]
    MAE = nn.L1Loss()
 
    # training epoch


    
    
    best_score = 10000
    start_time = time.time()    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}", unit='batch') as pbar:
            for data in train_loader:
                structure, mask, equality, labels, rot_list = data
                

                structure, mask, equality, labels = structure.to(device), mask.to(device), equality.to(device), labels.to(device)
                optimizer.zero_grad()
                
                

                if args.model == "ecomformer_new":

                    outputs = model(structure, mask, equality)

                    #outputs=upper_to_full_symmetric(outputs)
                    labels=full_symmetric_to_upper(labels)

                    loss = criterion(outputs, labels)
                    
                elif args.model == "icomformer":
                    outputs = model(structure)#.view(-1, 3, 3)
                    labels=full_symmetric_to_upper(labels)
                    loss = criterion(outputs, labels)


                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'training_loss': running_loss / (pbar.n + 1)})
                pbar.update(1)
                scheduler.step()

        average_train_loss = running_loss / len(train_loader)
        #wandb.log({"Train Loss": average_train_loss})

        # Validation
        model.eval()
        running_loss = 0.0
        label_list = []
        output_list = []
        
        for data in val_loader:
            structure, mask, _, labels, rot_list = data
            structure, mask, labels = structure.to(device), mask.to(device), labels.to(device)
            if args.model == "ecomformer_new":
                outputs = model(structure, mask, _).detach()
                outputs=upper_to_full_symmetric(outputs)
                
            else:
                outputs = model(structure).detach()
                outputs=upper_to_full_symmetric(outputs)
                if outputs.shape[-1] > 3:
                    outputs = outputs.view(-1, 3, 3)

            output_list.append(outputs.reshape(-1, 9))

            label_list.append(labels.reshape(-1, 9))

        
        outputs = torch.stack(output_list).reshape(-1, 9)
        labels = torch.stack(label_list).reshape(-1, 9)
        mae = abs(outputs - labels).mean(dim=-1).mean()
        
        if mae < best_score and epoch > 100:
            best_score = mae
            torch.save(model.state_dict(), "runs/%s/model_best_%s_%d.pt"%(args.name, args.model, epoch + 1))

        print("Validation mae ", mae)
        #wandb.log({"Validation MAE": mae})
        
    end_time = time.time()
    print("Running times-----------------------------------------")
    print("time-train   :") 
    print(start_time-end_time) 
    torch.save(model.state_dict(), "runs/%s/final_model_test_corrected%s.pt"%(args.name, args.model))

    #wandb.finish()
    
    #test follow
    test(model, args,test_loader,dataset_test)
    
    

    return



def test_augment(dataset, args):

    if args.test_augment == "None":
        return dataset

    return dataset


def test(model, args,test_loader,dataset_test):

    if test_loader is None:
        # load the dataset
        if args.load_preprocessed:
            print("load preprocessed dataset ...")
        dataset_sym = get_dataset(dataset_name=args.target,use_corrected_structure=args.use_corrected_structure,load_preprocessed=args.load_preprocessed)
        count_parameters(model)
        id_train, id_val, id_test = get_id_train_val_test(
                total_size=len(dataset_sym),
                split_seed=args.split_seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                keep_data_order=False,
            )
        dataset_train = [dataset_sym[x] for x in id_train]
        seen_ele=np.zeros([120])
        for itm in dataset_train:
            elems = itm['structure'].atomic_numbers
            for je in range(len(elems)):
                if seen_ele[elems[je]] < 1e-5:
                    seen_ele[elems[je]] = 1.0
        
        unseen_list = []
        for i in range(120):
            if seen_ele[i] < 1e-5:
                unseen_list.append(i)
        print("unseen elements:", unseen_list)
        dataset_test = [dataset_sym[x] for x in id_test]
        dataset_test = test_augment(dataset_test, args)
        
        pyg_dataset_test = get_pyg_dataset(dataset_test, args.target)
    
        # form dataloaders
        collate_fn = pyg_dataset_test.collate
    
        test_loader = DataLoader(
            pyg_dataset_test,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )
        print("n_test:", len(test_loader.dataset))



    # set up training configs
    model.to(device)
    MAE = nn.L1Loss()

    # evaluation and store the model
    model.eval()


    i = 0
    mae_list =[]
    frob_list = []
    percen_list = []
    out_list = []
    error_eT = []

    for data in tqdm(test_loader):
        structure, mask, equality, labels, rot_list = data

        U = np.array(dataset_test[i]["rotation_U"])
        UT=U.T
        U=torch.tensor(U).float()
        UT=torch.tensor(UT).float()
        structure, mask, equality, labels = structure.to(device), mask.to(device), equality.to(device), labels.to(device)

        if args.model == "ecomformer_new":
            outputs = model(structure, mask, equality)
            outputs=upper_to_full_symmetric(outputs)
            
            outputs = outputs.cpu().detach()

            outputs=U@outputs@UT

            
        elif args.model == "icomformer":

            outputs = model(structure)  # 3 * 3

            outputs = outputs.unsqueeze(0)
            outputs = upper_to_full_symmetric(outputs)
            outputs = outputs.cpu().detach()

            outputs = U @ outputs @ UT
            
            

        
        out_list.append(outputs)

        labels = labels.cpu()
        
        labels=U@labels@UT

        mae_list.append(abs(outputs - labels).reshape(-1).mean())

        frob_ = ((labels.reshape(-1) - outputs.reshape(-1)) ** 2).sum() ** 0.5
        frob_norm = (labels.view(-1) ** 2).sum() ** 0.5
        frob_list.append(frob_)
        percen_list.append(frob_/frob_norm)

        i += 1

    print("MAE ", np.mean(mae_list))
    print("M_Frob", np.mean(frob_list))
    percen_list = np.array(percen_list)
    print("EwT 25", np.sum(percen_list < 0.25) / percen_list.shape[0])
    print("EwT 10", np.sum(percen_list < 0.1) / percen_list.shape[0])
    print("EwT 5", np.sum(percen_list < 0.05) / percen_list.shape[0])
    print("EwT 2", np.sum(percen_list < 0.02) / percen_list.shape[0])

    return



def main():
    parser = argparse.ArgumentParser(description='Training script')

    # Define command-line arguments
    # training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training and evaluating')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay')
    parser.add_argument('--loss', type=str, default='huber', help='mse or l1 or huber')
    parser.add_argument('--model', type=str, default='ecomformer_new', help='icomformer or ecomformer_new')
    

    parser.add_argument('--project', type=str, default='test', help='name of project for wandb visualization')
    parser.add_argument('--name', type=str, default='test', help='name of project for storage')
    parser.add_argument('--reduce_cell', type=bool, default=False, help='reduce the cell into irreducible atom sets, not used')
    parser.add_argument('--use_mask', type=bool, default=True, help='symmetry correction module introduced in the paper')
    # dataset parameters
    parser.add_argument('--split_seed', type=int, default=32, help='the random seed of spliting data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training ratio used in data split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='evaluate ratio used in data split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio used in data split')
    parser.add_argument('--target', type=str, default='dielectric', help='dielectric, piezoelectric, or elastic')
    parser.add_argument('--test_augment', type=str, default='None', help='None, XZ_exchange, Xrotate, Yrotate, Zrotate')
    parser.add_argument('--threshold', type=float, default=100., help='threshold to remove samples')
    parser.add_argument('--use_corrected_structure', type=bool, default=False, help='correct input structure or not')
    parser.add_argument('--load_model', type=bool, default=False, help='load pretrained model or not')#False True
    parser.add_argument('--load_preprocessed', type=bool, default=True, help='load previous processed dataset')#False True

    args = parser.parse_args()

    print('Training settings:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Learning rate: {args.learning_rate}')
    print(args)
    torch.manual_seed(args.split_seed)
    torch.cuda.manual_seed_all(args.split_seed)
    # load the model
    if args.model == "ecomformer_new":
        model = Ecomformer_new(args)
    elif args.model == "icomformer":
        model = iComformer(args)        


    if not os.path.exists('runs/' + args.name):
        # Create the directory
        os.makedirs('runs/' + args.name)
        
    if args.load_model:
        if args.model == "ecomformer_new":
            saved_model_path = "../final_model_test_correctedecomformer_new.pt"
            
        elif   args.model == "icomformer":
            saved_model_path = "../final_model_test_correctedicomformer.pt"
        state_dict = torch.load(saved_model_path)
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

    

    train(model, args)
    #test(model, args)

if __name__ == "__main__":
    main()
