#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from torch_geometric.data import Dataset
from typing import Iterator, List, Optional
import torch
from tqdm import tqdm
import math
from typing import Iterator, Optional, List, Union
from torch.utils.data import Dataset, Sampler
from collections.abc import Mapping, Sequence
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
import math
import warnings
from typing import TypeVar, Optional, Iterator, List
import torch
from torch.utils.data import Dataset, Sampler
T_co = TypeVar('T_co', covariant=True)


class PassNoneCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch_data = Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
            atom2res = []
            atom2frag = []
            pro_cum_nodes = 0  # Accumulated number of nodes
            lig_cum_nodes = 0  # Accumulated number of nodes
            for data in batch:
                atom2res.append(data.atom2res + pro_cum_nodes)  # Update atom2res
                atom2frag.append(data.atom2frag + lig_cum_nodes)  # Update atom2res
                pro_cum_nodes += data['protein'].num_nodes
                lig_cum_nodes += data['frag'].num_nodes
            batch_data.atom2res = torch.cat(atom2res)  # Concatenate all updated atom2res
            batch_data.atom2frag = torch.cat(atom2frag)  # Concatenate all updated atom2res
            return batch_data
        
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class PassNoneDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=PassNoneCollater(follow_batch, exclude_keys),
            **kwargs,
        )


class DynamicBatchSamplerX(torch.utils.data.sampler.Sampler[List[int]]):
    r"""Dynamically adds samples to a mini-batch up to a maximum size (either
    based on number of nodes or number of edges). When data samples have a
    wide range in sizes, specifying a mini-batch size in terms of number of
    samples is not ideal and can cause CUDA OOM errors.

    Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
    ambiguous, depending on the order of the samples. By default the
    :meth:`__len__` will be undefined. This is fine for most cases but
    progress bars will be infinite. Alternatively, :obj:`num_steps` can be
    supplied to cap the number of mini-batches produced by the sampler.

    .. code-block:: python

        from torch_geometric.loader import DataLoader, DynamicBatchSampler

        sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
        loader = DataLoader(dataset, batch_sampler=sampler, ...)

    Args:
        dataset (Dataset): Dataset to sample from.
        max_num (int): Size of mini-batch to aim for in number of nodes or
            edges.
        mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure
            batch size. (default: :obj:`"node"`)
        shuffle (bool, optional): If set to :obj:`True`, will have the data
            reshuffled at every epoch. (default: :obj:`False`)
        skip_too_big (bool, optional): If set to :obj:`True`, skip samples
            which cannot fit in a batch by itself. (default: :obj:`False`)
        num_steps (int, optional): The number of mini-batches to draw for a
            single epoch. If set to :obj:`None`, will iterate through all the
            underlying examples, but :meth:`__len__` will be :obj:`None` since
            it is be ambiguous. (default: :obj:`None`)
    """

    def __init__(self, dataset: Dataset, max_num: int, mode: str = 'node',
                 shuffle: bool = False, skip_too_big: bool = False,
                 num_steps: Optional[int] = None):
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError("`max_num` should be a positive integer value "
                             "(got {max_num}).")
        if mode not in ['node', 'edge']:
            raise ValueError("`mode` choice should be either "
                             f"'node' or 'edge' (got '{mode}').")

        if num_steps is None:
            num_steps = len(dataset)

        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        batch_n = 0
        num_steps = 0
        num_processed = 0

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), dtype=torch.long)
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long)

        while (num_processed < len(self.dataset)
               and num_steps < self.num_steps):
            # Fill batch
            for idx in indices[num_processed:]:
                # Size of sample
                data = self.dataset[idx]
                n = data['ligand'].num_nodes + data['protein'].num_nodes

                if batch_n + n > self.max_num:
                    if batch_n == 0:
                        if self.skip_too_big:
                            continue
                        else:
                            warnings.warn("Size of data sample at index "
                                          f"{idx} is larger than "
                                          f"{self.max_num} {self.mode}s "
                                          f"(got {n} {self.mode}s.")
                    else:
                        # Mini-batch filled
                        break

                # Add sample to current batch
                batch.append(idx.item())
                num_processed += 1
                batch_n += n

            yield batch
            batch = []
            batch_n = 0
            num_steps += 1

    def __len__(self) -> int:
        return self.num_steps


class GraphSizeDistributedSampler(torch.utils.data.sampler.Sampler[List[int]]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, max_nodes_per_batch: int = 100, node_counts: list = [range(100)]) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.node_counts = node_counts
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.init_iter()
    
    def cal_num(self, node_num):
        return node_num * (node_num - 1)

    def _compute_groups(self) -> List[List[int]]:

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        groups = []
        current_group = []
        current_node_count = 0

        for idx in indices:
            if current_node_count + self.cal_num(self.node_counts[idx]) <= self.max_nodes_per_batch:
                current_group.append(idx)
                current_node_count += self.cal_num(self.node_counts[idx])
            else:
                groups.append(current_group)
                current_group = [idx]
                current_node_count = self.cal_num(self.node_counts[idx])

        if current_group:
            groups.append(current_group)

        return groups

    def init_iter(self):
        self.groups = self._compute_groups()
        # type: ignore[arg-type]
        if self.drop_last and len(self.groups) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.groups) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
            totoal_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(
                len(self.groups) / self.num_replicas)  # type: ignore[arg-type]
            total_size = self.num_samples * self.num_replicas
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(self.groups)
            if padding_size <= len(self.groups):
                self.groups += self.groups[:padding_size]
            else:
                self.groups += (self.groups * math.ceil(padding_size /
                                len(self.groups)))[:padding_size]
    
    
    def __iter__(self) -> Iterator[int]:
        self.init_iter()
        groups = self.groups[self.rank::self.num_replicas]
        while len(groups) > 0:
            yield groups.pop()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

class GraphSizeBatchSampler(torch.utils.data.sampler.Sampler[List[int]]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 1) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.init_iter()

    def init_iter(self):
        # Initialize the indices based on the current epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        # Subsampling for distributed training
        self.num_samples_per_replica = int(math.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_replica * self.num_replicas

        # Pad the indices if necessary to ensure the total size is a multiple of the number of replicas
        if len(indices) < self.total_size:
            indices += indices[:(self.total_size - len(indices))]

        # Subsample the indices for this replica
        self.indices = indices[self.rank:self.total_size:self.num_replicas]

    def __iter__(self):
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Handle the last batch if drop_last is not set
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Number of batches
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return math.ceil(len(self.indices) / self.batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.init_iter()

