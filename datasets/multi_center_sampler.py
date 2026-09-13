import numpy as np
from torch.utils.data import Sampler


class MultiCenterBatchSampler(Sampler):
    """
    A custom batch sampler to ensure that each batch contains samples from multiple centers
    whenever possible, which is essential for stabilizing contrastive domain debiasing loss.
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
        balance_classes: bool = False,
        class_balance_power: float = 1.0,
        num_samples_multiplier: float = 1.0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.balance_classes = balance_classes
        self.class_balance_power = float(class_balance_power)
        self.epoch = 0
        self.num_samples = max(
            len(self.dataset),
            int(round(len(self.dataset) * float(num_samples_multiplier))),
        )
        self.center_indices = {}
        self.center_class_indices = {}
        self.class_counts = {}
        for idx, item in enumerate(self.dataset.data_list):
            center = item.get("center", "unknown")
            label = int(item.get("label", 0))
            if center not in self.center_indices:
                self.center_indices[center] = []
                self.center_class_indices[center] = {}
            self.center_indices[center].append(idx)
            self.center_class_indices[center].setdefault(label, []).append(idx)
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
        self.centers = list(self.center_indices.keys())
        self.num_centers = len(self.centers)
        self.class_sampling_weights = {
            label: 1.0 / float(count) ** self.class_balance_power
            for label, count in self.class_counts.items()
            if count > 0
        }

    def _sample_from_center(self, center, rng, center_pools=None):
        if not self.balance_classes:
            pool = center_pools[center] if center_pools is not None else []
            if not pool:
                return None
            return int(pool.pop())
        label_to_indices = self.center_class_indices[center]
        active_labels = [label for label, indices in label_to_indices.items() if indices]
        if not active_labels:
            return None
        label_weights = np.array(
            [self.class_sampling_weights.get(label, 1.0) for label in active_labels],
            dtype=np.float64,
        )
        label_weights = label_weights / label_weights.sum()
        chosen_label = int(rng.choice(active_labels, p=label_weights))
        chosen_indices = label_to_indices[chosen_label]
        return int(chosen_indices[rng.randint(len(chosen_indices))])

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        if self.balance_classes:
            center_pools = None
        else:
            center_pools = {
                center: list(indices) for center, indices in self.center_indices.items()
            }
            for center in center_pools:
                rng.shuffle(center_pools[center])

        total_samples = self.num_samples if self.balance_classes else len(self.dataset)
        num_batches = total_samples // self.batch_size
        if not self.drop_last and total_samples % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            batch = []

            if self.num_centers <= 1 or self.batch_size <= 1:
                while len(batch) < self.batch_size:
                    if self.balance_classes:
                        still_active = list(self.centers)
                        # Explicit null-check for categorical center partitioning pools
                        center_pools_arg = None
                    else:
                        still_active = [c for c in self.centers if len(center_pools[c]) > 0]
                        center_pools_arg = center_pools
                    if not still_active:
                        break
                    chosen_center = rng.choice(still_active)
                    sampled_idx = self._sample_from_center(
                        chosen_center, rng, center_pools=center_pools_arg
                    )
                    if sampled_idx is None:
                        break
                    batch.append(sampled_idx)
            else:
                active_centers = (
                    list(self.centers)
                    if self.balance_classes
                    else [c for c in self.centers if len(center_pools[c]) > 0]
                )
                rng.shuffle(active_centers)

                for center in active_centers:
                    if len(batch) < self.batch_size:
                        sampled_idx = self._sample_from_center(
                            center, rng, center_pools=center_pools
                        )
                        if sampled_idx is not None:
                            batch.append(sampled_idx)
                    else:
                        break

                while len(batch) < self.batch_size:
                    if self.balance_classes:
                        still_active = list(self.centers)
                    else:
                        still_active = [c for c in self.centers if len(center_pools[c]) > 0]
                    if not still_active:
                        break
                    chosen_center = rng.choice(still_active)
                    sampled_idx = self._sample_from_center(
                        chosen_center, rng, center_pools=center_pools
                    )
                    if sampled_idx is None:
                        break
                    batch.append(sampled_idx)

            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                rng.shuffle(batch)
                yield batch
        # self.seed += 1  # Note: Removed in favor of self.epoch for cleaner epoch-tracking

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size
