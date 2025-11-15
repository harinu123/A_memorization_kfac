import math
import random
from itertools import cycle, islice
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import weightwatcher as ww
from torch.utils.data import DataLoader, Subset


# ------------------------------
# Configuration
# ------------------------------
EXPERIMENT = "wd_0.01_replicate_iit_scale"
DOWNLOAD_DIRECTORY = "."
TRAIN_POINTS = 1000
OPTIMIZATION_STEPS = 100_000
BATCH_SIZE = 200
LOSS_FUNCTION = "MSE"
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
LEARNING_RATE = 5e-4
INITIALIZATION_SCALE = 4.0
DEPTH = 3
WIDTH = 200
ACTIVATION = "ReLU"
LOG_FREQUENCY = math.ceil(OPTIMIZATION_STEPS / 150)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
SEED = 0
NUM_CLASSES = 10


torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)


ACTIVATION_DICT = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid, "GELU": nn.GELU}
OPTIMIZER_DICT = {"AdamW": optim.AdamW, "Adam": optim.Adam, "SGD": optim.SGD}
LOSS_FUNCTION_DICT = {"CrossEntropy": nn.CrossEntropyLoss, "MSE": nn.MSELoss}


# ------------------------------
# Dataset preparation
# ------------------------------
train_dataset_full = torchvision.datasets.MNIST(
    root=DOWNLOAD_DIRECTORY, train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = torchvision.datasets.MNIST(
    root=DOWNLOAD_DIRECTORY, train=False, transform=transforms.ToTensor(), download=True
)

samples_per_class = TRAIN_POINTS // NUM_CLASSES
indices: List[int] = []

# MNIST targets are torch.Tensor on recent torchvision versions.
targets_np = train_dataset_full.targets.cpu().numpy()
for digit in range(NUM_CLASSES):
    digit_indices = np.where(targets_np == digit)[0]
    chosen_indices = np.random.choice(digit_indices, samples_per_class, replace=False)
    indices.extend(chosen_indices.tolist())

train_dataset = Subset(train_dataset_full, indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

fixed_train_batch = next(iter(train_loader))
fixed_train_inputs = fixed_train_batch[0].to(DEVICE)
fixed_train_labels = fixed_train_batch[1].to(DEVICE)


# ------------------------------
# Model construction helpers
# ------------------------------
def build_mlp(depth: int, width: int, activation: nn.Module) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(784, width))
            layers.append(activation())
        elif i == depth - 1:
            layers.append(nn.Linear(width, NUM_CLASSES))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation())
    model = nn.Sequential(*layers).to(DEVICE)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(INITIALIZATION_SCALE)
    return model


# ------------------------------
# WeightWatcher utilities
# ------------------------------
def get_weightwatcher_df(model: nn.Module) -> pd.DataFrame:
    watcher = ww.WeightWatcher(model=model, log_level="ERROR")
    details = watcher.analyze(
        vectors=True,
        detX=True,
        plot=False,
        fix_fingers="clip_xmax",
        randomize=True,
    )
    return details


# ------------------------------
# Simple Linear-layer KFAC collector
# ------------------------------
class LinearKFACAccumulator:
    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.A: Optional[torch.Tensor] = None
        self.G: Optional[torch.Tensor] = None
        self.n: int = 0
        self._activations: Optional[torch.Tensor] = None
        self._handle_fwd = layer.register_forward_pre_hook(self._forward_hook)
        self._handle_bwd = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        a = inputs[0].detach()
        if a.dim() > 2:
            a = a.reshape(-1, a.size(-1))
        self._activations = a

    def _backward_hook(
        self,
        _: nn.Module,
        __: Tuple[Optional[torch.Tensor], ...],
        grad_outputs: Tuple[Optional[torch.Tensor], ...],
    ) -> None:
        if self._activations is None:
            return
        grad_output = grad_outputs[0]
        if grad_output is None:
            return
        g = grad_output.detach()
        if g.dim() > 2:
            g = g.reshape(-1, g.size(-1))
        a = self._activations
        if self.A is None or self.G is None:
            d_in = a.size(-1)
            d_out = g.size(-1)
            self.A = torch.zeros(d_in, d_in, dtype=a.dtype, device=a.device)
            self.G = torch.zeros(d_out, d_out, dtype=g.dtype, device=g.device)
        self.A.add_(a.T @ a)
        self.G.add_(g.T @ g)
        self.n += a.size(0)
        self._activations = None

    def compute_metrics(self, damping: float = 1e-6) -> Dict[str, float]:
        if self.n == 0 or self.A is None or self.G is None:
            return {}
        A_avg = (self.A / self.n).detach().cpu()
        G_avg = (self.G / self.n).detach().cpu()

        A_eye = torch.eye(A_avg.size(0), dtype=A_avg.dtype)
        G_eye = torch.eye(G_avg.size(0), dtype=G_avg.dtype)

        eig_A = torch.linalg.eigvalsh(A_avg + damping * A_eye)
        eig_G = torch.linalg.eigvalsh(G_avg + damping * G_eye)

        metrics = {
            "A_trace": float(torch.trace(A_avg).item()),
            "G_trace": float(torch.trace(G_avg).item()),
            "A_logdet": float(torch.log(eig_A).sum().item()),
            "G_logdet": float(torch.log(eig_G).sum().item()),
            "A_fro": float(torch.linalg.norm(A_avg, ord="fro").item()),
            "G_fro": float(torch.linalg.norm(G_avg, ord="fro").item()),
            "A_condition": float((eig_A.max() / eig_A.min()).item()),
            "G_condition": float((eig_G.max() / eig_G.min()).item()),
            "samples": float(self.n),
        }
        return metrics

    def close(self) -> None:
        self._handle_fwd.remove()
        self._handle_bwd.remove()
        self._activations = None


def collect_kfac_statistics(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    loss_type: str,
    damping: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    collectors: Dict[str, LinearKFACAccumulator] = {}
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                collector = LinearKFACAccumulator(module)
                collectors[name] = collector
        model.train()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if loss_type == "CrossEntropy":
                loss = loss_fn(outputs, labels)
            elif loss_type == "MSE":
                targets = F.one_hot(labels, num_classes=NUM_CLASSES).to(outputs.dtype)
                loss = loss_fn(outputs, targets)
            else:
                raise ValueError(f"Unsupported loss function: {loss_type}")
            loss.backward()

        kfac_stats = {
            name: collector.compute_metrics(damping=damping) for name, collector in collectors.items()
        }
    finally:
        for collector in collectors.values():
            collector.close()
        model.zero_grad(set_to_none=True)
    return kfac_stats


# ------------------------------
# Metrics helpers
# ------------------------------
def compute_accuracy(
    network: nn.Module, dataset: Subset, device: torch.device, N: Optional[int] = None
) -> float:
    network.eval()
    correct, total = 0, 0
    loader = DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = network(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
            if N is not None and total >= N:
                break
    network.train()
    return correct / max(total, 1)


def compute_loss(
    network: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    loss_type: str,
    device: torch.device,
    N: Optional[int] = None,
) -> float:
    network.eval()
    total_loss = 0.0
    total = 0
    loader = DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = network(inputs)
            if loss_type == "CrossEntropy":
                loss_val = loss_fn(logits, labels)
            elif loss_type == "MSE":
                one_hots = F.one_hot(labels, num_classes=NUM_CLASSES).to(logits.dtype)
                loss_val = loss_fn(logits, one_hots)
            else:
                raise ValueError(f"Unsupported loss function: {loss_type}")
            total_loss += loss_val.item() * inputs.size(0)
            total += inputs.size(0)
            if N is not None and total >= N:
                break
    network.train()
    return total_loss / max(total, 1)


# ------------------------------
# Training setup
# ------------------------------
activation_cls = ACTIVATION_DICT[ACTIVATION]
model = build_mlp(DEPTH, WIDTH, activation_cls)

optimizer_cls = OPTIMIZER_DICT[OPTIMIZER]
optimizer_instance = optimizer_cls(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = LOSS_FUNCTION_DICT[LOSS_FUNCTION]()

torch.save(model.state_dict(), f"model_initial_10m_{EXPERIMENT}.pth")


# ------------------------------
# Logging containers
# ------------------------------
train_losses: List[float] = []
test_losses: List[float] = []
train_accuracies: List[float] = []
test_accuracies: List[float] = []
norms: List[float] = []
last_layer_norms: List[float] = []
log_steps: List[int] = []

ww_alphas: List[Optional[float]] = []
ww_lognorms: List[Optional[float]] = []
ww_spectrals: List[Optional[float]] = []
ww_steps: List[int] = []
ww_dfs: List[pd.DataFrame] = []

kfac_records: List[Dict[str, float]] = []

best_test_acc = -1.0
one_hots = torch.eye(NUM_CLASSES, NUM_CLASSES, device=DEVICE)


# ------------------------------
# Training loop
# ------------------------------
steps = 0
with torch.autograd.set_detect_anomaly(False):
    for inputs, labels in islice(cycle(train_loader), OPTIMIZATION_STEPS):
        if (
            steps < 30
            or (steps < 150 and steps % 10 == 0)
            or steps % LOG_FREQUENCY == 0
        ):
            train_loss = compute_loss(model, train_dataset, loss_fn, LOSS_FUNCTION, DEVICE, N=len(train_dataset))
            train_acc = compute_accuracy(model, train_dataset, DEVICE, N=len(train_dataset))
            test_loss = compute_loss(model, test_dataset, loss_fn, LOSS_FUNCTION, DEVICE, N=len(test_dataset))
            test_acc = compute_accuracy(model, test_dataset, DEVICE, N=len(test_dataset))

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            log_steps.append(steps)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), f"model_best_test_acc_10m_{EXPERIMENT}.pth")

            with torch.no_grad():
                total_norm = sum(torch.pow(param, 2).sum() for param in model.parameters())
                norms.append(float(torch.sqrt(total_norm).item()))
                last_layer = sum(torch.pow(param, 2).sum() for param in model[-1].parameters())
                last_layer_norms.append(float(torch.sqrt(last_layer).item()))

            details_df = get_weightwatcher_df(model)
            details_df["train_accuracy"] = train_acc
            details_df["test_accuracy"] = test_acc
            ww_dfs.append(details_df.copy())

            if "alpha" in details_df.columns:
                valid_alphas = details_df["alpha"].dropna().to_numpy()
                avg_alpha = float(np.mean(valid_alphas)) if len(valid_alphas) > 0 else None
            else:
                avg_alpha = None
            if "lognorm" in details_df.columns:
                valid_lognorms = details_df["lognorm"].dropna().to_numpy()
                avg_lognorm = float(np.mean(valid_lognorms)) if len(valid_lognorms) > 0 else None
            else:
                avg_lognorm = None
            if "spectral_norm" in details_df.columns:
                valid_spectrals = details_df["spectral_norm"].dropna().to_numpy()
                avg_spectral = float(np.mean(valid_spectrals)) if len(valid_spectrals) > 0 else None
            else:
                avg_spectral = None
            ww_alphas.append(avg_alpha)
            ww_lognorms.append(avg_lognorm)
            ww_spectrals.append(avg_spectral)
            ww_steps.append(steps)

            kfac_stats = collect_kfac_statistics(
                model,
                fixed_train_inputs,
                fixed_train_labels,
                loss_fn,
                LOSS_FUNCTION,
            )
            for layer_name, metrics in kfac_stats.items():
                if not metrics:
                    continue
                record = {"step": steps, "layer": layer_name}
                record.update(metrics)
                kfac_records.append(record)

        optimizer_instance.zero_grad(set_to_none=True)
        outputs = model(inputs.to(DEVICE))
        if LOSS_FUNCTION == "CrossEntropy":
            loss_value = loss_fn(outputs, labels.to(DEVICE))
        elif LOSS_FUNCTION == "MSE":
            loss_value = loss_fn(outputs, one_hots[labels])
        else:
            raise ValueError(f"Unsupported loss function: {LOSS_FUNCTION}")
        loss_value.backward()
        optimizer_instance.step()
        steps += 1


torch.save(model.state_dict(), f"model_final_10m_{EXPERIMENT}.pth")


# ------------------------------
# Persist logs
# ------------------------------
full_ww_df = pd.concat(ww_dfs, keys=log_steps, names=["Step", "Row"]).reset_index()
full_ww_df.to_csv(f"weightwatcher_full_history_balanced_10m_{EXPERIMENT}.csv", index=False)

progress_df = pd.DataFrame(
    {
        "Step": log_steps,
        "train_accuracy": train_accuracies,
        "test_accuracy": test_accuracies,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "model_norm": norms,
        "last_layer_norm": last_layer_norms,
        "ww_alpha": ww_alphas,
        "ww_lognorm": ww_lognorms,
        "ww_spectral": ww_spectrals,
    }
)
progress_df.to_csv(f"training_history_10m_{EXPERIMENT}.csv", index=False)

kfac_df = pd.DataFrame(kfac_records)
kfac_df.to_csv(f"kfac_statistics_10m_{EXPERIMENT}.csv", index=False)
