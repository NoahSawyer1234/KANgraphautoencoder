"""
fit_missing_values.py

1. Generates N evenly spaced x-values in a given range and evaluates a target
   function f(x) at each of them.
2. Randomly "filters" (masks out) half of these points -- these are the
   missing values the network has never seen during training.
3. Trains a small feed-forward neural network on the visible half only,
   using RMSE as the loss.
4. Periodically snapshots the network's predictions over the FULL x range
   (visible + missing points) so you can later animate/plot how the fit
   evolves over training.

Requires: numpy, torch, matplotlib (matplotlib only needed for the optional
plotting helper at the bottom).
"""

import numpy as np
import torch
import torch.nn as nn

import numpy as np
import scipy.special as special

def manual_airy_ai(x):
    """
    Computes the closed-form Airy Ai(x) function using 
    generalized hypergeometric limit functions (_0F_1).
    """
    # Pre-compute the fractional powers and Gamma constants
    gamma_23 = special.gamma(2/3)
    gamma_13 = special.gamma(1/3)
    
    # Equation terms
    term1 = 1 / (3**(2/3) * gamma_23) * special.hyp0f1(2/3, (x**3) / 9)
    term2 = x / (3**(1/3) * gamma_13) * special.hyp0f1(4/3, (x**3) / 9)
    
    return term1 - term2

def single_coupler(x,a=1,l=0.9):
    rho = l*a**2
    res = l*(1-rho)/(1+rho**2+2*rho*np.cos(x-np.pi/2))
    return res

#----------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------
def generate_data(func, n_points=200, x_range=(-5, 5), mask_fraction=0.5, noise_std=0.0, seed=0):
    """
    Generate evenly spaced x values, evaluate func on them, optionally add
    Gaussian noise to the outputs, and split into visible (training) and
    missing (held-out) sets.

    noise_std: standard deviation of Gaussian noise added to y_all.
               0.0 (default) means no noise -- clean function values.

    Returns a dict with numpy arrays:
        x_all, y_all           -- full evenly spaced grid (noisy if noise_std > 0)
        x_train, y_train       -- visible half
        x_missing, y_missing   -- filtered-out half (ground truth, for eval only)
        train_mask             -- boolean mask into x_all/y_all (True = visible)
    """
    rng = np.random.default_rng(seed)

    x_all = np.linspace(x_range[0], x_range[1], n_points)
    y_all_clean = func(x_all)
    y_all = y_all_clean.copy()

    if noise_std > 0:
        y_all = y_all + rng.normal(loc=0.0, scale=noise_std, size=y_all.shape)

    n_train = int(round(n_points * (1 - mask_fraction)))
    indices = rng.permutation(n_points)
    train_idx = indices[:n_train]
    missing_idx = indices[n_train:]

    train_mask = np.zeros(n_points, dtype=bool)
    train_mask[train_idx] = True

    # Sort so training data isn't in a random order (helps plotting later)
    train_idx = np.sort(train_idx)
    missing_idx = np.sort(missing_idx)

    return {
        "x_all": x_all,
        "y_all": y_all,
        "y_all_clean": y_all_clean,
        "x_train": x_all[train_idx],
        "y_train": y_all[train_idx],
        "x_missing": x_all[missing_idx],
        "y_missing": y_all[missing_idx],
        "train_mask": train_mask,
    }


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def rmse_loss(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


# ---------------------------------------------------------------------------
# 2b. KAN (Kolmogorov-Arnold Network) layer and model variants
# ---------------------------------------------------------------------------
class KAN_layer(nn.Module):
    """
    Fourier-basis KAN layer (batching ignored -- operates on a single
    input vector of shape [input_size] and returns a vector of shape
    [output_size]).
    """

    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super(KAN_layer, self).__init__()
        self.harmonics = num_harmonics
        self.addbias = addbias
        self.in_size = input_size
        self.out_size = output_size
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, output_size, input_size, num_harmonics)
            / (np.sqrt(input_size) * np.sqrt(num_harmonics))
        )
        k = torch.arange(1, num_harmonics + 1).view(1, 1, num_harmonics)
        self.register_buffer("k", k)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        # x: [input_size]
        x_expanded = x.view(1, self.in_size, 1)          # [1, in_size, 1]
        x_scaled = x_expanded * self.k                    # [1, in_size, num_harmonics]
        cos_terms = torch.cos(x_scaled)                    # [1, in_size, num_harmonics]
        sin_terms = torch.sin(x_scaled)                     # [1, in_size, num_harmonics]
        y_cos = torch.einsum("nih,oih->o", cos_terms, self.fouriercoeffs[0])
        y_sin = torch.einsum("nih,oih->o", sin_terms, self.fouriercoeffs[1])
        y = y_cos + y_sin                                    # [out_size]
        if self.addbias:
            y = y + self.bias
        return y


class KAN(nn.Module):
    """
    Stack of KAN_layer modules. Since KAN_layer ignores batching (operates
    on a single vector at a time), this wraps a loop over the batch
    dimension internally so it can still be dropped into the same training
    loop as the MLP.
    """

    def __init__(self, layer_sizes, num_harmonics):
        """
        layer_sizes: list of ints, e.g. [1, 32, 32, 1] describes
                     input -> hidden -> hidden -> output dimensions.
        num_harmonics: number of Fourier harmonics per layer.
        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(KAN_layer(layer_sizes[i], layer_sizes[i + 1], num_harmonics))
        self.layers = nn.ModuleList(layers)

    def forward_single(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        # x: [batch, in_size] -- loop over the batch since KAN_layer
        # itself ignores batching.
        outputs = [self.forward_single(xi) for xi in x]
        return torch.stack(outputs, dim=0)


def make_kan_small():
    """1 layer, 1 harmonic, width 16."""
    return KAN(layer_sizes=[1, 16, 1], num_harmonics=2)


def make_kan_large():
    """3 layers, 2 harmonics, width 32."""
    return KAN(layer_sizes=[1, 32, 32, 32, 1], num_harmonics=3)


# ---------------------------------------------------------------------------
# 3. Training loop with periodic prediction snapshots
# ---------------------------------------------------------------------------
def train_and_snapshot(
    data,
    model=None,
    hidden_dim=64,
    n_layers=3,
    lr=1e-3,
    epochs=3000,
    snapshot_every=50,
    device="cpu",
    seed=0,
    normalize_inputs=True,
):
    """
    model: an already-constructed nn.Module (e.g. MLP(...), make_kan_small(),
           make_kan_large()). If None, defaults to an MLP built from
           hidden_dim/n_layers.
    normalize_inputs: standardize x before feeding to the model. Usually
           helpful for the MLP; for KAN (which uses raw x inside cos/sin)
           you may prefer to set this False if your x_range is already
           small, but by default normalization is harmless and keeps
           the input scale consistent going into either model type.
    """
    torch.manual_seed(seed)

    x_train = torch.tensor(data["x_train"], dtype=torch.float32).view(-1, 1).to(device)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1).to(device)
    x_all = torch.tensor(data["x_all"], dtype=torch.float32).view(-1, 1).to(device)
    y_missing = torch.tensor(data["y_missing"], dtype=torch.float32).view(-1, 1).to(device)
    missing_mask = torch.tensor(~data["train_mask"]).to(device)

    # Simple input normalization helps training stability
    x_mean, x_std = x_train.mean(), x_train.std()

    def normalize(x):
        return (x - x_mean) / x_std if normalize_inputs else x

    if model is None:
        model = MLP(hidden_dim=hidden_dim, n_layers=n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_rmse": [],
        "missing_rmse": [],  # how well we recover the held-out points (diagnostic only)
        "predictions": [],   # snapshot of predictions over x_all at this epoch
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_train = model(normalize(x_train))
        loss = rmse_loss(pred_train, y_train)
        loss.backward()
        optimizer.step()

        if epoch % snapshot_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred_all = model(normalize(x_all))
                missing_rmse = rmse_loss(pred_all[missing_mask], y_missing).item()

            history["epoch"].append(epoch)
            history["train_rmse"].append(loss.item())
            history["missing_rmse"].append(missing_rmse)
            history["predictions"].append(pred_all.cpu().numpy().flatten())

            print(
                f"Epoch {epoch:5d} | train RMSE: {loss.item():.5f} | "
                f"missing RMSE: {missing_rmse:.5f}"
            )

    return model, history


# ---------------------------------------------------------------------------
# 4. Optional: visualize the fitting process
# ---------------------------------------------------------------------------
def plot_history(data, history, save_path=None, show=True):
    import matplotlib.pyplot as plt

    x_all = data["x_all"]
    n_snaps = len(history["epoch"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["x_train"], data["y_train"], color="tab:blue", s=20, label="Visible (train, noisy)")
    ax.plot(data["x_all"], data["y_all_clean"], color="tab:red", linewidth=1.5,
            linestyle="--", label="True function (noiseless)")

    cmap = plt.cm.viridis
    for i, epoch in enumerate(history["epoch"]):
        color = cmap(i / max(n_snaps - 1, 1))
        alpha = 0.2 + 0.8 * (i / max(n_snaps - 1, 1))
        ax.plot(x_all, history["predictions"][i], color=color, alpha=alpha,
                linewidth=1, label=f"epoch {epoch}" if i == n_snaps - 1 else None)

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Network predictions over training")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def make_animation(data, history, save_path="fit_animation.gif", fps=15):
    """Optional: build a GIF animation of the fitting process."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x_all = data["x_all"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["x_train"], data["y_train"], color="tab:blue", s=20, label="Visible (train, noisy)")
    ax.plot(data["x_all"], data["y_all_clean"], color="tab:red", linewidth=1.5,
            linestyle="--", label="True function (noiseless)")
    line, = ax.plot([], [], color="black", linewidth=2, label="Prediction")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")

    y_all_vals = np.concatenate(history["predictions"] + [data["y_all"]])
    ax.set_ylim(y_all_vals.min() - 0.5, y_all_vals.max() + 0.5)

    def update(frame):
        line.set_data(x_all, history["predictions"][frame])
        title.set_text(f"Epoch {history['epoch'][frame]}")
        return line, title

    anim = animation.FuncAnimation(fig, update, frames=len(history["epoch"]), blit=True)
    anim.save(save_path, fps=fps)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cpu"
    print(f"Using device: {device}")

    # Define the target function here
    target_func = manual_airy_ai
    #target_func = single_coupler

    data = generate_data(
        func=target_func,
        n_points=200,
        x_range=(-10, 10),
        mask_fraction=0.5,
        noise_std=0.0,
        seed=42,
    )

    # --- Model configurations to compare ---
    # 1. Baseline MLP
    # 2. KAN: 1 layer, 1 harmonic, width 16
    # 3. KAN: 3 layers, 2 harmonics, width 32
    model_configs = {
        "mlp": lambda: MLP(hidden_dim=64, n_layers=3),
        "kan_small": make_kan_small,   # 1 layer, 1 harmonic, width 16
        "kan_large": make_kan_large,   # 3 layers, 2 harmonics, width 32
    }

    results = {}
    for name, build_model in model_configs.items():
        print(f"\n=== Training {name} ===")
        model, history = train_and_snapshot(
            data,
            model=build_model(),
            lr=1e-3,
            epochs=1000,
            snapshot_every=100,
            device=device,
            seed=42,
        )
        results[name] = (model, history)

        # Plot a series of model output curves, one per snapshot (every 100 epochs)
        plot_history(data, history, save_path=f"fit_progress_{name}.png", show=False)

        # Uncomment to also produce an animated GIF (requires pillow or ffmpeg)
        # make_animation(data, history, save_path=f"fit_animation_{name}.gif")

    print("\nDone. Saved static progress plots: " +
          ", ".join(f"fit_progress_{name}.png" for name in model_configs))