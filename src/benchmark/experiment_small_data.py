import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
import wandb
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from src.benchmark.consts import SCENE_DICT, PU_CLASSES
from src.benchmark.pred_ml import predict_soil_classes
from src.benchmark.utils import get_confidence_interval
from src.config import ExperimentConfig
from src.data.dataset import HyperspectralScene
from src.eval.eval_loop import Evaluator
from src.experiment import Experiment
from src.models.modeller import Modeller
from src.train.train_loop import train


plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update(
    {
        "font.family": ["serif"],
        "font.serif": "Times New Roman",
    }
)
# plt.rcParams.update(
#     {
#         "figure.figsize": (3.3, 2.5),
#         "figure.dpi": 600,
#         "font.size": 21,
#         "mathtext.fontset": "cm",
#         "axes.formatter.use_mathtext": True,
#         # "axes.unicode_minus": True,
#         "font.family": ["serif"],
#         "font.serif": "Times New Roman",
#         # X axis
#         "xtick.direction": "in",
#         "xtick.major.size": 6,
#         "xtick.major.width": 1.0,
#         "xtick.minor.size": 3.0,
#         "xtick.minor.width": 1.0,
#         "xtick.minor.visible": True,
#         "xtick.top": True,
#         # Y axis
#         "ytick.direction": "in",
#         "ytick.major.size": 6,
#         "ytick.major.width": 1.0,
#         "ytick.minor.size": 3.0,
#         "ytick.minor.width": 1.0,
#         "ytick.minor.visible": True,
#         "ytick.right": True,
#     }
# )


def plot_clusters_predictions_gt(features, preds, gt_labels):
    """
    Perform PCA and KMeans clustering on features, and plot 2D visualization
    side by side: colored by predictions, clusters, and ground truth.

    Args:
        features (np.ndarray): Feature matrix (num_samples, num_features)
        preds (np.ndarray): Predicted class labels (num_samples,)
        gt_labels (np.ndarray): Ground truth labels (num_samples,)
        n_clusters (int, optional): Number of clusters for KMeans. Defaults to number of unique preds.
    """
    if n_clusters is None:
        n_clusters = len(set(preds))

    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # 2. Reduce features to 2D with PCA
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)

    # 3. Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(21, 6))
    fig.suptitle("Raw")

    # Predicted classes plot
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], c=preds, cmap="tab10", alpha=0.7)
    axes[0].set_title("Predicted Classes")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")
    fig.colorbar(scatter1, ax=axes[0], label="Class")

    # Ground truth plot
    scatter3 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], c=gt_labels, cmap="tab10", alpha=0.7)
    axes[1].set_title("Ground Truth Classes")
    axes[1].set_xlabel("PCA Component 1")
    axes[1].set_ylabel("PCA Component 2")
    fig.colorbar(scatter3, ax=axes[1], label="Ground Truth")

    plt.tight_layout()
    plt.savefig("clustering_raw.png")


def plot_clusters_predictions_hatsx(features, gt_labels):
    fig, axes = plt.subplots(1, 5, figsize=(21, 6))
    fig.suptitle("PCA")

    for i in range(5):
        hat_f = features[:, i, :]
        # scaler = StandardScaler()
        # hat_f = scaler.fit_transform(hat_f)
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(hat_f)

        scatter1 = axes[i].scatter(
            features_2d[:, 0], features_2d[:, 1], c=gt_labels, cmap="tab10", alpha=0.4, linewidths=0
        )
        axes[i].set_title(f"Hat {i}")
        axes[i].set_xlabel("PCA Component 1")
        axes[i].set_ylabel("PCA Component 2")
    fig.colorbar(scatter1, ax=axes[i], label="Class")

    plt.tight_layout()
    plt.savefig("proj_hats.png")


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap  # pip install umap-learn


def plot_clusters_predictions_hats(features, gt_labels):
    """
    Plot PCA, UMAP, and t-SNE projections for each "hat" (features[:, i, :]).

    Args:
        features (np.ndarray): Feature matrix with shape (n_samples, n_hats, n_features)
        gt_labels (np.ndarray): Ground truth labels for coloring (n_samples,)
    """
    n_hats = features.shape[1]
    fig, axes = plt.subplots(3, n_hats, figsize=(5 * n_hats, 15))
    fig.suptitle("Feature Projections (PCA, UMAP, t-SNE)", fontsize=16)

    for i in range(n_hats):
        hat_f = features[:, i, :]

        print("pca")

        # --- PCA ---
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(hat_f)
        scatter_pca = axes[0, i].scatter(
            features_pca[:, 0],
            features_pca[:, 1],
            c=gt_labels,
            cmap="tab10",
            alpha=0.4,
            linewidths=0,
        )
        axes[0, i].set_title(f"PCA - Hat {i}")
        axes[0, i].set_xlabel("PC1")
        axes[0, i].set_ylabel("PC2")

        print("umap")

        # --- UMAP ---
        reducer = umap.UMAP(n_components=2, random_state=42)
        features_umap = reducer.fit_transform(hat_f)
        axes[1, i].scatter(
            features_umap[:, 0],
            features_umap[:, 1],
            c=gt_labels,
            cmap="tab10",
            alpha=0.4,
            linewidths=0,
        )
        axes[1, i].set_title(f"UMAP - Hat {i}")
        axes[1, i].set_xlabel("UMAP1")
        axes[1, i].set_ylabel("UMAP2")

        print("tsne")

        # --- t-SNE ---
        tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        features_tsne = tsne.fit_transform(hat_f)
        axes[2, i].scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            c=gt_labels,
            cmap="tab10",
            alpha=0.4,
            linewidths=0,
        )
        axes[2, i].set_title(f"t-SNE - Hat {i}")
        axes[2, i].set_xlabel("t-SNE1")
        axes[2, i].set_ylabel("t-SNE2")

    # Add colorbar once (shared across all subplots)
    # fig.colorbar(scatter_pca, ax=axes, label="Class", shrink=0.6)

    plt.tight_layout()
    plt.savefig("proj_hats.png", dpi=200)
    plt.show()


def plot_clusters_predictions_scale(features, scales, gt_labels):
    """
    Plot PCA, UMAP, and t-SNE projections for each "hat" (features[:, i, :]).

    Args:
        features (np.ndarray): Feature matrix with shape (n_samples, n_hats, n_features)
        gt_labels (np.ndarray): Ground truth labels for coloring (n_samples,)
    """
    n_hats = 2
    fig, axes = plt.subplots(3, n_hats, figsize=(5 * n_hats, 15))
    fig.suptitle("Feature Projections (PCA, UMAP, t-SNE)", fontsize=16)
    labels = ["All features", "Scale features only"]
    feature_list = [features, scales]

    cmap = cmc.batlowS  # batlowS colormap
    # colors = [cmap(i) for i in range(len(PU_CLASSES))]

    for i in range(n_hats):
        hat_f = feature_list[i]

        print("pca")

        # --- PCA ---
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(hat_f)
        scatter_pca = axes[0, i].scatter(
            features_pca[:, 0],
            features_pca[:, 1],
            c=gt_labels,
            cmap=cmap,
            alpha=0.4,
            linewidths=0,
        )
        axes[0, i].set_title(f"PCA - {labels[i]}")
        axes[0, i].set_xlabel("PC1")
        axes[0, i].set_ylabel("PC2")

        print("umap")

        # --- UMAP ---
        reducer = umap.UMAP(n_components=2, random_state=42)
        features_umap = reducer.fit_transform(hat_f)
        axes[1, i].scatter(
            features_umap[:, 0],
            features_umap[:, 1],
            c=gt_labels,
            cmap=cmap,
            alpha=0.4,
            linewidths=0,
        )
        axes[1, i].set_title(f"UMAP - Hat {labels[i]}")
        axes[1, i].set_xlabel("UMAP1")
        axes[1, i].set_ylabel("UMAP2")

        print("tsne")

        # --- t-SNE ---
        tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        features_tsne = tsne.fit_transform(hat_f)
        axes[2, i].scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            c=gt_labels,
            cmap=cmap,
            alpha=0.4,
            linewidths=0,
        )
        axes[2, i].set_title(f"t-SNE - Hat {labels[i]}")
        axes[2, i].set_xlabel("t-SNE1")
        axes[2, i].set_ylabel("t-SNE2")

    # Add colorbar once (shared across all subplots)
    cbar = fig.colorbar(scatter_pca, ax=axes, label="Class", shrink=0.6)
    cbar.ax.set_yticklabels(PU_CLASSES)

    plt.tight_layout()
    plt.savefig("proj_scale_full.png", dpi=200)
    plt.show()


def plot_tsne(features, gt_labels):
    fig, axes = plt.subplots(1, 1)
    batlow = cmc.batlowS
    colors = batlow(np.linspace(0, 1, len(PU_CLASSES)))
    cmap = ListedColormap(colors)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    features_tsne = tsne.fit_transform(features)
    scatter = axes.scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=gt_labels,
        cmap=cmap,
        s=4,
        alpha=0.4,
        linewidths=0,
    )
    axes.set_yticklabels([])
    axes.set_xticklabels([])

    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(-0.5, len(PU_CLASSES), 1), ncolors=len(PU_CLASSES))
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(len(PU_CLASSES)), ax=axes)
    cbar.ax.set_yticklabels(PU_CLASSES)

    plt.savefig("tsne.pdf")
    plt.show()


class SmallDataExperiment(Experiment):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)

    # repeat 10 times, 10 different splits, test fixed among sample number

    def run(self) -> None:
        # repeat = 10
        # samples = [0.5, 0.1, 0.05, 0.01, 0.005]
        repeat = 1
        samples = [0.5]
        data_cfg = SCENE_DICT[self.cfg.dataset_name]
        self.cfg.channels = data_cfg.channels
        dataset = HyperspectralScene(data_cfg)
        fulloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        n = len(dataset)
        raw = False

        model = self._setup_autoencoder() if self.ae else self._setup_bias_variance_model()
        if not raw:
            if not self.cfg.save_model:
                modeller = Modeller(self.cfg.channels, self.cfg.k, self.num_params)
                modeller.load_state_dict(torch.load(self.cfg.modeller_path))
                modeller = modeller.to(self.cfg.device)
            else:
                model = train(model, fulloader, None, self.cfg)
                modeller = model.encoder if self.ae else model.modeller
                torch.save(modeller.state_dict(), self.cfg.modeller_path)
                print("Evaluating")
                bias = dataset.mean / dataset.std
                bias = bias.unsqueeze(1).unsqueeze(1).numpy()
                Evaluator(model, self.cfg, self.ae, bias).evaluate(valloader)

        for s in samples:
            s_oa = []
            s_aa = []
            for r in range(repeat):
                splits = self.get_splits(n, seed=r)  # same seed for each sample size, different repeats
                splits[0] = splits[0][: int(s * n)]
                # testset remains fixed for each sample size
                trainset, predset, testset = self.prepare_datasets(dataset, splits)
                gt_train = dataset.gt_flat[splits[0]].numpy()
                gt_test = dataset.gt_flat[splits[2]].numpy()
                if raw:
                    features_train = dataset.pixels[splits[0]]
                    features_test = dataset.pixels[splits[2]]
                else:
                    trainloader, valloader, testloader = self.prepare_dataloaders(trainset, predset, testset)

                    features_train = self.extract_features(modeller, trainloader)
                    features_test = self.extract_features(modeller, testloader)
                    print(features_test.shape)

                    # plot_clusters_predictions_hats(features_test[..., 0, 0], gt_test)
                    scales = features_test[..., 3, 0, 0]
                    print(scales.shape)

                    features_train = features_train.reshape(-1, self.cfg.k * 4)  # num params
                    features_test = features_test.reshape(-1, self.cfg.k * 4)

                # oacc, aacc, preds = predict_soil_classes(features_train, features_test, gt_train, gt_test, False)
                # s_oa.append(oacc)
                # s_aa.append(aacc)

            # plot_clusters_predictions_gt(features_test, preds, gt_test)
            print(features_test.shape)
            # plot_clusters_predictions_scale(features_test, scales, gt_test)
            plot_tsne(features_test, gt_test)
            # plot_clusters_predictions_hats(features_test, gt_test)

            # oa_low, oa_high = get_confidence_interval(s_oa)
            # aa_low, aa_high = get_confidence_interval(s_aa)

            # wandb.log(
            #     {
            #         f"OA_{100*s}%": np.mean(oacc),
            #         f"AA_{100*s}%": np.mean(aacc),
            #         f"OA_int_{100*s}%": (oa_high - oa_low) / 2,
            #         f"AA_int_{100*s}%": (aa_high - aa_low) / 2,
            #     }
            # )

        wandb.finish()

    def get_splits(self, n: int, train_p: float = 0.1, seed: int = 42) -> np.ndarray:
        train_n = int(0.4 * n)
        val_n = int(train_p * n)
        test_n = n - train_n - val_n
        split_ratio = [train_n, val_n, test_n]
        rng = np.random.default_rng(seed + 1)
        splits = np.split(rng.permutation(range(n)), np.cumsum(split_ratio))
        return splits

    def prepare_datasets(self, dataset: Dataset, splits: np.ndarray) -> tuple[Dataset, Dataset, Dataset]:
        trainset = Subset(dataset, splits[0])
        valset = Subset(dataset, splits[1])
        testset = Subset(dataset, splits[2])
        return trainset, valset, testset

    def prepare_dataloaders(self, trainset: Dataset, valset: Dataset, testset: Dataset) -> tuple[DataLoader]:
        return (
            DataLoader(trainset, batch_size=self.cfg.batch_size, shuffle=False),
            DataLoader(valset, batch_size=self.cfg.batch_size, shuffle=False),
            DataLoader(testset, batch_size=self.cfg.batch_size, shuffle=False),
        )

    def extract_features(self, modeller, dataloader):
        device = self.cfg.device
        features = []

        with torch.no_grad():
            for img in dataloader:
                img = img.to(device)
                ft = modeller(img)
                features.append(ft.cpu().numpy())

        return np.concatenate(features, axis=0)
