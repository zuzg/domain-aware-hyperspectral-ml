import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
import wandb
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset

from src.benchmark.consts import SCENE_DICT, PU_CLASSES
from src.benchmark.pred_ml import predict_soil_classes
from src.benchmark.utils import get_confidence_interval
from src.config import ExperimentConfig
from src.consts import VIZ_PATH
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


def plot_tsne(features: np.ndarray, gt_labels: np.ndarray) -> None:
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
    plt.savefig(VIZ_PATH / "tsne.pdf")
    plt.show()


class SmallDataExperiment(Experiment):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)
        self._init_dirs()

    def run(self) -> None:
        repeat = 10
        samples = [0.5, 0.1, 0.05, 0.01, 0.005]
        data_cfg = SCENE_DICT[self.cfg.dataset_name]
        self.cfg.channels = data_cfg.channels
        dataset = HyperspectralScene(data_cfg)
        fulloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        n = len(dataset)
        raw = False
        tsne = False

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
                    features_train = features_train.reshape(-1, self.cfg.k * 4)  # num params
                    features_test = features_test.reshape(-1, self.cfg.k * 4)

                oacc, aacc, preds = predict_soil_classes(features_train, features_test, gt_train, gt_test, False)
                s_oa.append(oacc)
                s_aa.append(aacc)
            if tsne:
                plot_tsne(features_test, gt_test)

            oa_low, oa_high = get_confidence_interval(s_oa)
            aa_low, aa_high = get_confidence_interval(s_aa)

            wandb.log(
                {
                    f"OA_{100*s}%": np.mean(oacc),
                    f"AA_{100*s}%": np.mean(aacc),
                    f"OA_int_{100*s}%": (oa_high - oa_low) / 2,
                    f"AA_int_{100*s}%": (aa_high - aa_low) / 2,
                }
            )

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

    def extract_features(self, modeller: Modeller, dataloader: DataLoader) -> np.ndarray:
        device = self.cfg.device
        features = []

        with torch.no_grad():
            for img in dataloader:
                img = img.to(device)
                ft = modeller(img)
                features.append(ft.cpu().numpy())
        return np.concatenate(features, axis=0)
