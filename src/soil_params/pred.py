import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import wandb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.consts import CHANNELS, GT_PATH
from src.models.modeller import Modeller


def prepare_datasets(
    dataset: Dataset,
    model: Modeller,
    k: int,
    num_params: int,
    batch_size: int,
    channels: int,
    device: str,
    ae: bool,
) -> tuple[np.ndarray]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    imgs = []
    preds = []

    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        data = data.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        # last param is shift - take it only once for all hats
        if not ae:
            shift = pred[:, 0, num_params - 1 :]
            pred[:, :, num_params - 1] = shift
        mask = data[:, 0] == 0

        expanded_mask = np.expand_dims(mask, axis=1)
        crop_mask = np.repeat(expanded_mask, repeats=CHANNELS, axis=1)
        masked_data = np.where(crop_mask == 0, data, np.nan)
        data_mean = np.nanmean(masked_data, axis=(2, 3))

        if not ae:
            expanded_mask = np.expand_dims(np.expand_dims(mask, axis=1), axis=2)
            crop_mask = np.repeat(np.repeat(expanded_mask, repeats=k, axis=1), repeats=num_params, axis=2)
            masked_pred = np.where(crop_mask == 0, pred, np.nan)
            pred_mean = np.nanmean(masked_pred, axis=(3, 4))
        else:
            expanded_mask = np.expand_dims(mask, axis=1)
            crop_mask = np.repeat(expanded_mask, repeats=k * num_params, axis=1)
            masked_data = np.where(crop_mask == 0, pred, np.nan)
            pred_mean = np.nanmean(masked_data, axis=(2, 3))
        imgs.append(data_mean)
        preds.append(pred_mean)

    imgs = np.array(imgs)
    preds = np.array(preds)
    img_means = imgs.reshape(imgs.shape[0] * batch_size, channels)
    pred_means = preds.reshape(preds.shape[0] * batch_size, k * num_params)
    return img_means, pred_means


def prepare_gt(ids: np.ndarray) -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH)
    gt = gt.loc[gt["sample_index"].isin(ids)]
    gt = gt.drop(["sample_index"], axis=1)
    return gt


def predict_params(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    # model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=1e-3))
    model = MultiOutputRegressor(MLPRegressor(max_iter=1000))
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def samples_number_experiment(
    x: np.ndarray, y: np.ndarray, sample_nums: list[int], n_runs: int = 10
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mses_mean = []
    mses_std = []

    for sn in sample_nums:
        mses_for_sample = []

        for run in range(n_runs):
            x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            x_train, y_train = x_train_base[:sn], y_train_base[:sn]

            mse = predict_params(x_train, x_test, y_train, y_test)
            mses_for_sample.append(mse)

        mses_for_sample = np.array(mses_for_sample)
        mses_mean.append(mses_for_sample.mean(axis=0))
        mses_std.append(mses_for_sample.std(axis=0))
    wandb.log({"soil/P": mses_mean[0][0], "soil/K": mses_mean[0][1], "soil/Mg": mses_mean[0][2],"soil/pH": mses_mean[0][3],})
    return np.array(mses_mean), np.array(mses_std)


def plot_soil_params(
    samples: list[int],
    mean_img: np.ndarray,
    std_img: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    gt: pd.DataFrame,
    k: int,
    num_params: int,
) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=gt.columns)
    x = list(range(len(samples)))
    means = [mean_img, mean_pred]
    stds = [std_img, std_pred]
    palette = px.colors.qualitative.Plotly
    names = ["Raw-150", f"ModG-{k*num_params-k+1}"]  # TODO Latent-n
    colors = [palette[i % len(palette)] for i in range(2)]

    for i in range(4):
        row = i // 2 + 1
        col = i % 2 + 1
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means[j][:, i],
                    mode="lines",
                    line=dict(color=colors[j]),
                    name=names[j],
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    name="-std",
                    x=x,
                    y=means[j][:, i] - stds[j][:, i],
                    opacity=0.4,
                    marker=dict(color=colors[j]),
                    mode="lines",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    name="+std",
                    x=x,
                    y=means[j][:, i] + stds[j][:, i],
                    opacity=0.4,
                    marker=dict(color=colors[j]),
                    mode="lines",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Number of samples", tickvals=x, ticktext=samples, row=row, col=col)
        fig.update_yaxes(title_text="MSE", row=row, col=col)

    fig.update_layout(title_text="MLP: mean and std of MSE for each predicted soil parameter", showlegend=True, hovermode="x")
    wandb.log({"soil_params": fig})


def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
) -> None:
    imgs_agg, preds_agg = prepare_datasets(dataset, model, cfg.k, num_params, cfg.batch_size, CHANNELS, cfg.device, ae)
    gt = prepare_gt(dataset.ids)
    samples = [500, 250, 200, 150, 100, 50, 25, 10]

    mses_mean_img, mses_std_img = samples_number_experiment(imgs_agg, gt, samples)
    mses_mean_pred, mses_std_pred = samples_number_experiment(preds_agg, gt, samples)

    plot_soil_params(samples, mses_mean_img, mses_std_img, mses_mean_pred, mses_std_pred, gt, cfg.k, num_params)
