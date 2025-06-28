import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.consts import GT_PATH, MEAN_PATH, water_bands
from src.models.modeller import Modeller


def collate_fn_pad(batch):
    max_h = 3# max(img.shape[1] for img in batch)  # Find max height in batch
    max_w = 3#max(img.shape[2] for img in batch)  # Find max width in batch

    padded_batch = []
    for img in batch:
        padded_img = nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))  # Pad to max size
        padded_batch.append(padded_img)

    return torch.stack(padded_batch)  # Stack into a single tensor


def apply_baseline_mask(data: np.ndarray, mask: np.ndarray, channels: int) -> np.ndarray:
    expanded_mask = np.expand_dims(mask, axis=1)
    crop_mask = np.repeat(expanded_mask, repeats=channels, axis=1)
    return np.where(crop_mask == 0, data, 0)


def apply_non_baseline_mask(pred: np.ndarray, mask: np.ndarray, k: int, num_params: int, ae: bool) -> np.ndarray:
    if ae:
        expanded_mask = np.expand_dims(mask, axis=1)
        crop_mask = np.repeat(expanded_mask, repeats=k * num_params, axis=1)
    else:
        shift = pred[:, 0, num_params - 2 : num_params - 1]
        pred[:, :, num_params - 2] = shift  # TODO drop instead of copying
        expanded_mask = np.expand_dims(np.expand_dims(mask, axis=1), axis=2)
        crop_mask = np.repeat(np.repeat(expanded_mask, repeats=k, axis=1), repeats=num_params, axis=2)
    return np.where(crop_mask == 0, pred, 0)


def prepare_datasets(
    dataset: Dataset,
    model: Modeller,
    k: int,
    channels: int,
    num_params: int,
    batch_size: int,
    device: str,
    ae: bool = False,
    baseline: bool = False,
) -> np.ndarray:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_pad)
    features = []
    img_means = []
    model.eval()
    for data in dataloader:
        mask = data[:, 0] == 0
        if baseline:
            masked_pred = apply_baseline_mask(data, mask, channels)
        else:
            data = data.to(device)
            pred = model(data)
            pred = pred.cpu().detach().numpy()
            masked_pred = apply_non_baseline_mask(pred, mask, k, num_params, ae)
        data = data.cpu().detach().numpy()
        data[data == 0] = np.nan
        img_means.append(np.nanmean(data, axis=(2, 3)))
        features.append(masked_pred)

    features = np.array(features)
    img_means = np.array(img_means)
    # sort by mu
    sort_key = features[:, :, :, 0, :, :]
    sort_indices = np.argsort(sort_key, axis=2)
    sorted_features = np.take_along_axis(features, np.expand_dims(sort_indices, axis=3), axis=2)
    print(len(img_means))

    f_dim = channels if baseline else k * num_params
    sorted_features = sorted_features.reshape(
        features.shape[0] * batch_size, f_dim, features.shape[-2], features.shape[-1]
    )
    # Indices of shift_* except shift_1 (index 3)
    drop_indices = [i * num_params + 3 for i in range(1, k)]
    # Remove those feature channels
    features_filtered = np.delete(sorted_features, drop_indices, axis=1)
    return features_filtered, img_means.reshape(features.shape[0] * batch_size, img_means.shape[-1])


def prepare_gt(ids: np.ndarray) -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH)
    gt = gt.loc[gt["sample_index"].isin(ids)]
    gt = gt.drop(["sample_index"], axis=1)
    gt = gt.reset_index(drop=True)
    return gt


def aggregate_features(features: np.ndarray, extended: bool = True) -> np.ndarray:
    features[features == 0] = np.nan
    preds_mean = np.nanmean(features, axis=(2, 3))
    if not extended:
        return preds_mean
    # preds_var = np.nanvar(features, axis=(2, 3))
    # preds_min = np.nanpercentile(features, q=1, axis=(2, 3))
    # preds_max = np.nanpercentile(features, q=99, axis=(2, 3))

    p1, p2, p3 = np.nanpercentile(features, q=[1, 50, 99], axis=(2, 3))
    # preds_median = np.nanmedian(features, axis=(2, 3))
    # preds_skew = skew(features, axis=(2, 3), nan_policy="omit")
    # preds_kurt = kurtosis(features, axis=(2, 3), nan_policy="omit")
    # q75, q25 = np.nanpercentile(features, [75, 25], axis=(2, 3))
    # preds_iqr = q75 - q25

    features_agg = np.concatenate(
        [preds_mean, p1, p2, p3], axis=1
    )  # , preds_median, preds_skew, preds_kurt, preds_iqr], axis=1)
    return features_agg


def compute_indices(img: np.ndarray) -> np.ndarray:
    # Band assignment (Sentinel-2)
    blue = img[1]
    green = img[2]
    red = img[3]
    nir = img[7]
    swir1 = img[10]
    swir2 = img[11]

    indices = {
        'NDVI': np.nanmean((nir - red) / (nir + red + 1e-6)),
        'EVI': np.nanmean(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)),
        'NDWI': np.nanmean((nir - swir1) / (nir + swir1 + 1e-6)),
        'SAVI': np.nanmean((1.5 * (nir - red)) / (nir + red + 0.5 + 1e-6)),
        'ClayIndex': np.nanmean(swir1 / (swir2 + 1e-6)),
        'IronOxideIndex': np.nanmean(red / (blue + 1e-6)),
        'NDSI': np.nanmean((swir1 - green) / (swir1 + green + 1e-6)),

        # 'NDVI_s': np.nanstd((nir - red) / (nir + red + 1e-6)),
        # 'EVI_s': np.nanstd(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)),
        # 'NDWI_s': np.nanstd((nir - swir1) / (nir + swir1 + 1e-6)),
        # 'SAVI_s': np.nanstd((1.5 * (nir - red)) / (nir + red + 0.5 + 1e-6)),
        # 'ClayIndex_s': np.nanstd(swir1 / (swir2 + 1e-6)),
        # 'IronOxideIndex_s': np.nanstd(red / (blue + 1e-6)),
        # 'NDSI_s': np.nanstd((swir1 - green) / (swir1 + green + 1e-6)),
    }
    vals = np.fromiter(indices.values(), dtype=float)
    return vals


def compute_indices_extended(img: np.ndarray) -> np.ndarray:
    # Sentinel-2 bands
    blue = img[1]
    green = img[2]
    red = img[3]
    red_edge1 = img[4]  # 705 nm
    red_edge2 = img[5]  # 740 nm
    red_edge3 = img[6]  # 783 nm
    nir = img[7]        # 842 nm
    swir1 = img[10]     # 1610 nm
    swir2 = img[11]     # 2190 nm

    indices = {
        # Vegetation health
        'NDVI': np.nanmean((nir - red) / (nir + red + 1e-6)), # TF
        'EVI': np.nanmean(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)), # FF
        'SAVI': np.nanmean((1.5 * (nir - red)) / (nir + red + 0.5 + 1e-6)), # FF
        'NDRE': np.nanmean((nir - red_edge2) / (nir + red_edge2 + 1e-6)), # FT
        'RECI': np.nanmean((nir / red_edge1) - 1),  # Red-edge chlorophyll index # FT

        # Soil/Water indicators
        'NDWI': np.nanmean((nir - swir1) / (nir + swir1 + 1e-6)), # TT
        'BSI': np.nanmean(((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-6)),  # Bare soil index # FT
        'SI': np.nanmean((swir1 - nir) / (swir1 + nir + 1e-6)),  # Simple soil index # TF

        # Chemical/mineral indicators
        'ClayIndex': np.nanmean(swir1 / (swir2 + 1e-6)), # TT
        'NDSI': np.nanmean((swir1 - green) / (swir1 + green + 1e-6)),  # iron oxide proxy# TT
        'FeIndex': np.nanmean(red / (blue + 1e-6)),  # Iron oxide index # TT
        'OMI': np.nanmean((red + green) / (nir + 1e-6)),  # Organic matter index (simplified) # FT
        
        # Red-edge slope (proxy for crop stress / subtle pigment changes)
        'RedEdgeSlope': np.nanmean((red_edge3 - red_edge1) / (740 - 705 + 1e-6)), # TF
    }

    return np.fromiter(indices.values(), dtype=float)



def extract_texture_features(img: np.ndarray, levels: int = 32) -> np.ndarray:
    red = img[3]
    nir = img[7]
    index_img = (nir - red) / (nir + red + 1e-6)
    img_scaled = exposure.rescale_intensity(index_img, out_range=(0, levels - 1)).astype(np.uint8)
    glcm = graycomatrix(img_scaled, [1], [0], levels=levels, symmetric=True, normed=True)
    textures =  {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        # 'energy': graycoprops(glcm, 'energy')[0, 0],
        # 'correlation': graycoprops(glcm, 'correlation')[0, 0],
    }
    vals = np.fromiter(textures.values(), dtype=float)
    return vals


def center_crop(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    c, h, w = img.shape
    ch, cw = size
    starth = h // 2 - (ch // 2)
    startw = w // 2 - (cw // 2)
    return img[:, starth:starth + ch, startw:startw + cw]


def random_crop(img: np.ndarray, size: tuple[int, int], max_attempts: int = 10) -> np.ndarray:
    c, h, w = img.shape
    ch, cw = size
    if ch > h or cw > w:
        raise ValueError("Crop size must be less than or equal to image size.")

    for attempt in range(max_attempts):
        starth = np.random.randint(0, h - ch + 1)
        startw = np.random.randint(0, w - cw + 1)
        crop = img[:, starth:starth + ch, startw:startw + cw]

        if not np.isnan(crop).all():
            return crop

    raise RuntimeError("Failed to find a valid crop without all NaNs after multiple attempts.")


DIMS = {4: (2, 2), 6: (3, 2), 7: (2, 3), 9: (3, 3)}


def load_msi_images(directory: Path, aug: bool = False) -> np.ndarray:
    filenames = [f for f in Path(directory).rglob("*.npz") if "msi_satellite" in str(f)]
    filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
    image_list = []
    aug_image_list = []
    aug_ids = []
    for i, filename in enumerate(filenames):
        with np.load(filename) as npz:
            arr = np.ma.MaskedArray(**npz)
            img = arr.data
            # texture = extract_texture_features(img)
            img[arr.mask] = np.nan
            size = np.array([img.shape[1] * img.shape[2]])
            channel_mean = np.nanmean(img, axis=(1, 2))
            # channel_std = np.nanstd(img, axis=(1,2))
            # channel_min = np.nanpercentile(img, q=1, axis=(1, 2))
            # channel_max = np.nanpercentile(img, q=99, axis=(1, 2))
            # p1, p2, p3, p4 = np.nanpercentile(img, q=[10, 25, 75, 90], axis=(1, 2))
            p1, p2, p3 = np.nanpercentile(img, q=[1, 50, 99], axis=(1, 2))
            indices = compute_indices(img)
            image_list.append(np.concatenate([channel_mean, p1, p2, p3, indices], axis=0)) # variance?

            if aug and size > 9:
                # for j in range(2): # two augs per large image
                new_size = np.random.choice([4, 6, 7, 9])
                new_img = random_crop(img, DIMS[new_size])
                if new_size == 7:
                    new_size = 6
                channel_mean = np.nanmean(new_img, axis=(1, 2))
                p1, p2, p3 = np.nanpercentile(img, q=[1, 50, 99], axis=(1, 2))
                indices = compute_indices(new_img)
                new_img[new_img == np.nan] = 0
                # texture = extract_texture_features(new_img)
                aug_image_list.append(np.concatenate([channel_mean, p1, p2, p3, indices], axis=0))
                aug_ids.append(i)
    return np.array(image_list), np.array(aug_image_list), aug_ids


def load_hsi_airborne_images(directory: Path, pred: bool = False) -> np.ndarray:
    filenames = [f for f in Path(directory).rglob("*.npz") if "hsi_airborne" in str(f)]
    filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
    image_list = []
    for filename in filenames:
        with np.load(filename) as npz:
            arr = np.ma.MaskedArray(**npz).astype(np.float32)
            img = arr.data
            img[arr.mask] = np.nan
            channel_mean = np.nanmean(img, axis=(1, 2))
            image_list.append(channel_mean)
    means = np.array(image_list)
    pca = PCA(n_components=2)
    if pred:
        with open("output/models/pca_inst.pickle", "rb") as f:
            pca = pickle.load(f)
            means_pca = pca.transform(means)
    else:
        means_pca = pca.fit_transform(means)
        print(pca.explained_variance_ratio_)
        with open("output/models/pca_inst.pickle", "wb") as f:
                pickle.dump(pca, f)
    return means_pca[:, 1:]  # omit first component - noise


def spectral_angle_mapper(s1: np.ndarray, s2: np.ndarray) -> float:
    dot_product = np.dot(s1, s2)
    norms = np.linalg.norm(s1) * np.linalg.norm(s2)
    return np.arccos(np.clip(dot_product / (norms + 1e-6), -1.0, 1.0))


# MINERALS = [
#         "Goethite", "Hematite", "Chalcopyrite", "Malachite", "Pyrolusite",
#         "Rhodochrosite", "Sphalerite", "Ulexite", "Gypsum"
#     ]
MINERALS = [
        "Goethite", "Hematite", "Gypsum", "Chalcopyrite", "Sphalerite", "Pyrolusite", "Ulexite"
    ]
# MINERALS = ["Goethite"]

def extract_sam_features_from_prisma_image(prisma_image: np.ndarray, usgs_reflectances: pd.DataFrame) -> np.ndarray:
    mean_spectrum = np.nanmean(prisma_image, axis=(1, 2))  # shape: (bands,)
    features = []
    for mineral in MINERALS:
        sam_angle = spectral_angle_mapper(mean_spectrum, usgs_reflectances[mineral])
        features.append(sam_angle)
    return np.array(features)


def load_minerals(directory: Path) -> np.ndarray:
    filenames = [f for f in Path(directory).rglob("*.npz") if "hsi_satellite" in str(f)]
    filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
    image_list = []
    usgs = pd.read_json("data/HYPERVIEW2/spectral_curves/prisma_mineral_curves.json")
    for filename in filenames:
        with np.load(filename) as npz:
            arr = np.ma.MaskedArray(**npz)
            img = arr.data
            sam_features = extract_sam_features_from_prisma_image(img, usgs)
            image_list.append(sam_features)
    return np.array(image_list)


class SoilDataset(Dataset):
    def __init__(
        self, image_list: np.ndarray, size: int, gt: pd.DataFrame, gt_div: np.ndarray, shuffle: bool = True
    ) -> None:
        super().__init__()
        self.size = size
        self.images = torch.from_numpy(image_list)
        self.gt = gt
        self.gt_div = gt_div
        self.gt_dim = len(gt_div)

        if shuffle:
            self.images, self.gt_values = self._shuffle_pixels_between_images()
        else:
            self.gt_values = self._prepare_gt_values()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return self.images[index], self.gt_values[index]

    def _prepare_gt_values(self) -> torch.Tensor:
        num_images = len(self.images)
        gt_values = torch.zeros((num_images, self.gt_dim, self.size, self.size), dtype=torch.float32)

        for i in range(num_images):
            soil = torch.from_numpy(self.gt.loc[i].values / self.gt_div).float()
            gt_unsqueezed = soil.unsqueeze(1).unsqueeze(2)
            gt_values[i] = gt_unsqueezed.repeat(1, self.size, self.size)

        return gt_values

    def _shuffle_pixels_between_images(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_images, channels, height, width = self.images.shape
        num_pixels = height * width

        # Flatten images and ground truth tensors
        flat_images = self.images.view(num_images, channels, num_pixels)
        flat_gt = self._prepare_gt_values().view(num_images, self.gt_dim, num_pixels)

        # Shuffle pixels across all images using random indices
        shuffled_indices = torch.randperm(num_pixels)
        shuffled_images = flat_images[:, :, shuffled_indices]
        shuffled_gt = flat_gt[:, :, shuffled_indices]

        # Reshape back to original dimensions
        shuffled_images = shuffled_images.view(num_images, channels, height, width)
        shuffled_gt = shuffled_gt.view(num_images, self.gt_dim, height, width)

        return shuffled_images, shuffled_gt


class ImgGtDataset(Dataset):
    def __init__(self, directory: str, ids: np.ndarray, std, max_val, gt: pd.DataFrame, gt_div: np.ndarray, size: int):
        super().__init__()
        self.gt = gt
        self.ids = ids
        self.gt_div = gt_div
        self.size = size
        self.max_val = max_val
        self.bias = self.load_bias(MEAN_PATH)
        self.images = self.load_images(directory)
        self.bias = self.bias[self.water_mask]
        self.transform = transforms.Compose([transforms.Normalize(0, std[self.water_mask])])
        self.gt_dim = len(gt_div)
        self.gt_values = self._prepare_gt_values()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        return image, self.gt_values[index]

    def _prepare_gt_values(self) -> torch.Tensor:
        num_images = len(self.images)
        gt_values = torch.zeros((num_images, self.gt_dim, self.size, self.size), dtype=torch.float32)

        for i in range(num_images):
            soil = torch.from_numpy(self.gt.loc[i].values / self.gt_div).float()
            gt_unsqueezed = soil.unsqueeze(1).unsqueeze(2)
            gt_values[i] = gt_unsqueezed.repeat(1, self.size, self.size)

        return gt_values
    
    def delete_water_bands(self, img: np.ndarray) -> np.ndarray:
        channel_indices = np.arange(img.shape[0])
        keep_mask = np.ones(img.shape[0], dtype=bool)
        for start, end in water_bands:
            keep_mask[(channel_indices > start) & (channel_indices < end)] = False
        self.water_mask = keep_mask
        filtered_img = img[keep_mask]
        return filtered_img

    def load_bias(self, bias_path: str) -> np.ndarray:
        with open(bias_path, "rb") as f:
            bias = np.load(f)
        return bias.reshape(bias.shape[0], 1, 1)

    def load_images(self, directory: str) -> list[Tensor]:
        filenames = list(Path(directory).rglob("*.npz"))
        filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
        image_list = []
        for filename in filenames:
            if int(filename.stem) in self.ids:
                with np.load(filename) as npz:
                    arr = np.ma.MaskedArray(**npz)
                    img = arr.data - self.bias
                    img = self.delete_water_bands(img)
                    img[img > self.max_val] = self.max_val  # clip outliers to max_val
                    image_list.append(torch.from_numpy(img).float())
        return image_list
