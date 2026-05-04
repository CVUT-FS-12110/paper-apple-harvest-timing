import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class ValueDataset(Dataset):
    def __init__(self, file_list, image_size=(512, 512), augment=False):
        """
        Args:
            file_list (list[str]): List of full paths to image files.
                                   Filenames must follow: img_<days>.jpg
            image_size (tuple): Final size (width, height) for the image.
        """
        if not file_list:
            raise ValueError("The file_list is empty. Provide at least one file.")

        self._augment = augment
        self._augmentation = A.Compose([
            A.OneOf([
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),  # region where the flare center may appear
                    angle_lower=0.5,  # lower bound for flare angle in radians
                    angle_upper=1.0,  # upper bound
                    num_flare_circles_lower=6,  # number of flare circles (small reflections)
                    num_flare_circles_upper=10,
                    src_radius=300,  # radius of main flare
                    src_color=(255, 255, 255),  # color (usually white/yellow)
                    p=1.0
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),  # region of interest (y_min, x_min, y_max, x_max)
                    num_shadows_lower=1,  # min number of shadows
                    num_shadows_upper=2,  # max number of shadows
                    shadow_dimension=5,  # complexity of shadow polygons
                    always_apply=False,
                    p=1.0  # probability of applying
                ),
                # --- Spatial transforms ---
                A.Affine(scale=1.0, shear=0, rotate=0, translate_percent=0.5, p=1.0),  # Translation only
                A.Affine(translate_percent=0, shear=0, rotate=0, scale=(0.6, 1.6), p=1.0),  # Scaling
                A.Affine(translate_percent=0, scale=1, rotate=0, shear=(-45, 45), p=1.0),  # Shearing
                A.Affine(translate_percent=0, shear=0, scale=1, rotate=(-45, 45), p=1.0),  # Rotation

                # --- Dropout transforms ---
                A.CoarseDropout(
                    num_holes_range=(1, 10),
                    hole_height_range=(0.05, 0.3),
                    hole_width_range=(0.05, 0.3),
                    p=1.0
                ),
                A.GridDropout(
                    ratio=0.5,
                    unit_size_range=(10, 100),
                    p=1.0
                ),

                # --- Blur / noise ---
                A.GaussNoise(std_range=(0.0, 0.25), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),  # Simulates motion during capture
                A.MedianBlur(blur_limit=5, p=1.0),  # Reduces noise while blurring details

                # --- Distortion ---
                A.GridDistortion(num_steps=8, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.05, p=1.0),  # Camera lens-like distortion

                # --- Color / brightness / contrast ---
                A.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.4,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),

                # --- Flips ---
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),

                # --- Safe default (no-op) ---
                A.NoOp(p=1.0),
            ], p=1.0)
        ])

        self.image_size = image_size
        self.file_list = file_list

        # ---- Preload images into memory ----
        self._images = []
        self._targets = []

        for file_path in file_list:
            file_name = os.path.basename(file_path)

            # Load image
            image = cv.imread(file_path, cv.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {file_path}")

            # Convert BGR → RGB and resize
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, image_size, interpolation=cv.INTER_AREA)

            # Store image as float32 in range [0,1]
            self._images.append(image.astype(np.float32) / 255.0)

            # Extract target value from filename: "img_<days>.jpg"
            base_name = os.path.splitext(file_name)[0]
            try:
                days = int(base_name.split("_")[2])
            except (IndexError, ValueError):
                raise ValueError(f"Filename {file_name} does not match expected format 'img_<days>.jpg'")

            self._targets.append(days / 100.0)

        self._images = np.stack(self._images)  # shape: [N, H, W, C]
        self._targets = np.array(self._targets, dtype=np.float32)  # shape: [N]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        # Copy to ensure augmentation doesn't modify original
        image = self._images[idx].copy()

        # Augmentation (if enabled)
        if self._augment:
            augmented = self._augmentation(image=image)
            image = augmented["image"]

        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # Target to tensor
        target_tensor = torch.tensor(self._targets[idx], dtype=torch.float32)

        return image_tensor, target_tensor


# ---- Example usage ----
if __name__ == "__main__":
    import glob
    import matplotlib.pylab as plt

    dataset_root = "../data/dataset_class"
    file_list = glob.glob(os.path.join(dataset_root, "*.jpg"))

    unique_days = sorted({
        int(os.path.basename(f).split("_")[2].split(".")[0]) for f in file_list
    })
    print("Unique days:", unique_days)

    # Optionally filter the list, e.g., keep only those days >= 5
    file_list = [f for f in file_list if int(os.path.basename(f).split("_")[1].split(".")[0]) >= 5][0:5]

    print(f"Using {len(file_list)} images for the dataset.")

    # Create dataset
    dataset = ValueDataset(file_list, image_size=(512, 512), augment=True)

    # Visualize a few samples
    for idx in range(min(5, len(dataset))):
        image, target = dataset[idx]

        print(f"Sample {idx}: Target = {target.item():.2f}")

        # Convert back to numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()

        plt.imshow(img_np)
        plt.title(f"Target (days/10): {target.item():.2f}")
        plt.axis("off")
        plt.savefig(f"sample_{idx}.png")
        plt.close()