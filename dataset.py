import io
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def get_train_transform(
        height: int = 600,
        width: int = 600,
) -> A.Compose:
    """학습용 이미지/마스크 동시 증강 파이프라인."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7,
        ),
        A.OneOf([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_valid_transform(
        height: int = 600,
        width: int = 600,
) -> A.Compose:
    """검증 및 추론용 전처리 파이프라인."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


class HumanSegDataset(Dataset):
    """바이트 인코딩된 image/mask 열을 갖는 Parquet 세그멘테이션 데이터셋.

    반환값:
        image: float32 텐서, shape=(3, H, W)
        mask:  float32 이진 텐서, shape=(1, H, W)
    """

    def __init__(
            self,
            parquet_path: str = "humanseg.parquet",
            transform: A.Compose | None = None,
    ) -> None:
        super().__init__()
        self.table = pq.read_table(parquet_path, columns=["image", "mask"])
        self.images = self.table["image"]
        self.masks = self.table["mask"]
        self.transform = transform or get_valid_transform()

    def __len__(self) -> int:
        return self.table.num_rows

    @staticmethod
    def _decode(data: bytes, flag: int, name: str) -> np.ndarray:
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
        if array is not None:
            return array

        try:
            with Image.open(io.BytesIO(data)) as image:
                if flag == cv2.IMREAD_COLOR:
                    array = np.array(image.convert("RGB"))
                    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                return np.array(image)
        except Exception as exc:
            raise ValueError(f"{name} 데이터를 디코딩하지 못했습니다.") from exc

    def _preprocess_data(self, image_bytes: bytes, mask_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._decode(
            image_bytes,
            cv2.IMREAD_COLOR,
            "image",
        )
        mask = self._decode(
            mask_bytes,
            cv2.IMREAD_UNCHANGED,
            "mask",
        )

        # OpenCV의 BGR 이미지를 일반적인 모델 입력인 RGB 순서로 변환한다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 컬러 또는 (H, W, 1) 마스크도 단일 채널로 통일한다.
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[..., 0]
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1))
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        image = image.to(dtype=torch.float32)
        mask = (mask > 0).to(dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image.contiguous(), mask.contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._preprocess_data(
            self.images[index].as_py(),
            self.masks[index].as_py(),
        )


def visualize_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.array((0, 255, 0), dtype=np.float32)  # BGR
    result = img.copy()

    mask = (mask > 0).astype(np.uint8)
    selected = mask.astype(bool)

    # 마스크 내부: alpha 0.5
    result[selected] = (
            result[selected].astype(np.float32) * 0.5
            + color * 0.5
    ).astype(np.uint8)

    # 마스크 테두리: 1px, alpha 1.0
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(result, contours, -1, color.tolist(), 1)

    return result


def load_parquet():
    # Parquet 전체를 메모리에 로딩
    table = pq.read_table(
        "humanseg.parquet",
        columns=["image", "mask"],
    )

    images = table["image"]
    masks = table["mask"]

    print(f"데이터 개수: {table.num_rows:,}")

    # cv2.namedWindow("HumanSeg", cv2.WINDOW_NORMAL)

    for i in tqdm(range(table.num_rows)):
        image_bytes = images[i].as_py()
        mask_bytes = masks[i].as_py()

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imdecode(
            np.frombuffer(mask_bytes, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # palette = cv2.hconcat([image, mask])
        palette = visualize_mask(image, mask)

        cv2.imshow("HumanSeg", palette)

        # 100ms마다 다음 이미지, ESC로 종료
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_parquet()
