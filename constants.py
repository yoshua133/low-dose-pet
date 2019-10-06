from PIL import Image as pil_image
"""
import deepmammo.models as models

MODEL2CLASS = {
  "densenet": models.DenseNet,
  "densenet_ddsm": models.DenseNetDDSM,
  "resnet": models.ResNet,
  "bagnet33": models.BagNet33,
  "bagnet17": models.BagNet17,
  "bagnet9": models.BagNet9,
}
"""

BIRADS_TO_LABEL = {
  '1': 1,
  '2': 1,
  '3': 2,
  '4': 3,
  '5': 3,
  '6': 3,  
}

ANNOTATED_METADATA_CSV_PATH = "/home/dxiang/unzipped_50_100_low_val_cases/csv.csv"
INTERPOLATION = "bicubic"
# mean and std dev of pixel values

PIL_INTERPOLATION_METHODS = {
  "nearest": pil_image.NEAREST,
  "bilinear": pil_image.BILINEAR,
  "bicubic": pil_image.BICUBIC,
}
# These methods were only introduced in version 3.4.0 (2016).
if hasattr(pil_image, "HAMMING"):
  PIL_INTERPOLATION_METHODS["hamming"] = pil_image.HAMMING
if hasattr(pil_image, "BOX"):
  PIL_INTERPOLATION_METHODS["box"] = pil_image.BOX
# This method is new in version 1.1.3 (2013).
if hasattr(pil_image, "LANCZOS"):
  PIL_INTERPOLATION_METHODS["lanczos"] = pil_image.LANCZOS
