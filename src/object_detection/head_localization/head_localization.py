XMLS_PATH = "C:/python/torch_projects/src/object_detection/object_localization_dataset/annotations/xmls/"
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch


def xmls_extraction(path: str):

    records = []
    for file in os.listdir(path):
        tree = ET.parse(path + file)
        root = tree.getroot()

        filename = root.findtext("filename")

        width: int = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))

        xmin = int(root.findtext("object/bndbox/xmin"))
        ymin = int(root.findtext("object/bndbox/ymin"))
        xmax = int(root.findtext("object/bndbox/xmax"))
        ymax = int(root.findtext("object/bndbox/ymax"))

        # Normalize coordinates to [0, 1]
        xmin /= width
        xmax /= width
        ymin /= height
        ymax /= height

        records.append([filename, width, height, xmin, ymin, xmax, ymax])

    pd.DataFrame(
        records, columns=["filename", "width", "height", "xmin", "ymin", "xmax", "ymax"]
    ).to_csv(
        "C:/python/torch_projects/src/object_detection/head_localization/annotations.csv",
        index=False,
    )
