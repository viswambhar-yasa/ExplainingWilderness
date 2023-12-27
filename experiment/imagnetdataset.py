# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: ImageNet Dataset can be downloaded and extract using this file
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The url may become invalid

import os
import requests
from tqdm import tqdm
from pathlib import Path

class DownloadDataset():
    def __init__(self, dataname="imagenet") -> None:
        """
        Initializes a new instance of the DownloadDataset class.

        Args:
            dataname (str, optional): The name of the dataset. Defaults to "imagenet".
        """
        self.datasetname = dataname
        pass

    def retrieveURL(self, url, output_path, filename):
        """
        Downloads a file from a given URL and saves it to the specified output path.

        Args:
            url (str): The URL of the file to be downloaded.
            output_path (str): The path where the downloaded file should be saved.
            filename (str): The name of the downloaded file.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(output_path, filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(save_path, 'wb') as f, tqdm(
                    desc=save_path,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print("Data downloaded successfully")

    def loadDataSet(self, output_path):
        """
        Loads the dataset.

        Args:
            output_path (str): The path where the dataset should be loaded.
        """
        if self.datasetname == "imagenet":
            ImageNet_ValdataURL = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
            output_file = 'ILSVRC2012_img_val.tar'
            self.retrieveURL(ImageNet_ValdataURL, output_path, output_file)
            ImageNet_classesURL = "https://datacloud.hhi.fraunhofer.de/s/iz2MYBiGFCpxfg5/download"
            ImageNet_classesfilename = "ILSVRC2012_devkit_t12.tar.gz"
            self.retrieveURL(ImageNet_classesURL, output_path, ImageNet_classesfilename)
            # ImageNet_ValdataURL = "https://datacloud.hhi.fraunhofer.de/s/RjnK4badZgG7gMq/download"
            # ImageNet_Valfilename="ILSVRC2012_img_val.tar"
            # self.retrieveURL(ImageNet_ValdataURL,output_path, ImageNet_Valfilename)