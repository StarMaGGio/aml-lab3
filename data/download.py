from io import BytesIO
import os
import shutil
from zipfile import ZipFile
import requests

DATASET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
EXTRACT_PATH = './tiny-imagenet'

def format_val_folder(base_dir: str = './tiny-imagenet/tiny-imagenet-200'):
    """Riorganizza la cartella val creando sottocartelle per ciascuna classe."""
    val_dir = os.path.join(base_dir, 'val')
    annotations_path = os.path.join(val_dir, 'val_annotations.txt')
    images_dir = os.path.join(val_dir, 'images')

    if not os.path.exists(annotations_path):
        print(
            'File val_annotations.txt non trovato o cartella già ristrutturata.'
        )
        return

    with open(annotations_path, 'r') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)

            src_path = os.path.join(images_dir, fn)
            dst_path = os.path.join(cls_dir, fn)

            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)

    # Rimuovi la cartella immagini originale dopo aver copiato i file
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    print('Ristrutturazione cartella val completata!')

def download_and_extract(
        url: str = DATASET_URL, extract_to: str = EXTRACT_PATH
):
    print('Downloading the dataset...')
    response = requests.get(url)
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(extract_to)
        print('Download and extraction complete!')
    else:
        print(f'Failed to download dataset. Status code: {response.status_code}')

    format_val_folder()

if __name__ == '__main__':
    download_and_extract()
