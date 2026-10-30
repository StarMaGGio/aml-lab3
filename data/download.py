from io import BytesIO
from zipfile import ZipFile
import requests

DATASET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
EXTRACT_PATH = './dataset'

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

if __name__ == '__main__':
    download_and_extract()
