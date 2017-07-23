"""
Source Code from
- https://github.com/Guim3/StackGAN/blob/master/download.py

Modification of
- https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
- http://stackoverflow.com/a/39225039

Downloads the following:
- CUB dataset
- Oxford-102 dataset
"""

import os
import sys
import argparse
import requests
import tarfile


parser = argparse.ArgumentParser(description='Download dataset for StackGAN.')
parser.add_argument('datasets', metavar='D', type=str.lower, nargs='+', choices=['cub', 'oxford-102'],
                   help='name of dataset to download [cub, oxford-102]')
parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                   help='path to store the data (default ./datasets)')

# Downloat a file from google drive given its id and the destination file.
def download_file_from_google_drive(id, destination):
    url = 'https://docs.google.com/uc?export=download'

    session = requests.Session()

    response = session.get(url, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(url, params = params, stream = True)

    save_response_content(response, destination)


# Confirm tokens from website
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


# Save the content of a given response from google drive
def save_response_content(response, destination):
    chunk_size = 32768

    print('start saving {}'.format(destination))
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print('saved {}'.format(destination)) 


# Extract .tar.gz and .tgz files
def extract_tar(tar_path, extract_path='.'):
    tar = tarfile.open(tar_path, 'r')
    print('start extracting {}'.format(tar_path))
    for item in tar:
        f_name = os.path.basename(os.path.abspath(item.name))
        if not f_name.startswith('.'):
            tar.extract(item, extract_path)
            if item.name.find('.tgz') != -1 or item.name.find('.tar') != -1:
                extract_tar(item.name, extract_path + '/' + item.name[:item.name.rfind('/')])
    os.remove(tar_path)
    print('finished extracting {}'.format(tar_path))


# Download and extracts the fila at url
def download_extract_tar(id, data_dir):
    filepath = os.path.join(data_dir, 'aux.tar.gz')
    download_file_from_google_drive(id, filepath)
    extract_tar(filepath, data_dir)


# Download CUB dataset
def download_cub(data_dir):
    data_dir = os.path.join(data_dir, 'cub')
    if os.path.exists(data_dir):
        print('Found CUB - skip')
        return
    
    os.mkdir(data_dir)
    url = '0B-y41dOfPRwROVBWUjlpM1BhbzQ'
    download_extract_tar(url, data_dir)


# Download Oxford-102 dataset
def download_oxford_102(data_dir):
    data_dir = os.path.join(data_dir, 'oxford-102')
    if os.path.exists(data_dir):
        print('Found Oxford-102 - skip')
        return

    os.mkdir(data_dir)
    url = '0B-y41dOfPRwRUzVxU3pMTEtaT1U'
    download_extract_tar(url, data_dir)


def prepare_data_dir(path='./data'):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.path is None:
        args.path = './datasets'
    prepare_data_dir(args.path)

    if 'cub' in args.datasets:
        download_cub(args.path)
    if 'oxford-102' in args.datasets:
        download_oxford_102(args.path)
