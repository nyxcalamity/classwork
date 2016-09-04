#!/usr/bin/env python
"""
    Licencing terms here
"""
import logging
import requests
import concurrent.futures as cnc
import multiprocessing as mp

import utilmix as um


__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


def download_image(url, dest):
    """
    Downloads a single image into provided destination.
    :param url:
    :param dest:
    """
    logging.info('Downloading {} into {}'.format(url, dest))
    dest = um.join_paths(dest, url.split('/')[-1])
    response = requests.get(url)
    if um.is_image_response(response):
        with open(dest, 'wb') as f:
            f.write(response.content)


if __name__ == '__main__':
    parser = um.get_cli_parser()
    parser.add_option("-f", "--file", dest="filename", default=um.join_paths(um.get_script_path(), 'links'),
                      help="data file with urls")
    parser.add_option("-d", "--dest", dest="dest", default=um.join_paths(um.get_script_path(), 'images'),
                      help="destination folder")
    opts = um.parse_cli_options(parser)
    um.set_logging()

    # create destination path if it doesn't exist
    um.make_path(opts.dest)

    with cnc.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        with open(opts.filename) as f:
            for line in f:
                executor.submit(download_image, line.replace('\n', ''), opts.dest)
