#!/usr/bin/env python
"""
    Licencing terms here
"""
import os
import logging
from optparse import OptionParser


__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


# Scripting
# ----------------------------------------------------------------------------------------------------------------------
def set_logging(log_level=logging.INFO):
    """
    Initiates default log formatting and level
    :param log_level:
    """
    log_format = '%(asctime)s [%(levelname)s]:%(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)


def get_cli_parser(usage='usage: %prog [options] arg', include_verbose=False):
    """
    Default initialization of option parser.
    :param usage:
    :param include_verbose:
    :return: the parser
    """
    parser = OptionParser(usage)
    if include_verbose:
        parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=True, help="print log messages")
    return parser


def parse_cli_options(parser):
    """
    Parses options and removes empty spaces in option values.
    :param parser:
    :return: stripped options
    """
    (opts, args) = parser.parse_args()
    for key in opts.__dict__.keys():
        opts.__dict__[key] = opts.__dict__[key].strip()
    return opts


# Language convenience
# ----------------------------------------------------------------------------------------------------------------------
def filter_none(l):
    """
    Filters out none objects in a list
    :param l:
    :return: filtered object
    """
    return list(filter(None, l))


def cut_tail(l):
    """
    Filters out None or empty objects in the end of a list
    :param l:
    :return: filtered list
    """
    while not l[-1]:
        l.pop()
    return l


# FS
# ----------------------------------------------------------------------------------------------------------------------
def get_script_path():
    """
    :return: directory, where invoked script resides
    """
    return os.path.dirname(os.path.realpath(__file__))


def join_paths(*paths):
    """
    Looks for a resource (e.g. configuration file).
    :param workspace:
    :param name:
    :param resources_dir:
    :return: path to requested resource
    """
    return os.path.join(*paths)


def make_path(path):
    """
    Attempts to create a path and all of the missing directories in the path.
    :param path:
    :return:
    """
    if path and not os.path.exists(path):
        os.makedirs(path)


# Web
# ----------------------------------------------------------------------------------------------------------------------
# strictly defined acceptable image content types
image_content_types = {
    'image/png': '.png',
    'image/pjpeg': '.jpg',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/bmp': '.bmp',
    'image/x-icon': '.ico',
    'image/tiff': '.tif',
    'image/x-tiff': '.tif'
}


def is_image_response(response):
    """
    Checks response for correctness and having an image.
    :param response:
    :return: True if checks were passed
    """
    if response and response.status_code == 200:
        if response.headers['Content-Type'] in image_content_types.keys():
            return True
