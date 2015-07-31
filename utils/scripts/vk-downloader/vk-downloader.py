#!/usr/bin/env python
"""
    Copyright 2015 Denys Sobchyshak
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


import logging
import os.path
import time
import urllib.request as req
import urllib.error
from optparse import OptionParser


def download(playlist_path, target_path):
    logging.info('Parsing playlist')
    with open(playlist_path) as f:
        contents = f.read()
        songs = []
        urls = []
        for l in contents.split('\n'):
            if l.startswith('#EXTINF:'):
                songs.append((l.split(',', maxsplit=1)[1]).replace('/', '').strip()+'.mp3')
            if l.startswith('http'):
                urls.append(l)
    playlist = dict(zip(songs, urls))

    logging.info('Downloading songs')
    progress_counter = 0
    song_counter = len(playlist)
    progress(progress_counter/song_counter)
    removed_songs = dict()
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for k, v in playlist.items():
        try:
            req.urlretrieve(v, os.path.join(target_path, k))
        except ConnectionResetError as e:
            # uuuuu look, they r smart....
            time.sleep(2)
            req.urlretrieve(v, os.path.join(target_path, k))
        except urllib.error.HTTPError as e:
            removed_songs[k] = v
        except OSError as e:
            removed_songs[k] = v
        progress_counter += 1
        progress(progress_counter/song_counter)
    #getting rid of return carriage
    print()

    #writing down missed files
    with open(playlist_path+'-missed', 'w+') as f:
        for song, url in removed_songs.items():
            f.write('#EXTINF:1,'+song+'\n')
            f.write(url+'\n')

    logging.info('Missed file contents were saved as new playlist at: '+playlist_path+'-missed')
    logging.info('Download complete')


def progress(percentage):
    print('\rProgress: {}%'.format(int(percentage*100)), end='')


def initialize():
    #set up logger
    log_format = "%(asctime)s [%(levelname)s]:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)
    #parse input args
    parser = OptionParser()
    parser.add_option('-p', '--playlist-file', dest='playlist_file', default='playlist.m3u',
                      help='specifies a path to playlist', metavar='PLAYLIST_FILE')
    parser.add_option('-t', '--target-dir', dest='target_dir', default='songs',
                      help='specifies directory where playlist files are to be downloaded', metavar='TARGET_DIR')
    (options, args) = parser.parse_args()
    #post process params
    options.playlist_file = options.playlist_file.strip()
    options.target_dir = options.target_dir.strip()
    return options


def terminate(msg):
    logging.error(msg)
    exit()


if __name__ == '__main__':
    opts = initialize()
    download(opts.playlist_file, opts.target_dir)