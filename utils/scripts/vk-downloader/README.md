## Download your vk.com playlist
---
### Step-by-step guide
These are exhaustive instrcutions that will convey a user from being forced to listen to music online to having all songs from his/her `vk.com` playlist downloaded onto a local hard drive. Let's get started.

1. Get a [Firefox] browser
2. Get [VK Music Playlist Download Add-on] from within your [Firefox] browser
3. Install [python] version `3.4+`
4. Open your playlist page in vk.com
5. Scroll the page to the bottom (yes, this will be tedious) untill you will see the very first song you've ever added to `vk.com` - this will load the entire playlist on current page
6. Use [VK Music Playlist Download Add-on] to download the playlist - it normally works by pressing the add-on button in the top right corner of your firefox
7. Copy the downloaded playlist to the same directory where your `vk-downloader.py` is located
8. Run the downloader in one of the following ways
    - using default parameters - which means that your playlist is named `playlist.m3u` and your songs will be downloaded to `songs` which will be created under the directory of `vk-downloader.py`
    - using customized parameters (lunch the script with `-h` to see more details) - you can specify where the playlist is and which folder to download the songs to as follows
    ```
    $ python3.4 ./vk-downloader.py -p ./playlist-den.m3u -t ../songs
    ```
9. Wait for the download to finish. This will take considerable time since the script is not running in parallel
10. Enjoy your songs

[Firefox]:https://www.mozilla.org/en-US/firefox/new/
[VK Music Playlist Download Add-on]:https://addons.mozilla.org/en-US/firefox/addon/vk-music-playlist-download/
[python]:https://www.python.org/downloads/

#### Disclaimer
Denys Sobchyshak DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL Denys Sobchyshak BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
