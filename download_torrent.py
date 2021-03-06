# @title Torrent Class
import libtorrent as lt
from queue import Queue
from threading import Thread
import time
import os
import datetime
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from tqdm.notebook import tqdm

# from google.colab import drive
# drive.mount("/content/drive")

from google.colab import files
from termcolor import cprint

tor_path = '/content/Torrent/tor/'
save_path = '/content/Torrent/'
completed_path = '/content/Torrent/c/'

for path in [tor_path, save_path, completed_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# define the session of libtorrent
ses = lt.session()
ses.listen_on(6881, 6891)
# sett = lt.session_settings()
# sett = {'allow_multiple_connections_per_ip': True,
#         'dont_count_slow_torrents': True,
#         'active_downloads': 10,
#         'active_seeds': 4,
#         'active_checking': 3, }
# ses.set_settings(sett)

# Gui
layout = widgets.Layout(width="auto")
style = {"description_width": "initial"}


class Torrent(Thread):
    state_str = [
        "Queued",
        "Checking",
        "Downloading Metadata",
        "Downloading",
        "Finished",
        "Seeding",
        "Allocating",
        "Checking Fastresume",
    ]
    tor_paths = {}  # Keeps the track of which torrents are running and which are not
    tor_file_status = {}
    STORAGE_SIZE = 65
    current_torrent_size = 0
    timestamp = 0
    torrent_progress = {}

    def __init__(self):
        print("got here")
        Thread.__init__(self)
        print("Thread class initalized")

    def run(self):
        self.magnets()

    # Get the torrents from magnet links
    def magnets(self):
        params = {
            'save_path': save_path,
            'storage_mode': lt.storage_mode_t(2),
            # 'paused': False,
            # 'auto_managed': True,
            # 'duplicate_is_error': True,
            'file_priorities': [0] * 1000
        }
        magnet_link = ''
        while True:
            print("\nEnter Magnet Link Or Type Exit: ")
            magnet_link = input('\n')
            if magnet_link.lower() == "exit":
                break
            # if not magent.link.lower().startswith('magnet'): continue
            # print(magnet_link)
            # print(type(ses),type(magnet_link), type(params))
            # params['uri'] = lt.add_magnet_uri(ses, magnet_link, params)
            handle = lt.add_magnet_uri(ses, magnet_link, params)
            # handle = ses.add_torrent(params)
            self.tor_file_status[handle] = False
            cprint('Magnet link added {}'.format(len(self.tor_file_status)), 'green')

    def add_torrent(self, link):
        # print(link)
        ti = lt.torrent_info(tor_path + link)
        torrent_size = ti.total_size() / (1024 * 1024 * 1024)

        if torrent_size > self.STORAGE_SIZE:
            if not os.path.exists(tor_path + 'oversized_torrent/'):
                os.makedirs(tor_path + 'oversized_torrent/')
            os.replace(tor_path + link, tor_path + 'oversized_torrent/' + link.split('/')[-1])
            # try:
            #   ses.remove_torrent(torrent)
            # except:
            #   pass
            return True

        # If system can't accommodate more torrent, then dont add the param to the torrent
        if self.current_torrent_size + torrent_size > self.STORAGE_SIZE:
            new_timestamp = ts = datetime.datetime.now().timestamp() + (
                        self.current_torrent_size + torrent_size - self.STORAGE_SIZE) * 100

            if self.timestamp == 0:
                self.timestamp = new_timestamp

            self.timestamp = min(new_timestamp, self.timestamp)
            return False

        # else add the torrent to session
        params = {
            'save_path': save_path,
            'storage_mode': lt.storage_mode_t(2),
            # 'paused': False,
            # 'auto_managed': True,
            # 'duplicate_is_error': True,
            'ti': ti
        }

        try:
            self.current_torrent_size += torrent_size
            ses.async_add_torrent(params)
            print('Added ', link.split('/')[-1])
            self.current_torrent_size += torrent_size
            # print("Torrent added !! ",link)
        except RuntimeError as e:
            cprint('!! Torrent already present', 'red')
        return True

    def load_torrents(self):
        # get the tor files from the drive
        paths = os.listdir(tor_path)
        paths = [path for path in paths if path.endswith('.torrent')]
        # print(paths)
        # print(self.tor_paths)

        for path in paths:
            if path not in self.tor_paths.keys():
                self.tor_paths[path] = False

        # Adding torrents from drive  to the list
        for tor, status in dict(self.tor_paths).items():
            if not status:
                self.tor_paths[tor] = self.add_torrent(tor)

    def check(self):
        # clear_output()
        cprint('\n' + ('-' * 60))
        for torrent in ses.get_torrents():
            s = torrent.status()

            if self.state_str[s.state] in ["Allocating", "Downloading Metadata"]:
                print(torrent.name(), 'is', self.state_str[s.state])
                continue

            if not torrent.has_metadata():
                continue

            print(torrent.name())
            cprint('{:.2f}% of {:.2f}GB ( down: {:.1f} mb/s  up: {:.1f} kB/s peers: {:d}) {} \n'.format(
                s.progress * 100, torrent.get_torrent_info().total_size() / (1024 * 1024 * 1024),
                s.download_rate / 1000000,
                s.upload_rate / 1000, s.num_peers, self.state_str[s.state]), 'magenta', 'on_white', attrs=['bold'])

            if (torrent.file_priority(0) == 0):
                ses.remove_torrent(torrent)
                print('Metadata downloaded for ', torrent.name())

            # If torrent is seeding, then move it to new location and remove it from session
            if (torrent.is_seed()):
                # print(torrent.save_path())

                # In case of only metadata downloading magnetic link, remove only torrent handle
                print('Checking !!')
                torrent_size = torrent.get_torrent_info().total_size() / (
                            1024 * 1024 * 1024)  # Get the size for total size
                torrent.move_storage(completed_path)
                torrent.force_recheck()
                print("Checking Done")

                # while not torrent.is_seed():
                os.remove(os.path.join(tor_path, torrent.name() + ".torrent"))
                # torrent_size = torrent.total_size() / (1024 * 1024 * 1024)
                self.current_torrent_size -= torrent_size
                ses.remove_torrent(torrent)
        # while (True):
        #     fast_check()
        #     print()
        #     if len(ses.get_torrents()) == 0:
        #         exit()
        #     time.sleep(60)

    ## Saves the torrent as .tor files
    def save_tor_file(self, torrent):
        # print(1, torrent.name(), torrent.has_metadata())
        if not torrent.has_metadata():
            return False

        if os.path.exists(os.path.join(tor_path, torrent.name() + ".torrent")): return True

        torrent_info = torrent.get_torrent_info()
        torrent_file = lt.create_torrent(torrent_info)
        torrent_path = os.path.join(tor_path, torrent.name() + ".torrent")
        with open(torrent_path, "wb") as f:
            f.write(lt.bencode(torrent_file.generate()))
        print('tor file created: ', torrent_path, '\n')
        ses.remove_torrent(torrent)
        print([t.name() for t in ses.get_torrents()])
        return True

    def gui_check(self):
        # clear_output()
        # cprint('\n'+('-' * 60))
        for torrent in ses.get_torrents():
            s = torrent.status()

            if self.state_str[s.state] in ["Allocating", "Downloading Metadata"]:
                print(torrent.name(), 'is', self.state_str[s.state])
                continue

            if not torrent.has_metadata():
                continue

            # tqdm progress bar
            if torrent in self.torrent_progress.keys():
                bar = self.torrent_progress[torrent]
                # bar.unpause()

                # bar.display(msg=torrent.name(), close=True)
                bar.write(torrent.name(), end='\n', nolock=False)

                bar.reset()
                bar.n = round(s.progress * torrent.get_torrent_info().total_size() / (1024 * 1024 * 1024), 2)
                bar.last_print_n = round(s.progress * torrent.get_torrent_info().total_size() / (1024 * 1024 * 1024), 2)
                bar.desc = " | ".join([torrent.name()[:40],
                                       self.state_str[s.state],
                                       ])

                d = {'down': str(round(s.download_rate / 1000000, 1)) + 'mb/s',
                     'up': str(round(s.upload_rate / 1000, 1)) + 'kb/s', 'peers': s.num_peers}
                bar.set_postfix(d)
                bar.refresh()
                # bar.pause()

            else:
                self.torrent_progress[torrent] = tqdm(
                    total=round(torrent.get_torrent_info().total_size() / (1024 * 1024 * 1024), 2),
                    dynamic_ncols=True,
                    unit='mb',
                    desc=" | ".join([torrent.name()[:40],
                                     self.state_str[s.state],
                                     ]),
                    postfix={}

                    )

            # print(self.torrent_progress)
            # print(torrent.name())
            # cprint('{:.2f}% of {:.2f}GB ( down: {:.1f} mb/s  up: {:.1f} kB/s peers: {:d}) {} \n'.format(
            #     s.progress * 100, torrent.get_torrent_info().total_size() / (1024 * 1024 * 1024),
            #     s.download_rate / 1000000,
            #     s.upload_rate / 1000, s.num_peers, self.state_str[s.state]), 'magenta', 'on_white',attrs=['bold'])

            if (torrent.file_priority(0) == 0):
                ses.remove_torrent(torrent)
                print('Metadata downloaded for ', torrent.name())

            # If torrent is seeding, then move it to new location and remove it from session
            if (torrent.is_seed()):
                # print(torrent.save_path())

                # In case of only metadata downloading magnetic link, remove only torrent handle
                print('Checking !!')
                torrent.move_storage(completed_path)
                torrent.force_recheck()
                print("Checking Done")

                if torrent.is_seed:
                    os.remove(os.path.join(tor_path, torrent.name() + ".torrent"))
                    ses.remove_torrent(torrent)

    def status_check(self):

        while True:
            # Regular checking and status printing
            self.check()

            # Load awaiting torrents
            if datetime.datetime.now().timestamp() > self.timestamp:
                # print('Part 1')
                t.load_torrents()
                # print(self.tor_paths)

            # Download tor files for awaiting torrents
            if len(self.tor_file_status) != 0:
                # print('Part 2')
                for torrent, status in dict(self.tor_file_status).items():
                    if status:
                        del self.tor_file_status[torrent]
                    else:
                        self.tor_file_status[torrent] = self.save_tor_file(torrent)
            time.sleep(200)
            if len(ses.get_torrents()) == 0:
                exit(0)


t = Torrent()
t.daemon = True
t.start()

# display(*download_bars)
t.status_check()
# t.magnets()

t.join()
exit()