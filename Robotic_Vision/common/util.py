import cv2
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os.path import join
import pandas as pd
import time
import plotly.graph_objects as go
from IPython.core.display import display, HTML, Image


class Timer:
    def __init__(self, msg='Time elapsed'):
        self.msg = msg
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print(f'{self.msg}: {duration:.2f}s')


class Event:
    __slots__ = 't', 'x', 'y', 'p'
    def __init__(self, t, x, y, p):
        self.t = t
        self.x = x
        self.y = y
        self.p = p
    def __repr__(self):
        return f'Event(t={self.t:.3f}, x={self.x}, y={self.y}, p={self.p})'


def normalize_image(image, percentile_lower=1, percentile_upper=99):
    mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)


class EventData:
    def __init__(self, event_list, width, height):
        self.event_list = event_list
        self.width = width
        self.height = height

    def add_frame_data(self, data_folder, max_frames=100):
        timestamps = np.genfromtxt(join(data_folder, 'image_timestamps.txt'), max_rows=int(max_frames))
        frames = []
        frame_timestamps = []
        with open(join(data_folder, 'image_timestamps.txt')) as f:
            for line in f:
                fname, timestamp = line.split(' ')
                timestamp = float(timestamp)
                frame = cv2.imread(join(data_folder, fname), cv2.IMREAD_GRAYSCALE)
                if not (frame.shape[0] == self.height and frame.shape[1] == self.width):
                    continue
                frames.append(frame)
                frame_timestamps.append(timestamp)
                if timestamp >= self.event_list[-1].t:
                    break
        frame_stack = normalize_image(np.stack(frames, axis=0))
        self.frames = [f for f in frame_stack]
        self.frame_timestamps = frame_timestamps


def animate(images, fig_title=''):
    fig = plt.figure(figsize=(0.1, 0.1))  # don't take up room initially
    fig.suptitle(fig_title)
    fig.set_size_inches(7.2, 5.4, forward=False)  # resize but don't update gui
    ims = []
    for image in images:
        im = plt.imshow(normalize_image(image), cmap='gray', vmin=0, vmax=1, animated=True)
        ims.append([im])
    ani = ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    plt.close(ani._fig)
    return HTML(ani.to_html5_video())


def load_events(path_to_events, n_events=None):
    print('Loading events...')
    header = pd.read_csv(path_to_events, delim_whitespace=True, names=['width', 'height'],
                         dtype={'width': np.int, 'height': np.int}, nrows=1)
    width, height = header.values[0]
    print(f'width, height: {width}, {height}')
    event_pd = pd.read_csv(path_to_events, delim_whitespace=True, header=None,
                              names=['t', 'x', 'y', 'p'],
                              dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int8},
                              engine='c', skiprows=1, nrows=n_events, memory_map=True)
    event_list = []
    for event in event_pd.values:
        t, x, y, p = event
        event_list.append(Event(t, int(x), int(y), -1 if p < 0.5 else 1))
    print('Loaded {:.2f}M events'.format(len(event_list) / 1e6))
    return EventData(event_list, width, height)

def plot_3d(event_data, n_events=-1):
    x, y, t, c = [], [], [], []
    for e in event_data.event_list[:int(n_events)]:
        x.append(e.x)
        y.append(e.y)
        t.append(e.t * 1e3)
        c.append('rgb(255,0,0)' if e.p == 1 else 'rgb(0,0,255)')
    fig = go.Figure(data=[go.Scatter3d(x=t, y=x, z=y, 
                                       mode='markers',
                                       marker=dict(
                                           size=2,
                                           color=c,                # set color to an array/list of desired values
                                           opacity=0.8
                                    ))])

    fig.update_layout(scene = dict(
                    xaxis_title='Time (ms)',
                    yaxis_title='X',
                    zaxis_title='Y'))
    fig.update_yaxes(autorange="reversed")

    fig.show()

def event_slice(event_data, start=0, duration_ms=30):
    events, height, width = event_data.event_list, event_data.height, event_data.width
    mask = np.zeros((height, width), dtype=np.int8)
    start_idx = int(start * (len(events) - 1))
    end_time = events[start_idx].t + duration_ms / 1000.0
    for e in events[start_idx:]:
        mask[e.y, e.x] = e.p
        if e.t >= end_time:
            break
    img_rgb = np.ones((height, width, 3), dtype=np.uint8) * 255
    img_rgb[mask == -1] = (255, 0, 0)
    img_rgb[mask == 1] = (0, 0, 255)
    fig = plt.figure(figsize=(7.2, 5.4))
    plt.imshow(img_rgb)
