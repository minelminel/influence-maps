import sys
import math
import random
import argparse
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Unit:
    def __init__(self, x, y, r, i, dimension=100):
        self.x = x
        self.y = y
        self.r = r
        self.i = i
        self.dimension = dimension
        self.sphere = None
        self.cache = None
        self.array = None

    def center(self):
        return (self.x, self.y)

    def shape(self):
        if self.array:
            return self.array.shape
        elif self.dimension:
            return (self.dimension, self.dimension)

    def build_map(self):
        cache = set()
        center = (self.x, self.y)
        for point in self.sphere:
            d = self.point_distance(center, point)
            i = self.intensity(d, self.i)
            cache.add((*point, i))
        self.cache = cache

    def as_array(self):
        if not self.array:
            arr = np.zeros((self.dimension, self.dimension))
            for each in self.cache:
                x, y, i = each
                try:
                    arr[x, y] = i
                except IndexError:
                    pass
            self.array = arr
        return self.array

    @staticmethod
    def influence(x, y, r):
        # TODO: skip calculations for points beyond the grid dimensions
        neighborhood = set()
        X = int(r)
        for i in range(-X, X + 1):
            Y = int(pow(r * r - i * i, 1/2))
            for j in range(-Y, Y + 1):
                neighborhood.add((x + i, y + j))
        return neighborhood

    @staticmethod
    def point_distance(a, b):
        ax, ay = a
        bx, by = b
        hypotenuse = (ax - bx)**2 + (ay - by)**2
        return math.sqrt(hypotenuse)

    @staticmethod
    def intensity(distance, strength, algo='linear'):
        # ideally, we need to account for the fact that this value should be higher than when distance=1
        if distance == 0: return strength
        if algo == 'linear': return strength/distance
        elif algo == 'square': return strength/(distance**2)
        else: raise ValueError()

    @staticmethod
    def display(grid, cmap='seismic', text=False, figure=True):
        # cmap = ['bwr', 'seismic']
        if text:
            print(grid)
        if figure:
            l = max(abs(grid.min()), abs(grid.max()))
            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.15)
            im = ax.imshow(grid, cmap=plt.get_cmap(cmap), norm=colors.Normalize(vmin=-l, vmax=l))
            fig.colorbar(im, cax=cax, orientation='vertical')
            plt.show()


# === Runtime Functions ===
def generate_encodings(dimension, min_rad=0, max_rad=None, min_i=1, max_i=10, include_enemies=True):
    import random
    max_rad = dimension//2 or max_rad
    x = random.randint(0, dimension)
    y = random.randint(0, dimension)
    r = random.randint(min_rad, max_rad)
    i = random.randint(min_i, max_i)
    if include_enemies:
        i = i * random.choice([-1, 1])
    return x, y, r, i


def display_encodings(*cache):
    from colorama import init, Fore
    init(autoreset=True)
    for c in cache:
        enemy = True if c[-1]<0 else False
        string = 'x={}\ty={}\tradius={}\tstrength={}'.format(*c)
        if enemy:
            print(Fore.RED + string)
        else:
            print(Fore.CYAN + string)


if __name__ == '__main__':

    def options():
        ap = argparse.ArgumentParser(usage='python3 -m main.py [options]', description='Implementation of tactical influence maps')
        ap.add_argument('-d', '--dimension', help='Side length of gridworld', default=50, type=int)
        ap.add_argument('-u', '--units', help='Number of units to generate', default=20, type=int)
        ap.add_argument('-f', '--figure', help='Whether to display plot of result', action='store_true', default=True)
        ap.add_argument('-a', '--ascii', help='Whether to print numpy array to terminal', action='store_true')
        ap.add_argument('-v', '--verbose', help='Control amount of printed output', action='store_true')
        ap.add_argument('-e', '--enemies', help='Generate both allies and enemies', action='store_true', default=True)
        args = ap.parse_args()
        return args

    def config(args):
        return args.__dict__


    def run(units=10, dimension=50, figure=True, ascii=False, enemies=True, verbose=False):
        cache = []
        for _ in range(units):
            cache.append(generate_encodings(dimension, include_enemies=enemies))

        print('[+] DIMENSION: {}\n[+] UNITS: {}'.format(units, dimension))
        if verbose: display_encodings(*cache)

        keys = list(range(units))
        objs = dict.fromkeys(keys)

        for n, values in enumerate(cache):
            objs[n] = Unit(*values, dimension=dimension)

        world = np.zeros((dimension, dimension))

        for key, unit in objs.items():
            unit.sphere = unit.influence(*unit.center(), unit.r)
            unit.build_map()
            world = world + unit.as_array()

        Unit.display(world, text=ascii, figure=figure)


    args = options()
    cfg = config(args)
    run(**cfg)
