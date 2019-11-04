import sys
import math
import random
import argparse
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


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
    def intensity(distance, strength):
        if distance == 0:
            # ideally, we need to account for the fact that
            # this value should be higher than when distance=1
            return strength
        return strength/(distance**2)

    @staticmethod
    def display(grid, cmap='bwr', text=False, figure=True):
        # cmap = 'bwr'       # [blue=0 : white=0.5 : red=1]
        # cmap = 'Blues'  # 0.0=white   1.0=blue
        if text:
            print(grid)
        if figure:
            fig, ax = plt.subplots()
            im = ax.imshow(grid, cmap=plt.get_cmap(cmap))
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
        ap.add_argument('-d', '--dimension', help='Side length of gridworld', default=50)
        ap.add_argument('-u', '--units', help='Number of units to generate', default=20)
        ap.add_argument('-f', '--figure', help='Whether to display plot of result', action='store_true', default=True)
        ap.add_argument('-a', '--ascii', help='Whether to print numpy array to terminal', action='store_true')
        ap.add_argument('-v', '--verbose', help='Control amount of printed output', action='store_true')
        ap.add_argument('-e', '--enemies', help='Generate both allies and enemies', action='store_true', default=True)

        args = ap.parse_args()
        return args


    def run(num_units, dimension, show_figure, show_array, enemies, verbose):
        cache = []
        for _ in range(num_units):
            cache.append(generate_encodings(dimension, include_enemies=enemies))

        print('[+] DIMENSION: {}\n[+] UNITS: {}'.format(num_units, dimension))
        if verbose: display_encodings(*cache)

        keys = list(range(num_units))
        objs = dict.fromkeys(keys)

        for n, values in enumerate(cache):
            objs[n] = Unit(*values, dimension=dimension)

        world = np.zeros((dimension, dimension))

        for key, unit in objs.items():
            unit.sphere = unit.influence(*unit.center(), unit.r)
            unit.build_map()
            world = world + unit.as_array()

        Unit.display(world, text=show_array, figure=show_figure)


    args = options()
    run(args.units, args.dimension, args.figure, args.ascii, args.enemies, args.verbose)
