import random as rn
import numpy as np
import math
import matplotlib.pyplot as plt


def make_circle(radius, size):
    x, y = size
    # Generates a grid of points, each with a value equal to their distance from the center of the grid
    xx, yy = np.mgrid[:y, :x]
    radiuses = np.sqrt((xx - x / 2) ** 2 + (yy - y / 2) ** 2)

    # Scales transition width logarithmically
    logres = math.log(min(*size), 2)

    # Generates circle with a log fadeout
    with np.errstate(over="ignore"):
        logistic = 1 / (1 + np.exp(logres * (radiuses - radius)))

    # The circle is positioned at the corners so that it is accessible at [0,0]
    # This will help when applying the circle mask to the entire field
    logistic = np.roll(logistic, y//2, axis=0)
    logistic = np.roll(logistic, x//2, axis=1)
    return logistic

    circleArray = np.zeros(((size * 2 + 1), (size * 2 + 1)))
    for i, row in enumerate((circleArray)):
        for j, element in enumerate(row):
            circleArray[i, j] = max(- (math.pow(abs(i - size), 2) + math.pow(abs(j - size), 2)
                                       - math.pow(radius, 2)) / math.pow(radius, 2), 0)
    # circleArray = np.zeros(((radius * 2 + 1), (radius * 2 + 1)))
    # for i, row in enumerate(circleArray):
    #     for j, element in enumerate(row):
    #         circleArray[i, j] = max(- (math.pow(abs(i - radius), 2) + math.pow(abs(j - radius), 2)
    #                                    - math.pow(radius, 2)) / math.pow(radius, 2), 0)
    return circleArray

class SmoothMath:
    def __init__(self):
        self.alphaN = 0.028
        self.alphaM = 0.147
        self.b1 = 0.278
        self.b2 = 0.365
        self.d1 = 0.267
        self.d2 = 0.445

    def lerp(self, a, b, t):
        return (1 - t) * a + t * b

    def sigma_1(self, x, a, alpha):
        temp = (x - a) * 4 / alpha
        return 1 / (1 + np.exp(-temp))

    def sigma_2(self, x, a, b, alpha):
        return self.sigma_1(x, a, alpha) * (1 - self.sigma_1(x, b, alpha))

    def sigma_m(self, x, y, m, alpha):
        return x * (1 - self.sigma_1(m, 0.5, alpha)) + y * self.sigma_1(m, 0.5, alpha)

    def s_function(self, n, m):
        return self.sigma_2(n, self.sigma_m(self.b1, self.b2, m, self.alphaM),
                            self.sigma_m(self.b1, self.b2, m, self.alphaM), self.alphaN)

class Multipliers:
    # When determining the values of cells and neighborhoods a kernal is needed
    # This class creates these kernals

    Inner_Radius = 7.0
    Outer_Radius = 3 * Inner_Radius

    def __init__(self, size, innerRadius=Inner_Radius, outerRadius=Outer_Radius):
        # Creates the initial masks
        cellMask = make_circle(innerRadius, size)
        neighborhoodMask = make_circle(outerRadius, size) - cellMask
        # Normalises Masks
        cellMask /= np.sum(cellMask)
        neighborhoodMask /= np.sum(neighborhoodMask)
        # Perform Fourier Transforms needed for fast convolutions
        self.M = np.fft.fft2(cellMask)
        self.N = np.fft.fft2(neighborhoodMask)

class SmoothLife:
    def __init__(self, size):
        # Get field size and generate field
        self.width = size[1]
        self.height = size[0]
        self.clear()
        # Initialize Multipliers
        self.multipliers = Multipliers(size)
        # Initialize Constants and math
        self.calc = SmoothMath()
        self.alphaN = 0.028
        self.alphaM = 0.147
        self.b1 = 0.278
        self.b2 = 0.365
        self.d1 = 0.267
        self.d2 = 0.445
        self.timeStep = 0.1

    def clear(self):
        self.field = np.zeros((self.height, self.width))

    def transition_function(self, n, m, field):
        # Determine the state of the cell
        # Create the transition function; the values for which a living cell will stay alive or die
        # And the values for which a dead cell will stay dead or come to life
        # Adjust state accordingly
        state = self.calc.sigma_1(m, 0.5, self.alphaM)
        threshold1 = self.calc.lerp(self.b1, self.d1, state)
        threshold2 = self.calc.lerp(self.b2, self.d2, state)
        newState = self.calc.sigma_2(n, threshold1, threshold2, self.alphaN)
        return np.clip(newState, 0, 1)

    def step(self):
        # Takes one timestep and returns the new field
        field = np.fft.fft2(self.field)
        mBuffer = field * self.multipliers.M
        nBuffer = field * self.multipliers.N
        mBuffer = np.real(np.fft.fft2(mBuffer))
        nBuffer = np.real(np.fft.fft2(nBuffer))
        # self.field = self.field + self.timeStep * (self.transition_function(nBuffer, mBuffer, self.field) - self.field)
        self.field = self.transition_function(nBuffer, mBuffer, self.field)
        return self.field

    def add_speckles(self, count=None, intensity=1):
        """Populate field with random living squares

        If count unspecified, do a moderately dense fill
        """
        if count is None:
            count = int(
                self.width * self.height / ((self.multipliers.Outer_Radius * 2) ** 2)
            )
        for i in range(count):
            radius = int(self.multipliers.Outer_Radius)
            r = np.random.randint(0, self.height - radius)
            c = np.random.randint(0, self.width - radius)
            self.field[r: r + radius, c: c + radius] = intensity


class CellArray:
    def __init__(self, size):
        self.cells = [[SmoothCell() for y in range(size[1])] for x in range(size[0])]
        self.neighborhoodMask = np.zeros((len(self.cells[0][0].neighborhoodMask), len(self.cells[0][0].neighborhoodMask[0])))

    def export_states(self):
        state_array = np.zeros((len(self.cells), len(self.cells[0])))
        for i, row in enumerate(self.cells):
            for j, element in enumerate(row):
                state_array[i, j] = element.state
        return state_array

    def generate_neighborhood(self, expanded_state_array, y_coord, x_coord):
        maskY = int((len(self.neighborhoodMask) - 1) / 2)
        maskX = int((len(self.neighborhoodMask[0]) - 1) / 2)
        neighborhoodSlice = expanded_state_array[(y_coord - maskY):(y_coord + maskY + 1),
                            (x_coord - maskX):(x_coord + maskX + 1)]
        return neighborhoodSlice

    def update_cells(self):
        state_array = self.export_states()
        update_array = np.append(state_array, state_array, axis=0)
        update_array = np.append(update_array, state_array, axis=0)
        update_array = np.append(update_array, update_array, axis=1)
        update_array = np.append(update_array[:, 0:len(state_array[0])], update_array, axis=1)
        for i in range(len(self.cells), len(self.cells) * 2):
            for j in range(len(self.cells[0]), len(self.cells[0]) * 2):
                array_slice = self.generate_neighborhood(update_array, i, j)
                self.cells[i - len(self.cells)][j - len(self.cells[0])].update(array_slice)
                # element.update(array_slice)


class CellConway:
    def __init__(self):
        self.state = math.floor(rn.random() * 2)

    def Update(self, neighbourhood):
        if self.state == 0 and np.sum(neighbourhood) == 3:
            self.state = 1
        elif self.state == 1 and (np.sum(neighbourhood) > 4 or np.sum(neighbourhood) < 3):
            self.state = 0


class Cell:
    def __init__(self):
        self.state = rn.random()
        self.movement = np.array([[0.5, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]])

    def Update(self, neighbourhood):
        self.state = np.sum(neighbourhood * self.movement) / 4
        self.state = min(1, self.state)
        self.movement = (self.movement - (neighbourhood - 0.5)*0.5)
        self.movement[1, 1] = 0
        for i in range(3):
            for j in range(3):
                self.movement[i, j] = max(self.movement[i, j], 0)
                self.movement[i, j] = min(self.movement[i, j], 1)
        return 0


class SmoothCell:
    def __init__(self):
        self.state = rn.random()
        self.cellMask = smoothCell
        self.cellNormalization = cellNorm
        self.neighborhoodMask = smoothNeighborhood
        self.maskNormalization = neighNorm
        self.alphaN = 0.028
        self.alphaM = 0.147
        self.b1 = 0.278
        self.b2 = 0.365
        self.d1 = 0.267
        self.d2 = 0.445
        self.stepsize = 1

    def update(self, neighborhood):
        n = np.sum(neighborhood * self.neighborhoodMask) / self.maskNormalization
        m = np.sum(neighborhood * self.cellMask) / self.cellNormalization
        dummy = self.s_function(n, m) * self.stepsize
        print(dummy)
        self.state += dummy * self.state
        return 0

    def sigma_1(self, x, a, alpha):
        temp = (x - a) * 4 / alpha
        return 1 / (1 + math.exp(-temp))

    def sigma_2(self, x, a, b, alpha):
        return self.sigma_1(x, a, alpha) * (1 - self.sigma_1(x, b, alpha))

    def sigma_m(self, x, y, m, alpha):
        return x * (1 - self.sigma_1(m,0.5, alpha)) + y * self.sigma_1(m, 0.5, alpha)

    def s_function(self, n, m):
        return self.sigma_2(n, self.sigma_m(self.b1, self.b2, m, self.alphaM),
                            self.sigma_m(self.b1, self.b2, m, self.alphaM), self.alphaN)