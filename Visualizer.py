import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import CA


test = CA.SmoothLife((1500, 1500))
test.add_speckles()

while True:
    plt.imshow(test.step())
    plt.pause(0.1)
# img = []
# for i in range(500):
#     img.append(test.step())
#     print(i)
#
# i = 0
# while True:
#     plt.imshow(img[i])
#     plt.pause(0.05)
#     i += 1
#     if i == 500:
#         i = 0

# while True:
#     ax.clear()
#     ax.imshow(cellWorld.export_states())
#     ax.set_title(f"Generation {gen}")
#     plt.pause(0.001)
#     gen += 1
#     print(np.max(cellWorld.export_states()))
#     cellWorld.update_cells()


# fig, ax = plt.subplots()
# data = np.random.random((50, 50, 50))
#
# lst = []
#
# for i in range(500):
#     lst.append(cell_map.export_states())
#     cell_map.update_cells()
#     print(i)
#
# i = 0
# while True:
#     ax.clear()
#     ax.imshow(lst[i], cmap='seismic')
#     ax.set_title(f"frame {i}")
#     plt.pause(0.001)
#     i += 1
#     if i == 500:
#         i = 0

