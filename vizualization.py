import matplotlib.pyplot as plt
import numpy as np

def visualize_depth_map(samples, test=False, model=None,size=6):
    input, target = samples
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    if test:
        pred = model.predict(input)
        fig, ax = plt.subplots(size, 3, figsize=(50, 50))
        for i in range(size):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
            ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)

    else:
        fig, ax = plt.subplots(size, 2, figsize=(50, 50))
        for i in range(size):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)

    plt.show()


def vizualize_point_cloud(input, target):
    img_plot = plt.imshow(input.squeeze())
    plt.show()

    depth_vis = np.flipud(target.squeeze())
    img_vis = np.flipud(input.squeeze())

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")

    STEP = 3
    for x in range(0, img_vis.shape[0], STEP):
        for y in range(0, img_vis.shape[1], STEP):
            ax.scatter(
                [depth_vis[x, y]] * 3,
                [y] * 3,
                [x] * 3,
                c=tuple(img_vis[x, y, :3] / 255),
                s=3,
            )
        ax.view_init(45, 135)

    plt.show()