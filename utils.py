from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

def custom_plot(image, box=None, polygons=None):
    #target.extra_fields['masks'].polygons è una lista in cui ogni elemento ha un .polygons
    # che è una lista di tensor

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if box is not None:
        for bb in box:
            rect = Rectangle((int(bb[0]), int(bb[1])), int(bb[2]), int(bb[3]), linewidth=1, edgecolor='r'
                                     , facecolor='none')
            ax.add_patch(rect)

    if polygons is not None:
        patches = []
        for p in polygons:
            for p2 in p.polygons:
                polygon = Polygon(p2.numpy().reshape((-1, 2)), False)
                patches.append(polygon)
        p = PatchCollection(patches, alpha=0.4)
        p.set_linewidth(2.0)
        p.set_edgecolor('r')
        ax.add_collection(p)

    plt.show()
