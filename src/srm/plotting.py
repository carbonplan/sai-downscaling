import matplotlib.pyplot as plt

def plot_comparisons(obs, raw, ds1, ds2=None, bias='absolute'):
    fig, axarr = plt.subplots(figsize=(20,8), nrows=2, ncols=4)
    obs.plot(ax=axarr[0,0])
    raw.plot(ax=axarr[1,0])
    ds1.plot(ax=axarr[0,1])
    if bias=='absolute':
        (ds1-obs).plot(ax=axarr[1,1])
    elif bias=='percentage':
        (((ds1-obs)/obs)*100).plot(ax=axarr[1,1])
    if ds2 is not None:
        ds1.plot(ax=axarr[0,2])
        if bias=='absolute':
            (ds2-obs).plot(ax=axarr[1,2])
            (ds2-ds1).plot(ax=axarr[0,3])
        elif bias=='percentage':
            (((ds2-obs)/obs)*100).plot(ax=axarr[1,2])
            (((ds2-obs)/obs)*100).plot(ax=axarr[0,3])
    axarr[1,3].axis('off')
    plt.tight_layout()