from .step import layer, step

import matplotlib.pyplot as plt


class DelayedCall:
    def __init__(self, fn):
        self.fn = fn
        self.args = None
        self.kwargs = None
        self.parent = None
        self.polar = False

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __getattr__(self, name):
        plt_attr = getattr(plt, name)
        if callable(plt_attr):
            chained = DelayedCall(plt_attr)
            chained.parent = self
            return chained
        return plt_attr

    def set_polar(self):
        self.polar = True
        return self

    def run(self):
        if self.parent:
            self.parent.run()
        try:
            return self.fn(*self.args, **self.kwargs)
        except:
            print("Error in delayed call")
            print("\tFunction:", self.fn)
            print("\tArguments:", self.args)
            print("\tKeyword Arguments:", self.kwargs)


class _lazy:
    def __getattr__(self, name):
        plt_attr = getattr(plt, name)
        if callable(plt_attr):
            return DelayedCall(plt_attr)
        return plt_attr


LazyPlot = _lazy()


def plot(figs, width=3, height=2, title=None):
    rows = len(figs)
    cols = max(len(r) for r in figs)
    fig = plt.figure(figsize=(width, height))
    for i, row in enumerate(figs):
        for j, plot in enumerate(row):
            kwargs = {"projection": "polar"} if plot.polar else {}
            plt.subplot(rows, cols, i * cols + j + 1, **kwargs)
            plot.run()

    fig.suptitle(f"{title} for layer {layer()}, step {step()}")
    plt.tight_layout()
    plt.show()
