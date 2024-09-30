def binary_crossentropy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
    ) -> torch.Tensor:

    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

def multiclass_crossentropy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
    ) -> torch.Tensor:

    return -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=1))

def plot_gallery(title, images, n_col: Optional[int] = 3, n_row: Optional[int] = 2, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape((64, 64)),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()