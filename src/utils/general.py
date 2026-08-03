import functools
import torch

def device_grad_decorator(device=None, no_grad=False):
    """
    A decorator to handle PyTorch tensor device allocation and gradient computation.

    Args:
        device (str or torch.device, optional): The device to which all tensor arguments should be moved.
                                                Examples: 'cpu', 'cuda', torch.device('cuda').
        no_grad (bool, optional): If True, the function will be executed within a torch.no_grad() context, 
                                  disabling gradient calculations. Defaults to False.

    Returns:
        Callable: A decorated function with tensor arguments moved to the specified device and
                  optionally executed without gradient tracking.

    Usage:
        @device_grad_decorator(device='cuda', no_grad=True)
        def some_function(tensor):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Move positional tensor arguments to the specified device
            if device:
                args = tuple(
                    arg.to(device) if isinstance(arg, torch.Tensor) else arg 
                    for arg in args
                )
                # Move keyword tensor arguments to the specified device
                kwargs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                }
            # Execute the function in no_grad context if specified
            if no_grad:
                with torch.no_grad():
                    return func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper
    return decorator

def visualize_img(ax, img, title, is_mask=False):
    """
    Displays an image on the given matplotlib axis with an optional grayscale colormap.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The axis on which to display the image.
        img (numpy.ndarray): The image data to be displayed.
        title (str): The title for the image.
        is_mask (bool, optional): If True, display the image in grayscale. Defaults to False.
    """
    # Keep the plotting dependency out of model/runtime imports. The helper's
    # caller supplies the matplotlib axis, so no module-level pyplot import is
    # required for training or inference consumers.
    ax.set_title(title)
    ax.imshow(img, cmap='gray' if is_mask else None)
    ax.axis('off')
