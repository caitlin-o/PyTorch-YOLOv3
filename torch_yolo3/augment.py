import torch


def horisontal_flip(image, targets):
    """horizontal flip image and bboxes

    >>> img = torch.zeros((3, 5, 10))
    >>> img[:, :2, 5:] = 1
    >>> bboxs = torch.tensor([[0.0000, 1.0000, 0.6833, 0.2500, 0.0948, 0.1302],
    ...                       [0.0000, 1.0000, 0.7604, 0.2188, 0.0531, 0.0760]])
    >>> img, bboxs = horisontal_flip(img, bboxs)
    >>> img[0, ...]
    tensor([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    >>> bboxs
    tensor([[0.0000, 1.0000, 0.3167, 0.2500, 0.0948, 0.1302],
            [0.0000, 1.0000, 0.2396, 0.2188, 0.0531, 0.0760]])
    """
    image = torch.flip(image, [-1])
    if targets is not None:
        targets[:, 2] = 1 - targets[:, 2]
    return image, targets


def vertical_flip(image, targets):
    """vertical flip image and bboxes

    >>> img = torch.zeros((3, 5, 10))
    >>> img[:, :2, 5:] = 1
    >>> bboxs = torch.tensor([[0.0000, 1.0000, 0.6833, 0.2500, 0.0948, 0.1302],
    ...                       [0.0000, 1.0000, 0.7604, 0.2188, 0.0531, 0.0760]])
    >>> img, bboxs = vertical_flip(img, bboxs)
    >>> img[0, ...]
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]])
    >>> bboxs
    tensor([[0.0000, 1.0000, 0.6833, 0.7500, 0.0948, 0.1302],
            [0.0000, 1.0000, 0.7604, 0.7812, 0.0531, 0.0760]])
    """
    image = torch.flip(image, [-2])
    if targets is not None:
        targets[:, 3] = 1 - targets[:, 3]
    return image, targets
