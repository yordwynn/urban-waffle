import torch.nn as nn
from torch import Tensor


class ConvNet(nn.Module):
    """[ConvUnit] x conv_num -> [LinearUnit] x linear_num -> [softmax]"""

    def __init__(self, im_size: tuple, conv_params: list, linear_params: list) -> None:
        """
        ConvNet initialization.
        :param im_size: ``tuple`` within width, height and number of channels for input image
        :param conv_params: ``list`` of convolutional units parameters. Each parameter is a ``tuple`` within following
        parameters: num_filters, conv_kernel, pooling_kernel. See ``ConvUnit`` class.
        :param linear_params: ``list`` of linear layer parameters. Each parameter is a ``int`` witch is a number of
        features at each layer. There is no need to specify if the linear unit will apply non-linearity,
        it will be set automatically. The last features size should be equal to the number of classes.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        im_width, im_height, channels = im_size

        for num_filters, conv_kernel, pooling_kernel in conv_params:
            self.layers.append(ConvUnit(channels, num_filters, conv_kernel, pooling_kernel))
            channels = num_filters  # input size for the next layer
            im_width, im_height = im_width // pooling_kernel, im_height // pooling_kernel

        self.layers.append(Flatten())

        in_features = im_height * im_width * channels
        for hidden_size in linear_params[:-1]:
            self.layers.append(LinearUnit(in_features, hidden_size))
            in_features = hidden_size

        hidden_size = linear_params[-1]
        self.layers.append(LinearUnit(in_features, hidden_size, none_linearity=False))

    def forward(self, input: Tensor) -> Tensor:
        scores = input

        for layer in self.layers:
            scores = layer(scores)

        return scores


class ConvUnit(nn.Module):
    """[conv - batch norm - LeakyReLU - conv - batch norm - LeakyReLU - max pool] nn unit"""

    def __init__(self, in_channels: int, num_filters: int, conv_kernel: int, pooling_kernel: int) \
            -> None:
        """
        ConvUnit initialization
        :param in_channels: number of in_channels for the first conv layer
        :param num_filters: number of out_channels for the first conv layer and in_channels for the second conv layer
        :param conv_kernel:  size of the convolving kernel
        :param pooling_kernel: the size of the window to take a max over
        """
        super().__init__()
        modules = [
            nn.Conv2d(in_channels, num_filters, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.MaxPool2d(pooling_kernel)
        ]

        self.nn = nn.Sequential(*modules)
        self.nn.apply(self.__conv_initialize)

    def forward(self, input: Tensor) -> Tensor:
        return self.nn(input)

    @staticmethod
    def __conv_initialize(layer: nn.Module) -> None:
        """Apply Kaiming initialization for Conv layers"""
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight)


class LinearUnit(nn.Module):
    """[linear - batch norm - ReLU] nn unit"""

    def __init__(self, in_features: int, out_features: int, none_linearity=True) -> None:
        """
        LinearUnit initialization
        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param none_linearity: if include ReLU non-linear function after linear transformations
        """
        super().__init__()
        modules = [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features)
        ]

        if none_linearity:
            modules.append(nn.ReLU())

        self.nn = nn.Sequential(*modules)

    def forward(self, input: Tensor) -> Tensor:
        return self.nn(input)

    @staticmethod
    def __conv_initialize(layer: nn.Module) -> None:
        """Apply Kaiming initialization for Linear layers"""
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)


class Flatten(nn.Module):
    """Layer for flatten operation"""
    def forward(self, input: Tensor) -> Tensor:
        n = input.shape[0]
        return input.view(n, -1)
