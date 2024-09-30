class NNModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: Tuple[int, int],
        pooling_size: int,
        device: torch.device
    ):
        super(NNModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            device=device,
            dtype=torch.float32
        )

        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            device=device,
            dtype=torch.float32
        )

        self.conv3 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            device=device,
            dtype=torch.float32
        )

        self.maxpooling = nn.MaxPool2d(
            kernel_size=pooling_size,
            stride=pooling_size
        )

        self.flat = nn.Flatten()
        self.relu = nn.ReLU()

        self.dense_layer = nn.Linear(
            in_features=hidden_dim * 32 * 32,
            out_features=hidden_dim,
            device=device,
            dtype=torch.float32
        )

        self.output_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            device=device,
            dtype=torch.float32
        )


    def initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.flat(x)
        x = self.dense_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)

        return x