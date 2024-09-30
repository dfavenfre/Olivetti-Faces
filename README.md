# Olivetti-Faces

This repository contains a custom Convolutional Neural Network (CNN) model implemented using PyTorch, designed for image classification using the Olivetti-Faces dataset. The dataset contains 400 grayscale images of 40 unique individuals (10 images per individual), making it a perfect dataset for facial recognition and classification tasks.


## Dataset: Olivetti-Faces
The Olivetti-Faces dataset includes:

![images](https://github.com/user-attachments/assets/756315d9-5618-4308-9893-0afe351198be)

Number of Classes: 40 individuals
Total Images: 400 (10 images per individual)
Image Size: 64x64 pixels (grayscale)
Challenge: The dataset contains variations in facial expressions, lighting conditions, and angles.

# Model Architecture

The CNN model is built from scratch using PyTorch and follows a standard architecture with 3 convolutional layers, ReLU activations, max-pooling, and fully connected layers. The model was trained on a Google Colab A100 Tesla GPU (CUDA-enabled), and the training took approximately 3.5 hours.

Key model layers:

* Three Convolutional Layers with increasing hidden dimensions.
* Max Pooling Layer for spatial downsampling.
* Fully Connected Layers to produce class predictions.
* ReLU Activations to introduce non-linearity.

```Python
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
```

## Preprocessing
Dataset was separated as inner & outer dataset as 90% / 10%, respectively, and of 75% inner dataset was used for training, and the remaining 25% was used for inner validation. Outer 10% was used for a later evaluation to test whether results with unseen
data check out with trained results.


## Training Results

The model was trained for a total of 3.5 hours, and the following graphs were generated using Weights and Biases to monitor the training progress:
Navigate to the following link to see the [training results](https://api.wandb.ai/links/alone-wolf/imr7dvvu)

* Epoch: 256
* Learning Rate: 1e-5
* Validation Accuracy: 0.9333

![Training Loss](https://github.com/user-attachments/assets/dc9e3d07-c2ab-4fa8-9054-93b9e5db11e5)
![Validation Loss](https://github.com/user-attachments/assets/37ee469e-a023-40ca-a6ca-5de18042c1eb)
![Validation Accuracy](https://github.com/user-attachments/assets/6e0668b1-10b6-43ad-b5fe-c36aea6969bb)
![Hyperparameter Importance](https://github.com/user-attachments/assets/d5dd2356-cb3f-4aca-8d80-dcae2e58723a)

### Download Training Results As PDF
[Olivetti Faces Dataset CNN Model Training With PyTorch.pdf](https://github.com/user-attachments/files/17186331/Olivetti.Faces.Dataset.CNN.Model.Training.With.PyTorch.pdf)

## Evaluation on Unseen Data
![image](https://github.com/user-attachments/assets/6cc913cf-ab5e-45bc-a8b8-99e6713b1f44)


