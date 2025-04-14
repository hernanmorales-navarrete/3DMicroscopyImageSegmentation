from tensorflow.keras import Model, Layer
from tensorflow.keras.layers import (
    Conv3D,
    ReLU,
    MaxPool3D,
    UpSampling3D,
    Input,
    Concatenate,
    GroupNormalization,
    Conv3DTranspose,
    GlobalAveragePooling3D,
    Dense,
    Reshape,
    Activation,
    Dot,
    Add,
    Multiply,
)


class DoubleConv(Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.conv1 = Conv3D(self.out_channels, kernel_size=3, padding="same")
        self.conv2 = Conv3D(self.out_channels, kernel_size=3, padding="same")
        self.relu = ReLU()
        self.groupnorm = GroupNormalization(8)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.groupnorm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.groupnorm(x)

        return x


class Encoder(Layer):
    def __init__(self, maxpool, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.ismaxpool = maxpool
        self.out_channels = out_channels
        self.maxpool = MaxPool3D(2, padding="same")
        self.doubleconv = DoubleConv(self.out_channels)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        if self.ismaxpool:
            x = self.maxpool(inputs)
        else:
            x = inputs

        x = self.doubleconv(x)

        return x


class Decoder(Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.convtrans = Conv3DTranspose(self.out_channels, 2, 2)
        self.attention = Attention(1024, self.out_channels)
        self.doubleconv = DoubleConv(self.out_channels)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, encoder_input, inputs):
        x = self.convtrans(inputs)
        summation = encoder_input + x

        excitation = self.attention(summation)

        return self.doubleconv(excitation)


class Attention(Layer):
    def __init__(self, latent_size, channels, **kwargs):
        super().__init__(**kwargs)
        self.latent_size = latent_size
        self.channels = channels

        self.global_avg = GlobalAveragePooling3D()
        self.fc1 = Dense(self.latent_size)
        self.fc2 = Dense(self.channels)
        self.reshape = Reshape((1, 1, 1, self.channels))
        self.activation = Activation("sigmoid")
        self.multiply = Multiply()
        self.conv = Conv3D(1, 1)
        self.add = Add()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # Channel Attention Mechanism
        avg_pool = self.global_avg(inputs)
        fc1 = self.fc1(avg_pool)
        fc2 = self.fc2(fc1)
        reshape = self.reshape(fc2)
        channel_sigmoid = self.activation(reshape)
        channel_attention = self.multiply([inputs, channel_sigmoid])

        # Spatial Attention Mechanism
        spatial_conv = self.conv(inputs)
        spatial_sigmoid = self.activation(spatial_conv)
        spatial_attention = self.multiply([inputs, spatial_sigmoid])

        # Sum
        return self.add([inputs + channel_attention + spatial_attention])


class AttentionUNet3D:
    def build_model(self, input_shape=(64, 64, 64, 1)):
        input = Input(input_shape)

        encoder1 = Encoder(False, 64)(input)
        encoder2 = Encoder(True, 128)(encoder1)
        encoder3 = Encoder(True, 256)(encoder2)
        encoder4 = Encoder(True, 512)(encoder3)
        encoder5 = Encoder(True, 1024)(encoder4)

        decoder4 = Decoder(512)(encoder4, encoder5)
        decoder3 = Decoder(256)(encoder3, decoder4)
        decoder2 = Decoder(128)(encoder2, decoder3)
        decoder1 = Decoder(64)(encoder1, decoder2)

        final_conv = Conv3D(1, 1, activation="sigmoid")(decoder1)

        return Model(input, final_conv)
