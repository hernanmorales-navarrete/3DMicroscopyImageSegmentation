from tensorflow.keras import Model, Layer
from tensorflow.keras.layers import Conv3D, ReLU, MaxPool3D, UpSampling3D, Input, Concatenate, GroupNormalization, Conv3DTranspose, GlobalAveragePooling3D, Dense, Reshape, Activation, Dot, Add, Multiply

class UNet3D:
    def Encoder(self, x, out_channels, isMaxPool):
        if isMaxPool: 
            x = MaxPool3D(2, padding='same')(x)
            
        x = Conv3D(out_channels, 3, padding='same')(x)
        x = ReLU()(x)
        x = Conv3D(out_channels, 3, padding='same')(x)
        x = ReLU()(x)
        return x

    def Decoder(self, x, encoder_input, out_channels):
        x = UpSampling3D(2)(x)
        x = Concatenate(axis=4)([x, encoder_input])
        x = self.Encoder(x, out_channels, False)
        return x

    def build_model(self, input_size=(64, 64, 64, 1)):
        x = Input(input_size)

        encoder1 = self.Encoder(x, 64, False)
        encoder2 = self.Encoder(encoder1, 128, True)
        encoder3 = self.Encoder(encoder2, 256, True)
        encoder4 = self.Encoder(encoder3, 512, True)
        encoder5 = self.Encoder(encoder4, 1024, True)

        decoder4 = self.Decoder(encoder5, encoder4, 512)
        decoder3 = self.Decoder(decoder4, encoder3, 256)
        decoder2 = self.Decoder(decoder3, encoder2, 128)
        decoder1 = self.Decoder(decoder2, encoder1, 64)

        output = Conv3D(1, 1, activation='sigmoid')(decoder1)

        return Model(x, output)