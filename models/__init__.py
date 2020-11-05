from .networks import (
    resnet_convlstm,
    cyclegan_convlstm,
    cyclegan_convlstm_deconv1,
    resnet_conv3d,
    unet_conv2d,
    conv2d_deconv2d,
    cyclegan_conv2d,
    two_cyclegan_convlstm_classifier,
    unet_conv2d_instance_norm,
)

prediction_networks_dict = {
    'resnet_convlstm': resnet_convlstm,
    'cyclegan_convlstm': cyclegan_convlstm,
    'cyclegan_convlstm_deconv1': cyclegan_convlstm_deconv1,
    'resnet_conv3d': resnet_conv3d,
    'unet_conv2d': unet_conv2d,
    'conv2d_deconv2d': conv2d_deconv2d,
    'cyclegan_conv2d': cyclegan_conv2d,
    'two_cyclegan_convlstm_classifier': two_cyclegan_convlstm_classifier,
    'two_cyclegan_convlstm_focal_loss': two_cyclegan_convlstm_classifier,
    'unet_conv2d_instance_norm': unet_conv2d_instance_norm
}
