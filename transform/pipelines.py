from transform.pair_transform import PairedTransformForDimma






###加载预处理（训练/测试）
TRAIN_FOR_1000="trainfor1000"

def load_transforms(transform_config):
    print(type(transform_config))
    if transform_config.name == TRAIN_FOR_1000:
        transforms = (
            PairedTransformForDimma(
                flip_prob=transform_config.flip_prob,
                crop_size=transform_config.image_size,
            ),
            PairedTransformForDimma(test=True),
        )
    return transforms