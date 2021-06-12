# Swin Transformer (Tensorflow)
Tensorflow reimplementation of **Swin Transformer** model.   
  
Based on [Official Pytorch implementation](https://github.com/microsoft/Swin-Transformer).
![image](https://user-images.githubusercontent.com/24825165/121768619-038e6d80-cb9a-11eb-8cb7-daa827e7772b.png)

## Requirements
- `tensorflow >= 2.4.1`

## Pretrained Swin Transformer Checkpoints
**ImageNet-1K and ImageNet-22K Pretrained Checkpoints**  
| name | pretrain | resolution |acc@1 | #params | model |
| :---: | :---: | :---: | :---: | :---: | :---: |
|`swin_tiny_224` |ImageNet-1K |224x224|81.2|28M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_tiny_224.tgz)|
|`swin_small_224`|ImageNet-1K |224x224|83.2|50M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_small_224.tgz)|
|`swin_base_224` |ImageNet-22K|224x224|85.2|88M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_224.tgz)|
|`swin_base_384` |ImageNet-22K|384x384|86.4|88M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_384.tgz)|
|`swin_large_224`|ImageNet-22K|224x224|86.3|197M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_large_224.tgz)|
|`swin_large_384`|ImageNet-22K|384x384|87.3|197M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_large_384.tgz)|

## Examples
Initializing the model:
```python
from swintransformer import SwinTransformer

model = SwinTransformer('swin_tiny_224', num_classes=1000, include_top=True, pretrained=False)
```
You can use a pretrained model like this:
```python
import tensorflow as tf
from swintransformer import SwinTransformer

model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMAGE_SIZE, 3]),
  SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```
If you use a pretrained model with TPU on kaggle, specify `use_tpu` option:
```python
import tensorflow as tf
from swintransformer import SwinTransformer

model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMAGE_SIZE, 3]),
  SwinTransformer('swin_tiny_224', include_top=False, pretrained=True, use_tpu=True),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```
Example: [TPU training on Kaggle](https://www.kaggle.com/rishigami/tpu-swin-transformer-tensorflow)
## Citation
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
