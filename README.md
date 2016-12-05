# Deep CNN

The classic deep convolution neural network project with TensorFlow. You can learn and run CNN easily.

* Train/test/inference images included.
* Visualize images with `matplotlib`.
* Support checkpoint and `tensorboard`.
* Easy to extend more convolution layers.

## Train

```
./pokemon_classifer.py --epoch_number 100
```

## Inference

```
./pokemon_classifer.py --mode inference --image ./data/inference/Pikachu.png
```

## Export model

```
./pokemon_classifer.py --epoch_number 0
```

## Run TensorFlow serving

```
./tensorflow_model_server --port=9000 --model_name=deep_cnn --model_base_path=./model
```

## Run gRPC client

```
./predict_client.py --host 127.0.0.1 --port 9000 --model_name deep_cnn --model_version 1
```

Notice that `cloudml` is not released now.

```
cloudml models predict -n deep_cnn -s 127.0.0.1:9000 -f ./data.json
```
