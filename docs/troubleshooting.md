## Troubleshooting


### Error: tensorflow.python.framework.errors_impl.NotFoundError: No algorithm worked! [Op:Conv2D]

```
(env) kc@kc-desktop:~/ishihara/workspace/multitask-with-attention$ python gradcam.py
2022-04-30 15:59:42.531480: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2022-04-30 15:59:43.610535: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2022-04-30 15:59:43.611048: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2022-04-30 15:59:43.638290: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.638562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: NVIDIA GeForce RTX 3080 computeCapability: 8.6
coreClock: 1.71GHz coreCount: 68 deviceMemorySize: 9.77GiB deviceMemoryBandwidth: 707.88GiB/s
2022-04-30 15:59:43.638580: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2022-04-30 15:59:43.640087: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2022-04-30 15:59:43.640120: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2022-04-30 15:59:43.640621: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2022-04-30 15:59:43.640752: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2022-04-30 15:59:43.642338: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2022-04-30 15:59:43.642717: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2022-04-30 15:59:43.642817: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2022-04-30 15:59:43.642877: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.643159: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.643424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2022-04-30 15:59:43.644549: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2022-04-30 15:59:43.644646: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.644900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: NVIDIA GeForce RTX 3080 computeCapability: 8.6
coreClock: 1.71GHz coreCount: 68 deviceMemorySize: 9.77GiB deviceMemoryBandwidth: 707.88GiB/s
2022-04-30 15:59:43.644914: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2022-04-30 15:59:43.644927: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2022-04-30 15:59:43.644939: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2022-04-30 15:59:43.644949: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2022-04-30 15:59:43.644960: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2022-04-30 15:59:43.644971: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2022-04-30 15:59:43.644981: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2022-04-30 15:59:43.644992: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2022-04-30 15:59:43.645030: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.645308: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.645676: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2022-04-30 15:59:43.645704: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2022-04-30 15:59:43.955406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-04-30 15:59:43.955433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0
2022-04-30 15:59:43.955439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N
2022-04-30 15:59:43.955577: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.955862: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.956110: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-04-30 15:59:43.956340: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 8529 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce RTX 3080, pci bus id: 0000:01:00.0, compute capability: 8.6)
Model: "MTA"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_image (InputLayer)        [(None, 160, 384, 3) 0
__________________________________________________________________________________________________
encoder (Functional)            [(None, 40, 96, 64), 21302473    input_image[0][0]
__________________________________________________________________________________________________
depth_task_att_module (TaskAtte ((None, 20, 48, 128) 75047       encoder[0][0]
__________________________________________________________________________________________________
semantic_task_att_module (TaskA ((None, 20, 48, 128) 75047       encoder[0][0]
__________________________________________________________________________________________________
control_attention_module (TaskA ((None, 5, 12, 512), 35379       encoder[0][3]
__________________________________________________________________________________________________
input_speed (InputLayer)        [(None, 1)]          0
__________________________________________________________________________________________________
depth_task_att_module_1 (TaskAt ((None, 10, 24, 256) 298475      encoder[0][1]
                                                                 depth_task_att_module[0][0]
__________________________________________________________________________________________________
semantic_task_att_module_1 (Tas ((None, 10, 24, 256) 298475      encoder[0][1]
                                                                 semantic_task_att_module[0][0]
__________________________________________________________________________________________________
flatten_module (Functional)     (None, 512)          2361856     control_attention_module[0][0]
__________________________________________________________________________________________________
speed_encoder (Functional)      (None, 64)           4800        input_speed[0][0]
__________________________________________________________________________________________________
depth_task_att_module_2 (TaskAt ((None, 5, 12, 512), 1190691     encoder[0][2]
                                                                 depth_task_att_module_1[0][0]
__________________________________________________________________________________________________
semantic_task_att_module_2 (Tas ((None, 5, 12, 512), 1190691     encoder[0][2]
                                                                 semantic_task_att_module_1[0][0]
__________________________________________________________________________________________________
latent_fusion_layer (Concatenat (None, 576)          0           flatten_module[0][0]
                                                                 speed_encoder[0][0]
__________________________________________________________________________________________________
input_nav_cmd (InputLayer)      [(None, 4)]          0
__________________________________________________________________________________________________
depth_task_att_module_3 (TaskAt ((None, 5, 12, 512), 35379       encoder[0][3]
                                                                 depth_task_att_module_2[0][0]
__________________________________________________________________________________________________
semantic_task_att_module_3 (Tas ((None, 5, 12, 512), 35379       encoder[0][3]
                                                                 semantic_task_att_module_2[0][0]
__________________________________________________________________________________________________
tl_attention_module (TaskAttent ((None, 5, 12, 512), 35379       encoder[0][3]
__________________________________________________________________________________________________
driving_module_branched (Functi {'steer': (None, 1), 1359644     latent_fusion_layer[0][0]
                                                                 input_nav_cmd[0][0]
__________________________________________________________________________________________________
depthnet (Functional)           (None, 160, 384, 1)  1680357     depth_task_att_module_3[0][0]
__________________________________________________________________________________________________
segnet (Functional)             (None, 160, 384, 13) 2036973     semantic_task_att_module_3[0][0]
__________________________________________________________________________________________________
tl_classifier (Functional)      (None, 4)            83716       tl_attention_module[0][0]
==================================================================================================
Total params: 32,099,761
Trainable params: 32,061,497
Non-trainable params: 38,264
__________________________________________________________________________________________________
forward pass
2022-04-30 15:59:46.619533: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2022-04-30 15:59:47.399429: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2022-04-30 15:59:47.401314: E tensorflow/stream_executor/cuda/cuda_blas.cc:226] failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
2022-04-30 15:59:47.401605: W tensorflow/core/framework/op_kernel.cc:1763] OP_REQUIRES failed at conv_ops.cc:1106 : Not found: No algorithm worked!
Traceback (most recent call last):
  File "gradcam.py", line 38, in <module>
    main(args)
  File "gradcam.py", line 25, in main
    output = model(*dummy)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer.py", line 1012, in __call__
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/model/mta.py", line 89, in call
    outputs = self.model([input_images, input_nav_cmd, input_speed], training=training)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer.py", line 1012, in __call__
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/functional.py", line 425, in call
    inputs, training=training, mask=mask)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/functional.py", line 560, in _run_internal_graph
    outputs = node.layer(*args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer.py", line 1012, in __call__
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/functional.py", line 425, in call
    inputs, training=training, mask=mask)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/functional.py", line 560, in _run_internal_graph
    outputs = node.layer(*args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer.py", line 1012, in __call__
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/keras/layers/convolutional.py", line 248, in call
    outputs = self._convolution_op(inputs, self.kernel)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/util/dispatch.py", line 201, in wrapper
    return target(*args, **kwargs)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/ops/nn_ops.py", line 1020, in convolution_v2
    name=name)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/ops/nn_ops.py", line 1150, in convolution_internal
    name=name)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/ops/nn_ops.py", line 2604, in _conv2d_expanded_batch
    name=name)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/ops/gen_nn_ops.py", line 932, in conv2d
    _ops.raise_from_not_ok_status(e, name)
  File "/home/kc/ishihara/workspace/multitask-with-attention/env/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 6862, in raise_from_not_ok_status
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.NotFoundError: No algorithm worked! [Op:Conv2D]
```

In my case, setting an environmental variable fixed it:
```
$ export TF_FORCE_GPU_ALLOW_GROWTH=true
```


## Pyenv >= 2.x seems to throw segmentation faults when installing python packages through pip
In this case, we recommend using pyenv 1.2.x as described in [readme.md](../README.md)
