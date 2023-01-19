# JAX model params serialization made easy!

1. TOC
{:toc}

## Introduction

From [`google/jax`](https://github.com/google/jax), _"JAX is Autograd and XLA, brought together
for high-performance machine learning research."_.

JAX tensors formatted as pytrees can be loaded using `numpy` and `pickle` to store the tree structure
in `pickle` and the tensors using `numpy`, but there's no unified way of doing so. Also, `pickle` is not
safe, that's why, among other multiple reasons, HuggingFace created [`huggingface/safetensors`](https://github.com/huggingface/safetensors).

The only JAX framework that contains a built-in serialization format is [`google/flax`](https://github.com/google/flax) which uses [`MessagePack`](https://flax.readthedocs.io/en/latest/api_reference/flax.serialization.html#serialization-with-messagepack)
and [`State Dict`](https://flax.readthedocs.io/en/latest/api_reference/flax.serialization.html#state-dicts).

But it also contains some drawbacks such as no layout control to enable lazy loading, which is useful in
distributed environments.

So on, `safetensors` is a complete and unified format for storing tensors for `torch`, `jax`/`flax`,
and `tensorflow`. See the table below from [`huggingface/safetensors`](https://github.com/huggingface/safetensors/blob/main/README.md):

| Format                  | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16
| ----------------------- | --- | --- | --- | --- | --- | --- | --- |
| pickle (PyTorch)        | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| H5 (Tensorflow)         | ✅ | ❌ | ✅ | ✅ | ~ | ~ | ❌ |
| SavedModel (Tensorflow) | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| MsgPack (flax)          | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Protobuf (ONNX)         | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Cap'n'Proto             | ✅ | ✅ | ~ | ✅ | ✅ | ~ | ❌ |
| Arrow                   | ? | ? | ? | ? | ? | ? | ❌ |
| Numpy (npy,npz)         | ✅ | ? | ? | ❌ | ✅ | ❌ | ❌ |
| SafeTensors             | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

That's the main reason why I decided to create [`safejax`](https://github.com/alvarobartt/safejax), to
easily provide a Python package to serialize and deserialize JAX (Flax, Haiku, and Objax) model params
using `safetensors` as the tensor-storage format.

## Why `safejax`?

`jax` uses `pytrees` to store the model parameters in memory, so
it's a dictionary-like class containing nested `jnp.DeviceArray` tensors.

`dm-haiku` uses a custom dictionary formatted as `<level_1>/~/<level_2>`, where the
levels are the ones that define the tree structure and `/~/` is the separator between those
e.g. `res_net50/~/intial_conv`, and that key does not contain a `jnp.DeviceArray`, but a 
dictionary with key-value pairs e.g. for both weights as `w` and biases as `b`.

`objax` defines a custom dictionary-like class named `VarCollection` that contains
some variables inheriting from `BaseVar` which is another custom `objax` type.

`flax` defines a dictionary-like class named `FrozenDict` that is used to
store the tensors in memory, it can be dumped either into `bytes` in `MessagePack`
format or as a `state_dict`.

So the motivation to create `safejax` is to easily provide a way to serialize `FrozenDict`, `VarCollection`,
and `Dict[str, jnp.DeviceArray]` using `safetensors` as the tensor storage format instead of 
`pickle`, as well as to provide a common and easy way to serialize and deserialize.

## Usage

* Convert `params` to `bytes`:

```python
from safejax import serialize, deserialize

encoded_bytes = serialize(params)
decoded_params = deserialize(encoded_bytes)
```

* Convert `params` to `bytes` in `params.safetensors` file

```python
from safejax import serialize, deserialize

encoded_bytes = serialize(params, filename="./params.safetensors")
decoded_params = deserialize("./params.safetensors")
```

There are also some framework-specific functions:

* `from safejax.flax import serialize, deserialize`
* `from safejax.objax import serialize, deserialize`
* `from safejax.haiku import serialize, deserialize`

Those functions handle the specific cases where the input is not at `Dict[str, jnp.DeviceArray]`, 
but a `VarCollection` in `objax` or a `FrozenDict` in `flax`.

## More information

More information can be found at [`alvarobartt/safejax`](https://github.com/alvarobartt/safejax).
