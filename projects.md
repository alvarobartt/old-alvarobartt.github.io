# projects

## ml

* [`safejax`](https://github.com/alvarobartt/safejax) is a Python package to easily serialize JAX
(Flax, Haiku, or Objax) model params using `safetensors` as the tensor-storage format.

* [`serving-pytorch-models`](https://github.com/alvarobartt/serving-pytorch-models) is a detailed guide
on how to setup and deploy TorchServe to expose an inference API for a pre-trained PyTorch CNN.

* [`serving-tensorflow-models`](https://github.com/alvarobartt/serving-tensorflow-models) is a detailed
guide on how to setup and deploy TensorFlow Serving to expose an inference API for a pre-trained TensorFlow
CNN.

* [`tensorflow-serving-streamlit`](https://github.com/alvarobartt/tensorflow-serving-streamlit) is a `streamlit`
application to showcase how to deploy `tensorflow-serving` and send requests to it from a UI. It uses the
same model and configuration as [`serving-tensorflow-models`](https://github.com/alvarobartt/serving-tensorflow-models).

* [`understanding-resnet`](https://github.com/alvarobartt/understanding-resnet) is a PyTorch implementation
of ResNet V1 in PyTorch with ported weights from [`timm`](https://github.com/rwightman/pytorch-image-models)
and some notes I took for educational/learning purposes.

* [`ml-monitoring-with-wandb`](https://github.com/alvarobartt/ml-monitoring-with-wandb) is a detailed guide
on how to train a PyTorch model using PyTorch Lightning and monitoring it using Weights & Biases (wandb).

## finance

* [`investpy`](https://github.com/alvarobartt/investpy) is a Python package to retrieve data from
Investing.com. It used to support all the financial products available at Investing.com, but in
late 2022, Investing.com protected their APIs with Cloudflare V2, so sadly this is no longer working...

* [`investiny`](https://github.com/alvarobartt/investiny) is a simpler, faster, and better-coded version
of `investpy`, that I developed when the former package was blocked, and I got it working for some weeks,
but later it was also blocked. AFAIK it still works if you send a few requests a day, otherwise, you'll get
blacklisted and blocked from sending requests to Investing.com.

* [`trendet`](https://github.com/alvarobartt/) is a simple Python package to detect trends in time
series data. It works with `pandas` as well as with `investpy` as it's directly integrated with it. Note
that now it does not work with `investpy` anymore due to the blocking from Investing.com, but it does still
work with any time series in a `pandas.DataFrame`.

## templates

* [`python-package-template`](https://github.com/alvarobartt/python-package-template) is Python package
template using `pyproject.toml`, `hatch`, `pre-commit`, `black`, `ruff`, and `mkdocs`.

## others

* [`wandbfsspec`](https://github.com/alvarobartt/wandbfsspec) is an `fsspec` interface for Weights &
Biases (wandb) files and artifacts.
