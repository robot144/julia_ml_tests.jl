# julia_ml_tests.jl

Some testing with julia and machine learning

This is just a collection of introductory material. Most things are copies from other places or a combination of these. I have made an effort to point to the sources everywhere, but feel free to send a message when I omitted a pointer to other work. The aim here is not to create new things, but to have a more or less arbitrary introductory overview of Machine Learning in Julia. I learned much from [this online book](https://book.sciml.ai/) and the [Flux.jl online documentation](https://fluxml.ai/Flux.jl/stable/).

## Getting started

- `flux_elementary.ipynb` : contains basic data-structures and methods from Flux.jl, e.g a Dense network layer etc.
- `flux_tiny_example_cpu.ipynb` : contains the classical XOR problem from the [Flux.jl quickstart](https://fluxml.ai/Flux.jl/stable/models/quickstart/) There are two versions, one using cpu and one using gpu.
- `flux_mnist.ipynb` : based on the classical classification of handwritten digits.
- `flux_mnist_autoencoder.ipynb` : convolutional auto-encoder tested on the mnist images