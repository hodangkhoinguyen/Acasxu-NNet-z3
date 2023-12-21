## Initial idea
As I had created `my_symbolic_execution` function that takes `dnn` as `Sequential` before, I tried to convert `nnet` to `Sequential` format. So I can convert `Sequential` to symbolic states that can use `z3`.

So far, I have a first version of my `readNNet` to convert `nnet` to `Sequential`. I will need to double check again, but that's my first approach.

My next step is to verify if my `readNNet` run correctly. Also, I will try to convert `nnet` directly to symbolic states instead of going through `Sequential`.

## How to run
`python3 my_nnet.py`

## Reference
My function `readNNet` is referred from [this NNET repo](https://github.com/sisl/NNet/blob/master/utils/readNNet.py)
