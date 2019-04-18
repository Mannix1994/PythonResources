# Introduction  
Some resources for Pytorch

## `pytorch/__init__.pyi`

`pytorch/__init__.pyi` is the file to solve the problem:
there is no code hints for Pytorch in Pycharm.

Steps:
* Upgrade Pycharm to 2019.1.1
* Download `pytorch/__init__.pyi`
* Upgrade Pytorch to 1.0.1.post2 (the `__init__.pyi` in this project
is modified from the `__init__.pyi` of Pytorch 1.0.1.post2)
* Replace `site-packages/torch/__init__.pyi` with download one.


## `caffe/__init__.pyi`

`caffe/__init__.pyi` is the file to solve the problem:
there is no code hints for caffe in Pycharm.

Steps:
* Upgrade Pycharm to 2019.1.1
* Download `caffe/__init__.pyi`
* Build caffe by guiding doc: [How to build caffe](https://github.com/Mannix1994/SfSNet-Python/blob/master/build-caffe.md)
* Copy `caffe/__init__.pyi` to `path_to_caffe/python/caffe`. if you
build caffe using the guiding doc mentioned above, 
copy `caffe/__init__.pyi` to `/opt/caffe/python/caffe`.
* Remove all `*.pyc` file in `path_to_caffe/python/caffe`.
  ```bash
  cd path_to_caffe/python/caffe
  rm *.pyc
  ```