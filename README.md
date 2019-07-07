# Introduction  
Some resources for Pytorch

## pytorch/\_\_init\_\_.pyi

`pytorch/__init__.pyi` is the file to solve the problem:
there is no code hints for Pytorch in Pycharm.

Steps:
* Upgrade Pycharm to 2019.1.1
* Download `pytorch/pytorch_version/__init__.pyi`
* Upgrade Pytorch to [pytorch_version] (the `__init__.pyi` in this project
is modified from the `__init__.pyi` of Pytorch [pytorch_version])
* Replace `site-packages/torch/__init__.pyi` with download one.


## caffe/\_\_init\_\_.pyi

`caffe/__init__.pyi` is the file to solve the problem:
there is no code hints for caffe in Pycharm.

Steps:
* Upgrade Pycharm to 2019.1.1
* Download `caffe/__init__.pyi`
* Copy `caffe/__init__.pyi` to pycaffe directory.
  * If you already built caffe,
  copy `caffe/__init__.pyi` to `path_to_caffe/python/caffe`.
  * If not, build caffe by guiding doc: 
  [How to build caffe](https://github.com/Mannix1994/SfSNet-Python/blob/master/build-caffe.md)
  , then copy `caffe/__init__.pyi` to `/opt/caffe/python/caffe`.
 
* Remove all `*.pyc` file in `path_to_caffe/python/caffe`.
  ```bash
  cd path_to_caffe/python/caffe
  rm *.pyc
  ```