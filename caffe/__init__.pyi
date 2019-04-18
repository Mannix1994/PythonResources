# generate form caffe-master/python/caffe/_caffe.cpp

from typing import Tuple, Optional, Union, overload, Any, Callable, Generator

from . import io
from .pycaffe import _Net_backward, _Net_forward, _Net_forward_all
from .pycaffe import _Net_forward_backward_all, _Net_set_input_arrays
from .pycaffe import _Net_blobs, _Net_blob_loss_weights, _Net_layer_dict
from .pycaffe import _Net_params, _Net_batch, _Net_inputs, _Net_outputs, _Net_get_id_name
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from .net_spec import layers, params, NetSpec, to_proto

# -------------help(caffe)-----------------
__version__ = '1.0.0'

# FUNCTIONS
def init_log(level: Optional[int]=None, stderr: Optional[bool]=None) -> None: ...

def log(info : str) -> None: ...

def has_nccl() -> bool: ...

def set_mode_cpu() -> None: ...

def set_mode_gpu() -> None: ...

def set_random_seed(seed: int) -> None:  ...

def set_device(device_id: int) -> None: ...

def solver_count() -> int: ...

def set_solver_count(val: int) -> None: ...

def solver_rank() -> int: ...

def set_solver_rank(val: int) -> None: ...

def set_multiprocess(val: bool) -> None: ...

# print(list(caffe.layer_type_list()))
def layer_type_list() -> StringVec: ...

def get_solver(filename: str) -> Solver: ...
# .........................................


class StringVec: ...

# -------------help(caffe.Net)-------------
class Net:
    @overload
    def __init__(self, network_file: str, phase: int, level: int, stages: list(str) , weights: str): ...

    @overload
    def __init__(self, network_file: str, weights: str,  phase: int): ...

    def reshape(self) -> None : ...
    """
    @brief Reshape all layers from bottom to top.
    
    This is useful to propagate changes to layer sizes without running
    a forward pass, e.g. to compute output feature size.
    """

    def clear_param_diffs(self) -> None : ...

    def copy_from(self, param: str, ) -> None : ...

    def share_with(self, other: Net) -> None : ...

    @property
    def layers(self): ...

    def save(self, filename: str) -> None : ...

    def load_hdf5(self, filename: str) -> None : ...

    def save_hdf5(self, filename: str) -> None : ...

    def before_forward(self, callback: Callable) -> None : ...

    def after_forward(self, callback: Callable) -> None : ...

    def before_backward(self, callback: Callable) -> None : ...

    @overload
    def after_backward(self, callback: Callable) -> None : ...

    @overload
    def after_backward(self, nccl: NCCL) -> None : ...


    backward = _Net_backward(self, diffs=None, start=None, end=None, **kwargs)
    """
    Backward pass: prepare diffs and run the net backward.
      
      Parameters
        ----------
        diffs : list of diffs to return in addition to bottom diffs.
        kwargs : Keys are output blob names and values are diff ndarrays.
                If None, top diffs are taken from forward loss.
        start : optional name of layer at which to begin the backward pass
        end : optional name of layer at which to finish the backward pass
            (inclusive)
        
        Returns
        -------
        outs: {blob name: diff ndarray} dict.
    """

    forward = _Net_forward(self, blobs=None, start=None, end=None, **kwargs)
    """
    Forward pass: prepare inputs and run the net forward.

        Parameters
        ----------
        blobs : list of blobs to return in addition to output blobs.
        kwargs : Keys are input blob names and values are blob ndarrays.
                 For formatting inputs for Caffe, see Net.preprocess().
                 If None, input is taken from data layers.
        start : optional name of layer at which to begin the forward pass
        end : optional name of layer at which to finish the forward pass
              (inclusive)
    
        Returns
        -------
        outs : {blob name: blob ndarray} dict.
    """

    forward_all = _Net_forward_all(self, blobs=None, **kwargs)
    """
    Run net forward in batches.

        Parameters
        ----------
        blobs : list of blobs to extract as in forward()
        kwargs : Keys are input blob names and values are blob ndarrays.
                 Refer to forward().
    
        Returns
        -------
        all_outs : {blob name: list of blobs} dict.
    """

    forward_backward_all = _Net_forward_backward_all(self, blobs=None, diffs=None, **kwargs)
    """
    Run net forward + backward in batches.

        Parameters
        ----------
        blobs: list of blobs to extract as in forward()
        diffs: list of diffs to extract as in backward()
        kwargs: Keys are input (for forward) and output (for backward) blob names
                and values are ndarrays. Refer to forward() and backward().
                Prefilled variants are called for lack of input or output blobs.
    
        Returns
        -------
        all_blobs: {blob name: blob ndarray} dict.
        all_diffs: {blob name: diff ndarray} dict.
    """

    set_input_arrays = _Net_set_input_arrays(self, data, labels)
    """
    Set input arrays of the in-memory MemoryDataLayer.
    (Note: this is only for networks declared with the memory data layer.)
    """


    # Attach methods to Net.
    blobs = _Net_blobs
    blob_loss_weights = _Net_blob_loss_weights
    layer_dict = _Net_layer_dict
    params = _Net_params
    _batch = _Net_batch
    inputs = _Net_inputs
    outputs = _Net_outputs
    top_names = _Net_get_id_name(Net._top_ids, "_top_names")
    bottom_names = _Net_get_id_name(Net._bottom_ids, "_bottom_names")
# .........................................


# -------------Solver----------------------
class Solver:
    def __init__(self, solver_parameter: str) -> None : ...
    def add_callback(self, arg2, arg3) -> None : ...
    def apply_update(self) -> None : ...
    def restore(self, resume_file: str) -> None : ...
    def share_weights(self, net: Net) -> None : ...
    def snapshot(self) -> None :...
    def solve(self, resume_file: str) -> None : ...
    def step(self, iters: int) -> None : ...
    @property
    def iter(self) -> int: ...
    @property
    def net(self)-> Net: ...
    @property
    def param(self)-> Any: ...
    @property
    def test_nets(self)-> Net: ...
# .........................................

# -------------help(caffe.SGDSolver)-------
class SGDSolver(Solver):
    @property
    def lr(self) -> float: ...
# .........................................


# -------help(caffe.NesterovSolver)-------
class NesterovSolver(SGDSolver):
    @property
    def lr(self) -> float: ...
# .........................................

# -------help(caffe.AdaGradSolver)--------
class AdaGradSolver(SGDSolver):
    @property
    def lr(self) -> float: ...
# .........................................


# -------help(caffe.RMSPropSolver)--------
class RMSPropSolver(SGDSolver):
    @property
    def lr(self) -> float: ...
# .........................................

# -------help(caffe.AdaDeltaSolver)--------
class AdaDeltaSolver(SGDSolver):
    @property
    def lr(self) -> float: ...
# .........................................


# -------help(caffe.AdamSolver)--------
class AdamSolver(SGDSolver):
    @property
    def lr(self) -> float: ...
# .........................................


# -------help(caffe.NCCL)--------
class NCCL:
    def __init__(self, solver: Solver, uid: str): ...
# .........................................


# -------help(caffe.Timer)--------
class Timer:
    def __init__(self): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def ms(self) -> None: ...
# .........................................