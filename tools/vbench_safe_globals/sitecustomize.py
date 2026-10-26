"""Lets VBench 0.1.5 load its AMT-S checkpoint under torch >= 2.6, where torch.load defaults to weights_only=True.

The checkpoint (vbench-cache/amt_model/amt-s.pth, keys epoch and state_dict) was saved with a typing.OrderedDict in it, which the
weights-only unpickler refuses by default. Only that one type is allowed here; weights_only stays on, so nothing else in a checkpoint can run code.
Python imports this module at start-up of every process that has this folder on PYTHONPATH, including the workers the vbench command launches.
"""
import typing

import torch

torch.serialization.add_safe_globals([typing.OrderedDict])
