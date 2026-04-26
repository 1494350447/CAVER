# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from .meter_recorder import AvgMeter
from .msg_logger import MsgLogger
from .timer import CustomizedTimer, TimeRecoder

__all__ = ("AvgMeter", "CalTotalMetric", "MsgLogger", "CustomizedTimer", "TimeRecoder")


def __getattr__(name):
    if name == "CalTotalMetric":
        from .metric_caller import CalTotalMetric

        return CalTotalMetric
    raise AttributeError(f"module 'utils.recorder' has no attribute {name}")
