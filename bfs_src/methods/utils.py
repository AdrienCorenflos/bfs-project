from collections import namedtuple

__all__ = ["MVNormalParameters"]

MVNormalParameters = namedtuple("MVNormalParameters", ["mean", "cov"])
