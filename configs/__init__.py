"""配置模块 - Configuration Module

此模块包含所有实验的配置类，使用 dataclass 定义。
This module contains all experiment configuration classes defined using dataclass.
"""

from .args import BaseArgs, AtariPPOArgs, CartPolePPOArgs

__all__ = ['BaseArgs', 'AtariPPOArgs', 'CartPolePPOArgs']
