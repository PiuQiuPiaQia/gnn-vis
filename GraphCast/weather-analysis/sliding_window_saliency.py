#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
滑动窗口梯度分析模块 (Sliding Window Saliency Analysis)
============================================================================

模块简介:
---------
本模块实现基于滑动窗口的 GraphCast 梯度分析，用于计算相邻时间点之间的
因果影响关系。与原始的固定输入梯度分析不同，本模块使用滚动窗口方法，
使每个预测都基于其直接前驱时间点进行梯度回溯。

原理说明:
---------
【原始方法】固定输入梯度分析:
    固定输入(00Z+06Z) → 12Z预测 → 18Z预测 → 次日00Z预测
                         ↑所有梯度都回溯到这里
    
    问题: 所有预测的梯度都回溯到初始输入，无法分析时间局部的因果关系

【本模块方法】滑动窗口梯度分析:
    窗口1: 00Z+06Z → 12Z预测 (梯度: 00Z/06Z → 12Z)
    窗口2: 06Z+12Z → 18Z预测 (梯度: 06Z/12Z → 18Z)  
    窗口3: 12Z+18Z → 次日00Z (梯度: 12Z/18Z → 次日00Z)
           ↑每次用前两个真实时间点作为新输入

    优点:
    1. 时间局部性: 分析相邻时间点的因果影响
    2. 动态追踪: 跟随台风移动路径分析每步的驱动因素
    3. 物理解释性: 更接近"当前状态如何影响下一状态"的因果关系
    4. 误差分析: 可以观察预测误差如何在时间上累积

数据要求:
---------
1. eval_inputs: 初始两个时间点的观测数据 (例如: 00Z, 06Z)
2. eval_targets: 后续时间点的真实观测数据 (例如: 12Z, 18Z, 次日00Z...)
3. eval_forcings: 各时间点的强迫项数据

使用方法:
---------
```python
from sliding_window_saliency import SlidingWindowSaliencyAnalyzer

# 初始化分析器
analyzer = SlidingWindowSaliencyAnalyzer(
    model_forward_fn=run_forward_jitted,
    task_config=task_config
)

# 计算滑动窗口梯度
results = analyzer.compute_sliding_window_gradients(
    eval_inputs=eval_inputs,
    eval_targets=eval_targets,
    eval_forcings=eval_forcings,
    cyclone_centers=CYCLONE_CENTERS,
    target_variable='geopotential',
    target_level=500
)
```

作者: AI Assistant
创建日期: 2026-01-11
版本: 1.0.0
依赖: jax, xarray, numpy, haiku

参考文献:
---------
[1] Lam et al. (2023). Learning skillful medium-range global weather forecasting.
    Science, 382(6677), 1416-1421.
[2] Simonyan et al. (2014). Deep Inside Convolutional Networks: Visualising
    Image Classification Models and Saliency Maps. ICLR.
============================================================================
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from xarray import Dataset, DataArray

# 添加 graphcast 源码路径（用于独立运行时）
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
PREPROCESS_DIR = SCRIPT_DIR.parent / "graphcast-preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

try:
    from graphcast import xarray_jax
    from latlon_utils import latlon_to_index
except ImportError:
    print("警告: 无法导入 graphcast 模块，请确保路径配置正确")


# ============================================================================
# 数据类定义
# ============================================================================

@dataclasses.dataclass
class SlidingWindowConfig:
    """滑动窗口配置"""
    window_size: int = 2  # 输入窗口大小（GraphCast 固定为2）
    step_size: int = 1    # 滑动步长
    target_variable: str = 'geopotential'  # 目标变量
    target_level: int = 500  # 目标气压层 (hPa)
    negative_gradient: bool = True  # 是否计算负梯度
    grid_resolution: float = 1.0  # 数据网格分辨率（度）


@dataclasses.dataclass
class GradientResult:
    """单个时间窗口的梯度计算结果"""
    window_idx: int  # 窗口索引
    input_times: List[str]  # 输入时间点列表 (例如: ['06Z', '12Z'])
    target_time: str  # 预测目标时间 (例如: '18Z')
    target_location: Tuple[float, float]  # 目标位置 (lat, lon)
    gradients: Dataset  # 梯度数据
    input_data: Dataset  # 用于该窗口的输入数据
    elapsed_time: float  # 计算耗时（秒）


# ============================================================================
# 滑动窗口梯度分析器
# ============================================================================

class SlidingWindowSaliencyAnalyzer:
    """
    滑动窗口梯度分析器
    
    使用滚动窗口方法计算 GraphCast 的输入梯度，
    实现相邻时间点之间的因果影响分析。
    """
    
    def __init__(
        self,
        model_forward_fn: Callable,
        task_config: Any,
        config: Optional[SlidingWindowConfig] = None
    ):
        """
        初始化分析器
        
        Args:
            model_forward_fn: JIT编译后的模型前向传播函数
            task_config: GraphCast 任务配置
            config: 滑动窗口配置，如果为None则使用默认配置
        """
        self.model_forward_fn = model_forward_fn
        self.task_config = task_config
        self.config = config or SlidingWindowConfig()
        
    def _create_single_step_targets_template(
        self,
        reference_targets: Dataset
    ) -> Dataset:
        """
        创建单步预测的目标模板
        
        Args:
            reference_targets: 参考目标数据（用于获取维度信息）
            
        Returns:
            单步预测的目标模板（填充NaN）
        """
        # 只保留第一个时间步
        single_step_targets = reference_targets.isel(time=slice(0, 1))
        # 填充 NaN
        return single_step_targets * np.nan
    
    def _create_single_step_forcings(
        self,
        full_forcings: Dataset,
        time_idx: int
    ) -> Dataset:
        """
        提取单步预测的强迫项
        
        Args:
            full_forcings: 完整的强迫项数据
            time_idx: 目标时间索引
            
        Returns:
            单步预测的强迫项
        """
        if 'time' in full_forcings.dims:
            # 确保索引不越界
            actual_idx = min(time_idx, len(full_forcings.time) - 1)
            return full_forcings.isel(time=slice(actual_idx, actual_idx + 1))
        return full_forcings
    
    def _construct_rolling_inputs(
        self,
        eval_inputs: Dataset,
        eval_targets: Dataset,
        window_idx: int
    ) -> Dataset:
        """
        构造滚动窗口的输入数据
        
        核心逻辑:
        - 将 eval_inputs 和 eval_targets 合并成完整时间序列
        - 从统一数组中切片取出连续的2个时间点
        
        时间序列示例:
        全序列: [00Z, 06Z, 12Z, 18Z, 次日00Z]
               [inp0, inp1, tgt0, tgt1, tgt2]
        
        - window_idx=0: [00Z, 06Z] → 预测 12Z
        - window_idx=1: [06Z, 12Z] → 预测 18Z
        - window_idx=2: [12Z, 18Z] → 预测 次日00Z
        
        Args:
            eval_inputs: 原始输入数据 (包含2个时间点: 00Z, 06Z)
            eval_targets: 目标数据 (包含多个时间点: 12Z, 18Z, ...)
            window_idx: 窗口索引
            
        Returns:
            该窗口的输入数据 (2个时间点)
        """
        # 方法1: 如果是第一个窗口，直接返回原始输入（避免拼接）
        if window_idx == 0:
            return eval_inputs
        
        # 方法2: 合并 inputs 和 targets 成完整时间序列
        # 完整序列: [inp[0], inp[1], tgt[0], tgt[1], tgt[2], ...]
        full_timeseries = xarray.concat([eval_inputs, eval_targets], dim='time')
        
        # 从完整序列中切片取窗口
        # window_idx=1: 需要索引 [1, 2] = [06Z, 12Z]
        # window_idx=2: 需要索引 [2, 3] = [12Z, 18Z]
        start_idx = window_idx
        end_idx = window_idx + 2
        
        rolling_inputs = full_timeseries.isel(time=slice(start_idx, end_idx))
        
        return rolling_inputs
    
    def _compute_single_window_gradient(
        self,
        inputs: Dataset,
        targets_template: Dataset,
        forcings: Dataset,
        target_idx: Tuple[int, int],
        target_time_idx: int = 0
    ) -> Dataset:
        """
        计算单个窗口的梯度
        
        Args:
            inputs: 输入数据 (2个时间点)
            targets_template: 目标模板 (填充NaN)
            forcings: 强迫项
            target_idx: 目标点的网格索引 (lat_idx, lon_idx)
            target_time_idx: 预测时间步索引（对于单步预测始终为0）
            
        Returns:
            输入梯度
        """
        lat_idx, lon_idx = target_idx
        config = self.config
        
        def target_loss(inputs_data):
            # 前向传播
            outputs = self.model_forward_fn(
                rng=jax.random.PRNGKey(0),
                inputs=inputs_data,
                targets_template=targets_template,
                forcings=forcings
            )
            
            # 提取目标变量
            target_data = outputs[config.target_variable]
            
            # 选择目标层次和位置
            if 'level' in target_data.dims:
                value = target_data.sel(level=config.target_level).isel(
                    time=target_time_idx, lat=lat_idx, lon=lon_idx
                )
            else:
                value = target_data.isel(
                    time=target_time_idx, lat=lat_idx, lon=lon_idx
                )
            
            if 'batch' in value.dims:
                value = value.isel(batch=0)
            
            # 转换为标量
            scalar = xarray_jax.unwrap_data(value, require_jax=True)
            scalar = jnp.squeeze(scalar)
            
            return -scalar if config.negative_gradient else scalar
        
        # 计算梯度
        grads = jax.grad(target_loss)(inputs)
        return grads
    
    def compute_sliding_window_gradients(
        self,
        eval_inputs: Dataset,
        eval_targets: Dataset,
        eval_forcings: Dataset,
        cyclone_centers: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[GradientResult]:
        """
        计算所有滑动窗口的梯度
        
        核心算法:
        对于每个目标时间点（从第3个时间点开始，因为前2个是输入）:
        1. 构造滚动窗口输入（前两个相邻时间点的真实观测）
        2. 计算从该输入到目标点的梯度
        3. 保存结果
        
        Args:
            eval_inputs: 原始输入数据 (2个时间点: 00Z, 06Z)
            eval_targets: 目标数据 (多个时间点: 12Z, 18Z, ...)
            eval_forcings: 强迫项数据
            cyclone_centers: 台风中心点列表，每个元素包含:
                - 'time': 时间标签
                - 'lat': 纬度
                - 'lon': 经度
                - 'is_input': 是否为输入时间点
                - 'target_time_idx': 目标时间索引（如果不是输入）
            verbose: 是否打印详细信息
            
        Returns:
            梯度结果列表
        """
        import time as time_module
        
        config = self.config
        results = []
        
        # 筛选需要计算梯度的时间点（跳过前两个输入时间点）
        prediction_centers = [c for c in cyclone_centers if not c.get('is_input', True)]
        
        if verbose:
            print("\n" + "=" * 70)
            print("【滑动窗口梯度分析】开始计算")
            print("=" * 70)
            print(f"输入时间点: 2 个 (来自 eval_inputs)")
            print(f"预测时间点: {len(prediction_centers)} 个")
            print(f"目标变量: {config.target_variable} @ {config.target_level}hPa")
            print(f"负梯度: {config.negative_gradient}")
        
        for window_idx, cyclone in enumerate(prediction_centers):
            if verbose:
                print(f"\n【窗口 {window_idx + 1}/{len(prediction_centers)}】")
                print(f"  目标时间: {cyclone['time']}")
                print(f"  台风位置: ({cyclone['lat']:.4f}°, {cyclone['lon']:.4f}°)")
            
            # 1. 构造滚动窗口输入
            rolling_inputs = self._construct_rolling_inputs(
                eval_inputs, eval_targets, window_idx
            )
            
            # 记录输入时间点
            if window_idx == 0:
                input_times = ["00Z (-6h)", "06Z (0h)"]
            elif window_idx == 1:
                input_times = ["06Z (0h)", "12Z (+6h)"]
            else:
                # 根据 window_idx 推断时间标签
                prev_center = cyclone_centers[window_idx]  # 第一个时间点
                curr_center = cyclone_centers[window_idx + 1]  # 第二个时间点
                input_times = [
                    prev_center.get('time', f'T-{(2-window_idx)*6}h'),
                    curr_center.get('time', f'T-{(1-window_idx)*6}h')
                ]
            
            if verbose:
                print(f"  输入窗口: {input_times}")
            
            # 2. 创建单步目标模板和强迫项
            targets_template = self._create_single_step_targets_template(eval_targets)
            forcings = self._create_single_step_forcings(
                eval_forcings, cyclone.get('target_time_idx', 0)
            )
            
            # 3. 计算目标点索引
            target_lat_idx, target_lon_idx = latlon_to_index(
                lat=cyclone['lat'],
                lon=cyclone['lon'],
                resolution=config.grid_resolution,
                lat_min=-90.0,
                lon_min=0.0
            )
            
            if verbose:
                print(f"  网格索引: ({target_lat_idx}, {target_lon_idx})")
            
            # 4. 计算梯度
            start_time = time_module.time()
            
            grads = self._compute_single_window_gradient(
                inputs=rolling_inputs,
                targets_template=targets_template,
                forcings=forcings,
                target_idx=(target_lat_idx, target_lon_idx),
                target_time_idx=0  # 单步预测，始终为0
            )
            
            elapsed = time_module.time() - start_time
            
            if verbose:
                print(f"  ✓ 梯度计算完成 (耗时: {elapsed:.2f}s)")
            
            # 5. 保存结果
            result = GradientResult(
                window_idx=window_idx,
                input_times=input_times,
                target_time=cyclone['time'],
                target_location=(cyclone['lat'], cyclone['lon']),
                gradients=grads,
                input_data=rolling_inputs,
                elapsed_time=elapsed
            )
            results.append(result)
        
        if verbose:
            total_time = sum(r.elapsed_time for r in results)
            print("\n" + "=" * 70)
            print(f"✓ 所有窗口梯度计算完成！共 {len(results)} 个窗口")
            print(f"  总耗时: {total_time:.2f}s")
            print("=" * 70)
        
        return results


# ============================================================================
# 便捷函数
# ============================================================================

def compute_sliding_gradients(
    model_forward_fn: Callable,
    task_config: Any,
    eval_inputs: Dataset,
    eval_targets: Dataset,
    eval_forcings: Dataset,
    cyclone_centers: List[Dict[str, Any]],
    target_variable: str = 'geopotential',
    target_level: int = 500,
    negative_gradient: bool = True,
    grid_resolution: float = 1.0,
    verbose: bool = True
) -> List[GradientResult]:
    """
    便捷函数：计算滑动窗口梯度
    
    Args:
        model_forward_fn: JIT编译后的模型前向传播函数
        task_config: GraphCast 任务配置
        eval_inputs: 原始输入数据
        eval_targets: 目标数据
        eval_forcings: 强迫项数据
        cyclone_centers: 台风中心点列表
        target_variable: 目标变量名
        target_level: 目标气压层
        negative_gradient: 是否计算负梯度
        grid_resolution: 网格分辨率
        verbose: 是否打印详细信息
        
    Returns:
        梯度结果列表
    """
    config = SlidingWindowConfig(
        target_variable=target_variable,
        target_level=target_level,
        negative_gradient=negative_gradient,
        grid_resolution=grid_resolution
    )
    
    analyzer = SlidingWindowSaliencyAnalyzer(
        model_forward_fn=model_forward_fn,
        task_config=task_config,
        config=config
    )
    
    return analyzer.compute_sliding_window_gradients(
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        eval_forcings=eval_forcings,
        cyclone_centers=cyclone_centers,
        verbose=verbose
    )


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n使用示例见文档头部的代码块。")
    print("本模块需要配合 GraphCast 模型和数据使用。")
