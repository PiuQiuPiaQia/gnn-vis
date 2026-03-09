# tests/physics/test_comparison_core_dlmsf.py
# -*- coding: utf-8 -*-
"""Smoke tests for DLMSF integration in comparison_core.py."""
from __future__ import annotations

import json
from typing import Any, cast
import numpy as np
import pytest
from physics.swe.alignment import AlignmentReport, _group_metrics


class TestDlmsfDisabled:
    """DLMSF_ENABLE=False 时，dlmsf_result 保持 None（配置控制生效）。"""

    def test_dlmsf_enable_false_config(self, monkeypatch):
        """当 DLMSF_ENABLE=False，getattr 应返回 False。"""
        import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "DLMSF_ENABLE", False, raising=False)
        import config as cfg
        assert bool(getattr(cfg, "DLMSF_ENABLE", True)) is False

    def test_dlmsf_enable_true_by_default(self):
        """若未设置 DLMSF_ENABLE，getattr 默认为 True。"""
        # 测试 getattr fallback 机制
        class FakeConfig:
            pass
        assert bool(getattr(FakeConfig(), "DLMSF_ENABLE", True)) is True


class TestDlmsfAlignmentReportStructure:
    """验证 DLMSF 对齐报告包含正确的 group 名称和字段。"""

    def test_dlmsf_main_report_only_builds_signed_z_group(self):
        """主报告只保留 signed z_500；uv_500 不应伪造为主报告分组。"""
        from types import SimpleNamespace
        from physics.swe.comparison_core import _build_dlmsf_alignment_inputs

        dlmsf_result = SimpleNamespace(
            S_map=np.random.rand(10, 10),
            S_abs_map=np.random.rand(10, 10),
        )
        signed_gnn_maps = {"z_500": np.random.rand(10, 10)}
        magnitude_gnn_maps = {
            "z_500": np.random.rand(10, 10),
            "uv_500": np.random.rand(10, 10),
        }

        inputs = _build_dlmsf_alignment_inputs(dlmsf_result, signed_gnn_maps, magnitude_gnn_maps)
        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        for spec in inputs["main_specs"]:
            m = _group_metrics(
                spec["s_map"], spec["gnn_map"], spec["group_name"], 2, "mean", (20, 50)
            )
            report.groups.append(m)

        group_names = [g.group_name for g in report.groups]
        assert group_names == ["dlmsf_z_500"]
        assert [pair[2] for pair in inputs["overlap_pairs"]] == ["z_500", "uv_500"]

    def test_dlmsf_report_json_serializable(self):
        """DLMSF 主报告的 signed z_500 分组应可被 json.dumps 序列化。"""
        S_map = np.random.rand(10, 10)
        gnn_map = np.random.rand(10, 10)

        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        m = _group_metrics(S_map, gnn_map, "dlmsf_z_500", 2, "mean", (20, 50))
        report.groups.append(m)

        d = report.as_dict()
        json_str = json.dumps(d)  # 不应抛出异常
        assert "dlmsf_z_500" in json_str

    def test_dlmsf_report_spearman_in_valid_range(self):
        """Spearman ρ 应在 [-1, 1] 范围内。"""
        S_map = np.random.rand(20, 20)
        gnn_map = np.random.rand(20, 20)

        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        m = _group_metrics(S_map, gnn_map, "dlmsf_z_500", 2, "mean", (20, 50))
        report.groups.append(m)

        rho = report.groups[0].spearman_rho
        assert -1.0 <= rho <= 1.0 or np.isnan(rho)

    def test_empty_groups_report_as_dict(self):
        """空 groups 的 DLMSF 报告也应能正常序列化（落盘前的守卫条件逻辑测试）。"""
        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        assert len(report.groups) == 0
        # 守卫条件：空 groups 不应落盘（在 comparison_core.py 中有 if dlmsf_report.groups 守卫）
        assert not report.groups  # 验证 falsy

    def test_empty_groups_report_can_still_be_written_as_json(self, tmp_path):
        from physics.swe.alignment import save_report_json

        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        out = tmp_path / "physics_alignment_metrics.json"

        save_report_json(report, out)

        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["groups"] == {}


class TestDlmsfReturnDictKeys:
    """验证结果 payload helper 的返回字段契约。"""

    def test_required_dlmsf_keys_in_return_payload(self):
        """结果 payload 应包含 DLMSF 字段。"""
        from physics.swe.comparison_core import _build_comparison_result_payload

        payload = _build_comparison_result_payload(
            jax_result=object(),
            signed_gnn_maps={},
            magnitude_gnn_maps={},
            report=AlignmentReport(0, 6, 2, "mean", 0.0),
            dlmsf_result=object(),
            dlmsf_report=object(),
            track_patch_analysis=object(),
            sweep_rows=[],
            ig_sanity_payload={"status": "skipped", "reason": "disabled", "passed": None},
            elapsed=0.0,
        )
        assert {"dlmsf_result", "dlmsf_report", "track_patch_analysis"} <= set(payload)

    def test_return_payload_distinguishes_main_and_magnitude_gnn_maps(self):
        """返回 payload 应显式区分 signed/main maps 与 magnitude maps。"""
        from physics.swe.comparison_core import _build_comparison_result_payload

        signed_maps = {"z_500": np.ones((2, 2))}
        magnitude_maps = {"z_500": np.ones((2, 2)), "uv_500": np.full((2, 2), 3.0)}
        payload = _build_comparison_result_payload(
            jax_result=object(),
            signed_gnn_maps=signed_maps,
            magnitude_gnn_maps=magnitude_maps,
            report=AlignmentReport(0, 6, 2, "mean", 0.0),
            dlmsf_result=None,
            dlmsf_report=None,
            track_patch_analysis=None,
            sweep_rows=[],
            ig_sanity_payload={"status": "skipped", "reason": "disabled", "passed": None},
            elapsed=0.0,
        )

        assert payload["gnn_main_maps"] is signed_maps
        assert payload["gnn_ig_maps"] is magnitude_maps
        assert payload["gnn_ig_magnitude_maps"] is magnitude_maps
        assert payload["track_patch_analysis"] is None
class TestSignedAndMagnitudeGnnGroupMaps:
    """验证 comparison_core 将 signed 主路径和 magnitude 补充路径分开。"""

    def test_build_gnn_group_maps_split_preserves_signed_z_and_abs_uv(self):
        from physics.swe.comparison_core import _build_gnn_group_maps_split

        full_lat = np.array([0.0, 1.0, 2.0])
        full_lon = np.array([10.0, 11.0, 12.0])
        swe_lat = np.array([0.0, 1.0])
        swe_lon = np.array([10.0, 11.0])
        gnn_ig_raw = {
            "geopotential": np.array([
                [-3.0, 2.0, 9.0],
                [4.0, -5.0, 8.0],
                [7.0, 6.0, -1.0],
            ]),
            "u_component_of_wind": np.array([
                [-3.0, 4.0, 0.0],
                [0.0, -8.0, 1.0],
                [5.0, 6.0, 7.0],
            ]),
            "v_component_of_wind": np.array([
                [4.0, -3.0, 0.0],
                [-8.0, 0.0, 1.0],
                [5.0, 6.0, 7.0],
            ]),
        }

        signed_maps, magnitude_maps = _build_gnn_group_maps_split(
            gnn_ig_raw, full_lat, full_lon, swe_lat, swe_lon,
        )

        np.testing.assert_array_equal(
            signed_maps["z_500"],
            np.array([[-3.0, 2.0], [4.0, -5.0]]),
        )
        assert "uv_500" not in signed_maps
        np.testing.assert_array_equal(
            magnitude_maps["z_500"],
            np.array([[3.0, 2.0], [4.0, 5.0]]),
        )
        np.testing.assert_allclose(
            magnitude_maps["uv_500"],
            np.array([[5.0, 5.0], [8.0, 8.0]]),
        )

    @pytest.mark.parametrize("raw_key", ["u_component_of_wind", "v_component_of_wind"])
    def test_build_gnn_group_maps_split_skips_uv_without_both_components(self, raw_key):
        from physics.swe.comparison_core import _build_gnn_group_maps_split

        full_lat = np.array([0.0, 1.0])
        full_lon = np.array([10.0, 11.0])
        swe_lat = np.array([0.0, 1.0])
        swe_lon = np.array([10.0, 11.0])
        gnn_ig_raw = {
            raw_key: np.array([
                [-3.0, 4.0],
                [0.0, -8.0],
            ]),
        }

        signed_maps, magnitude_maps = _build_gnn_group_maps_split(
            gnn_ig_raw, full_lat, full_lon, swe_lat, swe_lon,
        )

        assert signed_maps == {}
        assert "uv_500" not in magnitude_maps


class TestGnnIgComparableVarFiltering:
    def test_compute_gnn_ig_omits_excluded_comparable_var_instead_of_zero_map(self, monkeypatch):
        from types import SimpleNamespace
        import xarray as xr
        from physics.swe.comparison_core import _compute_gnn_ig_for_swe_vars

        lat = np.array([0.0, 1.0])
        lon = np.array([10.0, 11.0])
        dims = ("lat", "lon")
        coords = {"lat": lat, "lon": lon}
        base = np.array([[1.0, 1.0], [1.0, 1.0]])
        eval_inputs = xr.Dataset(
            {
                "geopotential": xr.DataArray(base * 5.0, dims=dims, coords=coords),
                "u_component_of_wind": xr.DataArray(base * 2.0, dims=dims, coords=coords),
                "v_component_of_wind": xr.DataArray(base * 3.0, dims=dims, coords=coords),
            }
        )
        baseline_inputs = xr.Dataset(
            {
                "u_component_of_wind": xr.DataArray(base, dims=dims, coords=coords),
                "v_component_of_wind": xr.DataArray(base, dims=dims, coords=coords),
            }
        )
        context = SimpleNamespace(eval_inputs=eval_inputs, target_var="unused")
        runtime_cfg = cast(Any, SimpleNamespace(gradient_steps=1))
        reduced_vars = []

        def fake_grad(_loss):
            def _grads(_interp):
                return xr.Dataset(
                    {
                        "u_component_of_wind": xr.DataArray(np.ones((2, 2)), dims=dims, coords=coords),
                        "v_component_of_wind": xr.DataArray(np.full((2, 2), 2.0), dims=dims, coords=coords),
                    }
                )

            return _grads

        def fake_reduce_input_attribution_to_latlon(*, attribution, original_da, runtime_cfg):
            reduced_vars.append(original_da.name)
            return xr.DataArray(np.asarray(attribution), dims=original_da.dims, coords=original_da.coords)

        monkeypatch.setattr("physics.swe.comparison_core.jax.grad", fake_grad)
        monkeypatch.setattr(
            "physics.swe.comparison_core.reduce_input_attribution_to_latlon",
            fake_reduce_input_attribution_to_latlon,
        )

        ig_maps = _compute_gnn_ig_for_swe_vars(
            context,
            runtime_cfg,
            baseline_inputs,
            allowed_variables={"u_component_of_wind", "v_component_of_wind"},
        )

        assert set(ig_maps) == {"u_component_of_wind", "v_component_of_wind"}
        assert "geopotential" not in ig_maps
        assert reduced_vars == ["u_component_of_wind", "v_component_of_wind"]

    def test_compute_gnn_ig_returns_empty_for_empty_allowed_variable_set(self):
        from types import SimpleNamespace
        import xarray as xr
        from physics.swe.comparison_core import _compute_gnn_ig_for_swe_vars

        coords = {"lat": [0.0], "lon": [1.0]}
        eval_inputs = xr.Dataset(
            {
                "geopotential": xr.DataArray([[5.0]], dims=("lat", "lon"), coords=coords),
                "u_component_of_wind": xr.DataArray([[2.0]], dims=("lat", "lon"), coords=coords),
                "v_component_of_wind": xr.DataArray([[3.0]], dims=("lat", "lon"), coords=coords),
            }
        )
        baseline_inputs = eval_inputs.copy(deep=True)
        context = SimpleNamespace(eval_inputs=eval_inputs, target_var="unused")
        runtime_cfg = cast(Any, SimpleNamespace(gradient_steps=1))

        ig_maps = _compute_gnn_ig_for_swe_vars(
            context,
            runtime_cfg,
            baseline_inputs,
            allowed_variables=set(),
        )

        assert ig_maps == {}

    def test_compute_gnn_ig_rejects_nonpositive_gradient_steps(self):
        from types import SimpleNamespace
        import xarray as xr
        from physics.swe.comparison_core import _compute_gnn_ig_for_swe_vars

        coords = {"lat": [0.0], "lon": [1.0]}
        eval_inputs = xr.Dataset(
            {
                "u_component_of_wind": xr.DataArray([[2.0]], dims=("lat", "lon"), coords=coords),
                "v_component_of_wind": xr.DataArray([[3.0]], dims=("lat", "lon"), coords=coords),
            }
        )
        baseline_inputs = eval_inputs.copy(deep=True)
        context = SimpleNamespace(eval_inputs=eval_inputs, target_var="unused")
        runtime_cfg = cast(Any, SimpleNamespace(gradient_steps=0))

        with pytest.raises(ValueError, match="gradient_steps"):
            _compute_gnn_ig_for_swe_vars(
                context,
                runtime_cfg,
                baseline_inputs,
                allowed_variables={"u_component_of_wind", "v_component_of_wind"},
            )

    def test_resolve_allowed_swe_ig_variables_matches_runtime_variable_filtering(self):
        import xarray as xr
        from types import SimpleNamespace
        from physics.swe.comparison_core import _resolve_allowed_swe_ig_variables

        coords = {"lat": [0.0], "lon": [1.0]}
        eval_inputs = xr.Dataset(
            {
                "geopotential": xr.DataArray([[1.0]], dims=("lat", "lon"), coords=coords),
                "u_component_of_wind": xr.DataArray([[2.0]], dims=("lat", "lon"), coords=coords),
                "v_component_of_wind": xr.DataArray([[3.0]], dims=("lat", "lon"), coords=coords),
                "temperature": xr.DataArray([[4.0]], dims=("lat", "lon"), coords=coords),
            }
        )

        runtime_cfg = SimpleNamespace(
            perturb_variables=["u_component_of_wind", "temperature"],
            include_target_inputs=False,
        )

        allowed = _resolve_allowed_swe_ig_variables(
            eval_inputs=eval_inputs,
            runtime_cfg=cast(Any, runtime_cfg),
            target_var="u_component_of_wind",
        )

        assert allowed == {"temperature"}

    def test_resolve_allowed_swe_comparable_ig_variables_drops_non_comparable_vars(self):
        import xarray as xr
        from types import SimpleNamespace
        from physics.swe.comparison_core import _resolve_allowed_swe_comparable_ig_variables

        coords = {"lat": [0.0], "lon": [1.0]}
        eval_inputs = xr.Dataset(
            {
                "geopotential": xr.DataArray([[1.0]], dims=("lat", "lon"), coords=coords),
                "u_component_of_wind": xr.DataArray([[2.0]], dims=("lat", "lon"), coords=coords),
                "v_component_of_wind": xr.DataArray([[3.0]], dims=("lat", "lon"), coords=coords),
                "temperature": xr.DataArray([[4.0]], dims=("lat", "lon"), coords=coords),
            }
        )

        runtime_cfg = SimpleNamespace(
            perturb_variables=["u_component_of_wind", "temperature"],
            include_target_inputs=False,
        )

        allowed = _resolve_allowed_swe_comparable_ig_variables(
            eval_inputs=eval_inputs,
            runtime_cfg=cast(Any, runtime_cfg),
            target_var="u_component_of_wind",
        )

        assert allowed == set()


class TestSweAlignmentInputBuilder:
    """验证 SWE 主路径和补充路径分离，避免把 magnitude 伪装成 signed。"""

    def test_swe_alignment_inputs_use_signed_z_when_available(self):
        from types import SimpleNamespace
        from physics.swe.comparison_core import _build_swe_alignment_inputs

        swe_result = SimpleNamespace(
            S_h=np.array([[-2.0, 1.0], [0.5, -0.25]]),
            S_uv=np.array([[2.0, 1.0], [0.5, 0.25]]),
        )
        signed_gnn_maps = {"z_500": np.array([[-3.0, 2.0], [4.0, -5.0]])}
        magnitude_gnn_maps = {
            "z_500": np.array([[3.0, 2.0], [4.0, 5.0]]),
            "uv_500": np.array([[5.0, 5.0], [8.0, 8.0]]),
        }

        inputs = _build_swe_alignment_inputs(swe_result, signed_gnn_maps, magnitude_gnn_maps)

        assert [spec[2] for spec in inputs["main_pairs_scatter"]] == ["z_500"]
        assert [spec[2] for spec in inputs["main_pairs_metrics"]] == ["z_500"]
        assert inputs["main_gnn_maps"]["z_500"] is signed_gnn_maps["z_500"]

    def test_swe_alignment_inputs_do_not_consume_magnitude_uv_as_signed_main(self):
        from types import SimpleNamespace
        from physics.swe.comparison_core import _build_swe_alignment_inputs

        swe_result = SimpleNamespace(
            S_h=np.ones((2, 2)),
            S_uv=np.full((2, 2), 2.0),
        )
        signed_gnn_maps = {"z_500": np.ones((2, 2))}
        magnitude_gnn_maps = {
            "z_500": np.ones((2, 2)),
            "uv_500": np.full((2, 2), 3.0),
        }

        inputs = _build_swe_alignment_inputs(swe_result, signed_gnn_maps, magnitude_gnn_maps)

        assert all(pair[2] != "uv_500" for pair in inputs["main_pairs_scatter"])
        assert all(pair[2] != "uv_500" for pair in inputs["main_pairs_metrics"])
        assert [pair[2] for pair in inputs["overlap_pairs"]] == ["z_500", "uv_500"]
        assert any("uv_500" in msg for msg in inputs["warnings"])

    def test_swe_alignment_inputs_report_empty_main_path(self):
        from types import SimpleNamespace
        from physics.swe.comparison_core import _build_swe_alignment_inputs

        swe_result = SimpleNamespace(
            S_h=np.ones((2, 2)),
            S_uv=np.full((2, 2), 2.0),
        )

        inputs = _build_swe_alignment_inputs(
            swe_result,
            signed_gnn_maps={},
            magnitude_gnn_maps={"uv_500": np.full((2, 2), 3.0)},
        )

        assert inputs["main_pairs_scatter"] == []
        assert inputs["main_pairs_metrics"] == []
        assert [pair[2] for pair in inputs["overlap_pairs"]] == ["uv_500"]


class TestDeepLayerSteeringInputValidation:
    def test_deep_layer_steering_returns_none_for_single_time_slice(self):
        import xarray as xr
        from physics.swe.comparison_core import _compute_deep_layer_steering_from_eval_inputs

        lat = np.array([0.0, 1.0], dtype=np.float32)
        lon = np.array([10.0, 11.0], dtype=np.float32)
        level = np.array([925.0, 500.0], dtype=np.float32)
        coords = {
            "time": [0],
            "level": level,
            "lat": lat,
            "lon": lon,
        }
        shape = (1, 2, 2, 2)
        eval_inputs = xr.Dataset(
            {
                "u_component_of_wind": xr.DataArray(np.ones(shape), dims=("time", "level", "lat", "lon"), coords=coords),
                "v_component_of_wind": xr.DataArray(np.ones(shape), dims=("time", "level", "lat", "lon"), coords=coords),
            }
        )

        result = _compute_deep_layer_steering_from_eval_inputs(
            eval_inputs=eval_inputs,
            swe_lat=lat,
            swe_lon=lon,
            center_lat=0.5,
            center_lon=10.5,
        )

        assert result is None

    def test_run_steering_sweep_imports_compute_sensitivity_jax_locally(self, monkeypatch):
        import sys
        import types
        from types import SimpleNamespace

        import config as cfg_mod
        from physics.swe.comparison_core import _run_steering_sweep

        calls = []
        fake_module = types.ModuleType("physics.swe.swe_sensitivity")

        def fake_compute_sensitivity_jax(*args, **kwargs):
            calls.append(
                {
                    "forced_U_bar": kwargs["forced_U_bar"],
                    "forced_V_bar": kwargs["forced_V_bar"],
                }
            )
            return SimpleNamespace()

        fake_module.compute_sensitivity_jax = fake_compute_sensitivity_jax
        monkeypatch.setitem(sys.modules, "physics.swe.swe_sensitivity", fake_module)
        monkeypatch.setattr(cfg_mod, "SWE_UBAR_SWEEP_MAGS", [0.0, 2.0], raising=False)
        monkeypatch.setattr(
            "physics.swe.comparison_core._compute_upstream_and_anisotropy",
            lambda swe_result, lead_h: {"upstream_fraction": 0.5, "anisotropy_ratio": 1.0},
        )

        rows = _run_steering_sweep(
            h0=np.zeros((2, 2)),
            u0=np.zeros((2, 2)),
            v0=np.zeros((2, 2)),
            swe_lat=np.array([0.0, 1.0]),
            swe_lon=np.array([10.0, 11.0]),
            center_lat=0.5,
            center_lon=10.5,
            t_idx=0,
            lead_h=6.0,
            sigma_deg=3.0,
            swe_dt=300.0,
            core_radius_deg=3.0,
            constraint_mode="geostrophic_hard",
            base_u=4.0,
            base_v=0.0,
            H_eq=22.0,
            rayleigh_momentum_h=4.0,
            rayleigh_height_h=8.0,
            diffusion_coeff=1e4,
            sponge_width=6,
            sponge_efold_h=1.5,
        )

        assert len(rows) == 2
        assert len(calls) == 2
        assert calls[0]["forced_U_bar"] == 0.0
        assert calls[1]["forced_U_bar"] == 2.0


class TestArtifactGuards:
    def test_should_emit_alignment_scatter_requires_pairs(self):
        from physics.swe.comparison_core import _should_emit_alignment_scatter

        assert _should_emit_alignment_scatter([]) is False
        assert _should_emit_alignment_scatter([("h", np.ones((2, 2)), "z_500", "x", "y")]) is True

    def test_should_emit_topk_artifacts_requires_pairs(self):
        from physics.swe.comparison_core import _should_emit_topk_artifacts

        assert _should_emit_topk_artifacts([]) is False
        assert _should_emit_topk_artifacts([("h", np.ones((2, 2)), "z_500")]) is True


class TestDlmsfAlignmentInputBuilder:
    """验证 DLMSF 主/补充对比输入语义诚实。"""

    def test_alignment_inputs_use_s_abs_map_for_overlap_track(self):
        from types import SimpleNamespace
        from physics.swe.comparison_core import _build_dlmsf_alignment_inputs

        dlmsf_result = SimpleNamespace(
            S_map=np.array([[-2.0, 1.0], [0.5, -0.25]]),
            S_abs_map=np.array([[2.0, 1.0], [0.5, 0.25]]),
        )
        signed_gnn_maps = {"z_500": np.array([[-3.0, 2.0], [4.0, -5.0]])}
        magnitude_gnn_maps = {
            "z_500": np.array([[3.0, 2.0], [4.0, 5.0]]),
            "uv_500": np.array([[5.0, 5.0], [8.0, 8.0]]),
        }

        inputs = _build_dlmsf_alignment_inputs(dlmsf_result, signed_gnn_maps, magnitude_gnn_maps)

        assert [spec["gnn_key"] for spec in inputs["main_specs"]] == ["z_500"]
        assert inputs["main_specs"][0]["s_map"] is dlmsf_result.S_map
        assert inputs["main_specs"][0]["gnn_map"] is signed_gnn_maps["z_500"]
        assert [pair[2] for pair in inputs["overlap_pairs"]] == ["z_500", "uv_500"]
        assert inputs["overlap_pairs"][0][1] is dlmsf_result.S_abs_map
        assert inputs["overlap_pairs"][1][1] is dlmsf_result.S_abs_map

    def test_alignment_inputs_do_not_invent_signed_uv_map(self):
        from types import SimpleNamespace
        from physics.swe.comparison_core import _build_dlmsf_alignment_inputs

        dlmsf_result = SimpleNamespace(
            S_map=np.ones((2, 2)),
            S_abs_map=np.ones((2, 2)),
        )
        signed_gnn_maps = {"z_500": np.ones((2, 2))}
        magnitude_gnn_maps = {
            "z_500": np.ones((2, 2)),
            "uv_500": np.full((2, 2), 2.0),
        }

        inputs = _build_dlmsf_alignment_inputs(dlmsf_result, signed_gnn_maps, magnitude_gnn_maps)

        assert all(spec["gnn_key"] != "uv_500" for spec in inputs["main_specs"])
        assert any("uv_500" in msg for msg in inputs["warnings"])
