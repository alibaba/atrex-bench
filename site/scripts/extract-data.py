#!/usr/bin/env python3
"""Extract operator data from data/ for the static site."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
HARDWARE_DIR = REPO_ROOT / "configs" / "hardware"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "src" / "data" / "operators.json"

AI_THRESHOLD = 10.0


def parse_hardware_yaml(path: Path) -> dict:
    text = path.read_text()
    result = {"name": "", "p_peak": {}, "b_peak_hbm": 0, "launch_overhead_us": None}
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "name:" in line and "hardware" not in line.lower().split("name")[0]:
            result["name"] = stripped.split("name:")[1].strip().strip('"').strip("'")
        if stripped.startswith("launch_overhead_s") and ":" in stripped:
            val = stripped.split(":")[1].strip().split("#")[0].strip()
            if val and val != "null":
                try:
                    result["launch_overhead_us"] = float(val) * 1e6
                except ValueError:
                    pass
        if stripped.startswith("p_peak"):
            section = "p_peak"
        elif stripped.startswith("b_peak"):
            section = "b_peak"
        elif section == "p_peak" and ":" in stripped:
            key = stripped.split(":")[0].strip()
            val = stripped.split(":")[1].strip().split("#")[0].strip()
            try:
                result["p_peak"][key] = int(val)
            except ValueError:
                pass
        elif section == "b_peak" and "hbm" in stripped:
            val = stripped.split(":")[1].strip().split("#")[0].strip()
            try:
                result["b_peak_hbm"] = int(val)
            except ValueError:
                pass
    return result


def main():
    importance_path = DATA_DIR / "operator_importance.json"
    importance_data = json.loads(importance_path.read_text()) if importance_path.exists() else {}
    importance_ops = importance_data.get("operators", {})

    hardware_configs = {}
    for yaml_file in sorted(HARDWARE_DIR.glob("*.yaml")):
        hw = parse_hardware_yaml(yaml_file)
        if hw["name"]:
            cfg: dict = {
                "p_peak": hw["p_peak"],
                "b_peak_hbm": hw["b_peak_hbm"],
            }
            if hw["launch_overhead_us"] is not None:
                cfg["launch_overhead_us"] = hw["launch_overhead_us"]
            hardware_configs[hw["name"]] = cfg

    operators = []
    total_shapes = 0
    total_prod_perf_shapes = 0

    for op_dir in sorted(DATA_DIR.iterdir()):
        if not op_dir.is_dir():
            continue
        metadata_path = op_dir / "metadata.json"
        shapes_path = op_dir / "shapes.json"
        roofline_path = op_dir / "roofline.json"
        if not metadata_path.exists() or not shapes_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        shapes = json.loads(shapes_path.read_text())
        metadata_shapes = metadata.get("shapes", {})
        if not isinstance(metadata_shapes, dict):
            metadata_shapes = {}
        num_shapes = len(shapes)
        total_shapes += num_shapes

        imp_info = importance_ops.get(op_dir.name, {})
        origin = metadata.get("origin", {})
        framework = origin.get("framework", "unknown")
        if "/" in framework:
            framework = framework.split("/")[0]

        # Roofline data — per-shape, per-hardware
        roofline_summary = None
        roofline_shapes = {}
        if roofline_path.exists():
            roofline = json.loads(roofline_path.read_text())
            roof_shapes = roofline.get("shapes", {})
            if roof_shapes:
                ais = []
                all_hw_names = set()
                for sid, sd in roof_shapes.items():
                    w_dict = sd.get("semantic_W_flops", {})
                    total_w = sum(w_dict.values())
                    q_r = sd.get("semantic_Q_read_bytes", 0)
                    q_w = sd.get("semantic_Q_write_bytes", 0)
                    total_q = q_r + q_w
                    ai = total_w / total_q if total_q > 0 else 0
                    ais.append(ai)
                    sol = sd.get("SOL_time_ms", {})
                    all_hw_names.update(sol.keys())

                    desc = ""
                    shape_meta = metadata_shapes.get(sid, {})
                    if isinstance(shape_meta, dict):
                        desc = shape_meta.get("description", "")
                    if not desc:
                        desc = shapes.get(sid, {}).get("description", "")
                    roofline_shapes[sid] = {
                        "W_flops": total_w,
                        "Q_bytes": total_q,
                        "ai": round(ai, 1),
                        "SOL_time_ms": {k: round(v * 1000, 2) for k, v in sol.items() if v is not None},  # ms -> us
                        "description": desc,
                    }

                median_ai = sorted(ais)[len(ais) // 2]
                regime = "compute-bound" if median_ai > AI_THRESHOLD else "memory-bound"
                if median_ai < 0.01:
                    regime = "indexing/structural"

                roofline_summary = {
                    "median_ai": round(median_ai, 1),
                    "regime": regime,
                    "hardware_targets": sorted(all_hw_names),
                }

        # Production performance — keyed by GPU SKU. A SKU only appears for an
        # operator if at least one of its shapes was actually measured on that
        # SKU. SKUs with no measurement (e.g. H20 today) are simply absent, so
        # the site renders "—" for them instead of borrowing XPU-A's numbers.
        prod_by_sku: dict = {}  # sku -> {sid: perf_dict}
        for sid, shape_meta in metadata_shapes.items():
            if not isinstance(shape_meta, dict):
                continue
            shape_perf_by_sku = shape_meta.get("production_performance")
            if not isinstance(shape_perf_by_sku, dict):
                continue
            for sku, perf in shape_perf_by_sku.items():
                if isinstance(perf, dict) and perf.get("performance_us") is not None:
                    prod_by_sku.setdefault(sku, {})[sid] = perf

        prod_summary = {}  # sku -> summary
        prod_shapes = {}   # sku -> {sid: us}
        measured_sids = set()
        for sku, perf_map in prod_by_sku.items():
            times = []
            shapes_us = {}
            fw_set = set()
            for sid, v in perf_map.items():
                t = v.get("performance_us")
                if t is None:
                    continue
                times.append(t)
                shapes_us[sid] = round(t, 1)
                measured_sids.add(sid)
                fw = v.get("framework", "")
                if fw:
                    fw_set.add(fw)
            if not times:
                continue
            times_sorted = sorted(times)
            prod_summary[sku] = {
                "shapes_measured": len(times),
                "median_us": round(times_sorted[len(times_sorted) // 2], 1),
                "min_us": round(times_sorted[0], 1),
                "max_us": round(times_sorted[-1], 1),
                "framework": sorted(fw_set)[0] if fw_set else "",
            }
            prod_shapes[sku] = shapes_us
        total_prod_perf_shapes += len(measured_sids)

        # Source code for detail pages
        ref_path = op_dir / "reference.py"
        inp_path = op_dir / "input.py"
        reference_code = ref_path.read_text() if ref_path.exists() else ""
        input_code = inp_path.read_text() if inp_path.exists() else ""

        # Merged per-shape details (kwargs + roofline + prod)
        shape_details = {}
        for sid, skw in shapes.items():
            entry: dict = {}
            shape_meta = metadata_shapes.get(sid, {})
            if isinstance(shape_meta, dict):
                entry["description"] = shape_meta.get("description", "")
            entry["init_kwargs"] = skw.get("init_kwargs", {})
            entry["input_kwargs"] = skw.get("input_kwargs", {})
            rs = roofline_shapes.get(sid)
            if rs:
                entry["W_flops"] = rs["W_flops"]
                entry["Q_bytes"] = rs["Q_bytes"]
                entry["ai"] = rs["ai"]
                entry["SOL_time_ms"] = rs["SOL_time_ms"]
            prod_us_by_sku = {}
            for sku, sku_shapes in prod_shapes.items():
                if sid in sku_shapes:
                    prod_us_by_sku[sku] = sku_shapes[sid]
            entry["prod_us"] = prod_us_by_sku if prod_us_by_sku else None
            shape_details[sid] = entry

        operators.append({
            "name": op_dir.name,
            "id": metadata.get("id", ""),
            "dtype": metadata.get("dtype", ""),
            "framework": framework,
            "symbol": origin.get("symbol", ""),
            "module": origin.get("module", ""),
            "phase": imp_info.get("phase", ""),
            "shapes": num_shapes,
            "importance": round(imp_info.get("importance_score", 0), 6),
            "roofline": roofline_summary,
            "roofline_shapes": roofline_shapes,
            "production_perf": prod_summary,
            "production_shapes": prod_shapes,
            "reference_code": reference_code,
            "input_code": input_code,
            "shape_details": shape_details,
        })

    operators.sort(key=lambda x: x["importance"], reverse=True)

    # Hardware summary for the site
    hw_summary = {}
    for name, cfg in hardware_configs.items():
        p = cfg["p_peak"]
        spec: dict = {
            "bf16_tflops": round(p.get("bf16_tc", 0) / 1e12, 1),
            "fp8_tflops": round(p.get("fp8_e4m3_tc", 0) / 1e12, 1) if p.get("fp8_e4m3_tc") else None,
            "bw_tb_s": round(cfg["b_peak_hbm"] / 1e12, 1),
        }
        if cfg.get("launch_overhead_us") is not None:
            spec["launch_overhead_us"] = round(cfg["launch_overhead_us"], 2)
        hw_summary[name] = spec

    result = {
        "stats": {
            "operators": len(operators),
            "shapes": total_shapes,
            "dsls": 4,
            "hardware": len(hardware_configs),
            "prod_perf_shapes": total_prod_perf_shapes,
            "traces_profiled": importance_data.get("metadata", {}).get("num_profiles", 0),
        },
        "hardware": sorted(hardware_configs.keys()),
        "hardware_specs": hw_summary,
        "dsl_targets": ["Triton", "Gluon", "FlyDSL", "CuteDSL"],
        "operators": operators,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(operators)} operators ({total_shapes} shapes) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
