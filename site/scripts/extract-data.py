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
    result = {"name": "", "p_peak": {}, "b_peak_hbm": 0}
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "name:" in line and "hardware" not in line.lower().split("name")[0]:
            result["name"] = stripped.split("name:")[1].strip().strip('"').strip("'")
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
            hardware_configs[hw["name"]] = {
                "p_peak": hw["p_peak"],
                "b_peak_hbm": hw["b_peak_hbm"],
            }

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

        # Production performance
        prod_perf = metadata.get("production_performance", {})
        if not isinstance(prod_perf, dict):
            prod_perf = {}
        if not prod_perf:
            for sid, shape_meta in metadata_shapes.items():
                if not isinstance(shape_meta, dict):
                    continue
                shape_perf = shape_meta.get("production_performance")
                if isinstance(shape_perf, dict) and shape_perf:
                    prod_perf[sid] = shape_perf
        prod_summary = None
        prod_shapes = {}
        if prod_perf:
            total_prod_perf_shapes += len(prod_perf)
            times = []
            for sid, v in prod_perf.items():
                t = v.get("performance_us")
                if t is not None:
                    times.append(t)
                    prod_shapes[sid] = round(t, 1)
            fw_set = {v.get("framework", "") for v in prod_perf.values()}
            if times:
                times_sorted = sorted(times)
                prod_summary = {
                    "shapes_measured": len(times),
                    "median_us": round(times_sorted[len(times_sorted) // 2], 1),
                    "min_us": round(times_sorted[0], 1),
                    "max_us": round(times_sorted[-1], 1),
                    "framework": sorted(fw_set - {""})[0] if fw_set - {""} else "",
                }

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
        })

    operators.sort(key=lambda x: x["importance"], reverse=True)

    # Hardware summary for the site
    hw_summary = {}
    for name, cfg in hardware_configs.items():
        p = cfg["p_peak"]
        hw_summary[name] = {
            "bf16_tflops": round(p.get("bf16_tc", 0) / 1e12, 1),
            "fp8_tflops": round(p.get("fp8_e4m3_tc", 0) / 1e12, 1) if p.get("fp8_e4m3_tc") else None,
            "bw_tb_s": round(cfg["b_peak_hbm"] / 1e12, 1),
        }

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
