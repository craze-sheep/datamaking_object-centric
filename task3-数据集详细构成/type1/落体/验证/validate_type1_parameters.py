#!/usr/bin/env python3
"""Validate Type1 PyBullet parameters and generate tiny smoke-test outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pybullet as p
import pybullet_data
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

GROUND = {
    "lateralFriction": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "rollingFriction": [0, 0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15],
    "spinningFriction": [0, 0.0005, 0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.08],
    "restitution": [0.05, 0.15, 0.3, 0.5, 0.7, 0.9],
}

SPHERE = {
    "radius": [0.18, 0.22, 0.28],
    "position_x": [-0.6, 0, 0.6],
    "position_y": [-0.6, 0, 0.6],
    "position_z": [1.0, 1.5, 2.0, 2.5],
    "velocity_x": [-0.6, -0.2, 0.2, 0.6],
    "velocity_y": [-0.6, -0.2, 0.2, 0.6],
    "velocity_z": [-0.6, 0, 0.6],
    "angular_velocity_x": [0],
    "angular_velocity_y": [0],
    "angular_velocity_z": [0],
    "mass": [0.5, 1.0, 2.0],
    "restitution": [0.5],
    "friction": [0.2, 0.5, 0.8],
    "color": ["red", "blue", "yellow", "green"],
}

CUBE = {
    "size": [(0.18, 0.18, 0.18), (0.24, 0.24, 0.24), (0.30, 0.30, 0.30)],
    "position_x": [-0.6, 0, 0.6],
    "position_y": [-0.6, 0, 0.6],
    "position_z": [1.0, 1.5, 2.0, 2.5],
    "velocity_x": [-0.6, -0.2, 0.2, 0.6],
    "velocity_y": [-0.6, -0.2, 0.2, 0.6],
    "velocity_z": [-0.6, 0, 0.6],
    "angular_velocity_x": [0],
    "angular_velocity_y": [0],
    "angular_velocity_z": [0],
    "mass": [0.5, 1.0, 2.0],
    "restitution": [0.5],
    "friction": [0.2, 0.5, 0.8],
    "color": ["red", "blue", "yellow", "green"],
}

COLORS = {
    "red": [0.88, 0.10, 0.08, 1.0],
    "blue": [0.08, 0.20, 0.85, 1.0],
    "yellow": [0.95, 0.80, 0.08, 1.0],
    "green": [0.12, 0.55, 0.20, 1.0],
    "ground": [0.72, 0.74, 0.78, 1.0],
}


def connect() -> int:
    cid = p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    return cid


def disconnect() -> None:
    if p.isConnected():
        p.disconnect()


def make_ground(params: dict) -> int:
    ground_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(ground_id, -1, rgbaColor=COLORS["ground"])
    p.changeDynamics(
        ground_id,
        -1,
        lateralFriction=params["lateralFriction"],
        rollingFriction=params["rollingFriction"],
        spinningFriction=params["spinningFriction"],
        restitution=params["restitution"],
    )
    return ground_id


def make_sphere(params: dict) -> int:
    radius = params["radius"]
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=COLORS[params["color"]],
    )
    body = p.createMultiBody(
        baseMass=params["mass"],
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=params["initial_position"],
    )
    p.changeDynamics(
        body,
        -1,
        mass=params["mass"],
        lateralFriction=params["friction"],
        restitution=params["restitution"],
    )
    p.resetBaseVelocity(
        body,
        linearVelocity=params["initial_velocity"],
        angularVelocity=params["initial_angular_velocity"],
    )
    return body


def make_cube(params: dict) -> int:
    size = params["size"]
    half = [v / 2 for v in size]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=COLORS[params["color"]],
    )
    body = p.createMultiBody(
        baseMass=params["mass"],
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=params["initial_position"],
    )
    p.changeDynamics(
        body,
        -1,
        mass=params["mass"],
        lateralFriction=params["friction"],
        restitution=params["restitution"],
    )
    p.resetBaseVelocity(
        body,
        linearVelocity=params["initial_velocity"],
        angularVelocity=params["initial_angular_velocity"],
    )
    return body


def render_image() -> Image.Image:
    width, height = 320, 240
    view = p.computeViewMatrix(
        cameraEyePosition=[3.6, -5.0, 3.4],
        cameraTargetPosition=[0.0, 0.0, 1.1],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=50,
        aspect=width / height,
        nearVal=0.01,
        farVal=20,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
    )
    return Image.frombytes("RGBA", (width, height), bytes(rgba))


def render_png(path: Path) -> None:
    render_image().save(path)


def write_mp4(frames: list[Image.Image], path: Path, fps: int = 24) -> None:
    if not frames:
        raise ValueError("No frames to write")
    width, height = frames[0].size
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def simulate(body_id: int, frames: int = 64, steps_per_frame: int = 4) -> list[dict]:
    trajectory = []
    for frame in range(frames):
        for _ in range(steps_per_frame):
            p.stepSimulation()
        position, quaternion = p.getBasePositionAndOrientation(body_id)
        velocity, angular_velocity = p.getBaseVelocity(body_id)
        contacts = p.getContactPoints(bodyA=body_id)
        trajectory.append(
            {
                "frame": frame,
                "position": [float(v) for v in position],
                "quaternion": [float(v) for v in quaternion],
                "velocity": [float(v) for v in velocity],
                "angular_velocity": [float(v) for v in angular_velocity],
                "contact_count": len(contacts),
            }
        )
    return trajectory


def simulate_with_frames(
    body_id: int,
    frames: int = 72,
    steps_per_frame: int = 4,
) -> tuple[list[dict], list[Image.Image]]:
    trajectory = []
    rendered_frames = []
    for frame in range(frames):
        for _ in range(steps_per_frame):
            p.stepSimulation()
        position, quaternion = p.getBasePositionAndOrientation(body_id)
        velocity, angular_velocity = p.getBaseVelocity(body_id)
        contacts = p.getContactPoints(bodyA=body_id)
        trajectory.append(
            {
                "frame": frame,
                "position": [float(v) for v in position],
                "quaternion": [float(v) for v in quaternion],
                "velocity": [float(v) for v in velocity],
                "angular_velocity": [float(v) for v in angular_velocity],
                "contact_count": len(contacts),
            }
        )
        rendered_frames.append(render_image())
    return trajectory, rendered_frames


def validate_parameter_support() -> dict:
    results = {"ground": {}, "sphere": {}, "cube": {}}
    cid = connect()
    try:
        ground_id = p.loadURDF("plane.urdf")
        for name, values in GROUND.items():
            results["ground"][name] = []
            for value in values:
                try:
                    p.changeDynamics(ground_id, -1, **{name: value})
                    results["ground"][name].append({"value": value, "ok": True})
                except Exception as exc:  # pragma: no cover - report only
                    results["ground"][name].append({"value": value, "ok": False, "error": str(exc)})
    finally:
        disconnect()

    for shape, values, maker in [
        ("sphere", SPHERE, make_sphere),
        ("cube", CUBE, make_cube),
    ]:
        for name, candidates in values.items():
            results[shape][name] = []
            for value in candidates:
                cid = connect()
                try:
                    make_ground(
                        {
                            "lateralFriction": 0.5,
                            "rollingFriction": 0.005,
                            "spinningFriction": 0.001,
                            "restitution": 0.5,
                        }
                    )
                    if shape == "sphere":
                        params = {
                            "radius": 0.22,
                            "initial_position": [0, 0, 1.2],
                            "initial_velocity": [0, 0, 0],
                            "initial_angular_velocity": [0, 0, 0],
                            "mass": 1.0,
                            "restitution": 0.5,
                            "friction": 0.5,
                            "color": "red",
                        }
                        if name == "radius":
                            params["radius"] = value
                        elif name.startswith("position_"):
                            idx = {"position_x": 0, "position_y": 1, "position_z": 2}[name]
                            params["initial_position"][idx] = value
                        elif name.startswith("velocity_"):
                            idx = {"velocity_x": 0, "velocity_y": 1, "velocity_z": 2}[name]
                            params["initial_velocity"][idx] = value
                        elif name.startswith("angular_velocity_"):
                            idx = {
                                "angular_velocity_x": 0,
                                "angular_velocity_y": 1,
                                "angular_velocity_z": 2,
                            }[name]
                            params["initial_angular_velocity"][idx] = value
                        else:
                            params[name] = value
                    else:
                        params = {
                            "size": [0.24, 0.24, 0.24],
                            "initial_position": [0, 0, 1.2],
                            "initial_velocity": [0, 0, 0],
                            "initial_angular_velocity": [0, 0, 0],
                            "mass": 1.0,
                            "restitution": 0.5,
                            "friction": 0.5,
                            "color": "red",
                        }
                        if name == "size":
                            params["size"] = list(value)
                        elif name.startswith("position_"):
                            idx = {"position_x": 0, "position_y": 1, "position_z": 2}[name]
                            params["initial_position"][idx] = value
                        elif name.startswith("velocity_"):
                            idx = {"velocity_x": 0, "velocity_y": 1, "velocity_z": 2}[name]
                            params["initial_velocity"][idx] = value
                        elif name.startswith("angular_velocity_"):
                            idx = {
                                "angular_velocity_x": 0,
                                "angular_velocity_y": 1,
                                "angular_velocity_z": 2,
                            }[name]
                            params["initial_angular_velocity"][idx] = value
                        else:
                            params[name] = value
                    body = maker(params)
                    simulate(body, frames=4, steps_per_frame=2)
                    results[shape][name].append({"value": value, "ok": True})
                except Exception as exc:  # pragma: no cover - report only
                    results[shape][name].append({"value": value, "ok": False, "error": str(exc)})
                finally:
                    disconnect()
    return results


def run_case(case: dict) -> dict:
    cid = connect()
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        make_ground(case["ground"])
        if case["object"]["type"] == "sphere":
            body = make_sphere(case["object"])
        elif case["object"]["type"] == "cube":
            body = make_cube(case["object"])
        else:
            raise ValueError(case["object"]["type"])
        trajectory, frames = simulate_with_frames(body)
        image_path = OUT_DIR / f"{case['id']}.png"
        video_path = OUT_DIR / f"{case['id']}.mp4"
        json_path = OUT_DIR / f"{case['id']}.json"
        frames[-1].save(image_path)
        write_mp4(frames, video_path)
        json_path.write_text(
            json.dumps({"case": case, "trajectory": trajectory}, indent=2),
            encoding="utf-8",
        )
        return {
            "id": case["id"],
            "ok": True,
            "image": str(image_path.relative_to(ROOT)),
            "video": str(video_path.relative_to(ROOT)),
            "trajectory": str(json_path.relative_to(ROOT)),
            "final_z": trajectory[-1]["position"][2],
        }
    except Exception as exc:  # pragma: no cover - report only
        return {"id": case["id"], "ok": False, "error": str(exc)}
    finally:
        disconnect()


def smoke_cases() -> list[dict]:
    base_ground = {
        "lateralFriction": 0.5,
        "rollingFriction": 0.005,
        "spinningFriction": 0.001,
        "restitution": 0.5,
    }
    return [
        {
            "id": "level1_sphere",
            "level": 1,
            "ground": {**base_ground, "restitution": 0.7},
            "object": {
                "type": "sphere",
                "radius": 0.22,
                "initial_position": [0, 0, 1.8],
                "initial_velocity": [0, 0, -0.3],
                "initial_angular_velocity": [0, 0, 0],
                "mass": 1.0,
                "restitution": 0.5,
                "friction": 0.5,
                "color": "red",
            },
        },
        {
            "id": "level2_sphere",
            "level": 2,
            "ground": {**base_ground, "restitution": 0.2},
            "object": {
                "type": "sphere",
                "radius": 0.22,
                "initial_position": [0.5, -0.5, 2.0],
                "initial_velocity": [0, 0, 0],
                "initial_angular_velocity": [0, 0, 0],
                "mass": 1.0,
                "restitution": 0.5,
                "friction": 0.5,
                "color": "blue",
            },
        },
        {
            "id": "level3_sphere",
            "level": 3,
            "ground": {
                "lateralFriction": 1.5,
                "rollingFriction": 0.04,
                "spinningFriction": 0.02,
                "restitution": 0.7,
            },
            "object": {
                "type": "sphere",
                "radius": 0.18,
                "initial_position": [-0.5, 0.5, 1.6],
                "initial_velocity": [0.4, -0.15, 0.5],
                "initial_angular_velocity": [0, 0, 0],
                "mass": 0.5,
                "restitution": 0.5,
                "friction": 0.8,
                "color": "green",
            },
        },
        {
            "id": "level3_cube",
            "level": 3,
            "ground": {
                "lateralFriction": 0.25,
                "rollingFriction": 0.003,
                "spinningFriction": 0.001,
                "restitution": 0.15,
            },
            "object": {
                "type": "cube",
                "size": [0.30, 0.30, 0.30],
                "initial_position": [0.5, 0.5, 1.2],
                "initial_velocity": [-0.4, 0.15, -0.5],
                "initial_angular_velocity": [0, 0, 0],
                "mass": 2.0,
                "restitution": 0.5,
                "friction": 0.2,
                "color": "yellow",
            },
        },
    ]


def ok_count(results: dict) -> tuple[int, int]:
    ok = 0
    total = 0
    for group in results.values():
        for rows in group.values():
            for row in rows:
                total += 1
                ok += int(row["ok"])
    return ok, total


def write_report(param_results: dict, case_results: list[dict]) -> None:
    ok, total = ok_count(param_results)
    failed = [
        (group, name, row)
        for group, params in param_results.items()
        for name, rows in params.items()
        for row in rows
        if not row["ok"]
    ]
    case_failed = [row for row in case_results if not row["ok"]]
    lines = [
        "# Type1 参数验证报告",
        "",
        f"- 参数设置测试：{ok}/{total} 成功",
        f"- smoke case：{len(case_results) - len(case_failed)}/{len(case_results)} 成功",
        f"- 输出目录：`{OUT_DIR.relative_to(ROOT)}`",
        "",
        "## 结论",
        "- PyBullet 可配置地面参数：lateralFriction, rollingFriction, spinningFriction, restitution",
        "- PyBullet 可配置物体参数：radius/size, initial_position, initial_velocity, initial_angular_velocity, mass, restitution, friction, color",
        "- color 通过 visual shape 设置，不属于动力学参数",
        "- 其他属性使用 PyBullet 默认值",
        "- smoke case 视频：72 帧，24fps，时长约 3 秒",
        "",
        "## Smoke Cases",
    ]
    for row in case_results:
        if row["ok"]:
            lines.append(
                f"- {row['id']}：OK，video=`{row['video']}`，image=`{row['image']}`，trajectory=`{row['trajectory']}`"
            )
        else:
            lines.append(f"- {row['id']}：FAIL，error={row['error']}")
    if failed:
        lines += ["", "## 参数失败项"]
        for group, name, row in failed:
            lines.append(f"- {group}.{name}={row['value']}：{row.get('error', '')}")
    if case_failed:
        lines += ["", "## 生成失败项"]
        for row in case_failed:
            lines.append(f"- {row['id']}：{row.get('error', '')}")
    (ROOT / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    param_results = validate_parameter_support()
    case_results = [run_case(case) for case in smoke_cases()]
    (ROOT / "parameter_support.json").write_text(
        json.dumps(param_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ROOT / "smoke_results.json").write_text(
        json.dumps(case_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(param_results, case_results)
    ok, total = ok_count(param_results)
    failed_cases = [row for row in case_results if not row["ok"]]
    print(f"parameter support: {ok}/{total} ok")
    print(f"smoke cases: {len(case_results) - len(failed_cases)}/{len(case_results)} ok")
    if ok != total or failed_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
