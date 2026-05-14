#!/usr/bin/env python3
"""Generate an object availability checklist and review video.

The video is intentionally dependency-light: it uses PyBullet TinyRenderer for
local primitives/URDF objects and Pillow cards for remote or aggregate sources.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = ROOT / "summary"
CATEGORY_DIR = ROOT / "by_category"
BUILD_DIR = ROOT / "_build"
FRAMES_DIR = BUILD_DIR / "frames"
THUMBS_DIR = BUILD_DIR / "thumbnails"
OUT_MP4 = SUMMARY_DIR / "object_availability_review.mp4"
OUT_CONTACT = SUMMARY_DIR / "object_availability_contact_sheet.png"
OUT_CSV = SUMMARY_DIR / "object_availability.csv"
OUT_JSON = SUMMARY_DIR / "object_availability.json"
OUT_REPORT = SUMMARY_DIR / "object_availability_report.md"

W, H = 1280, 720
FPS = 1
SECONDS_PER_OBJECT = 2


@dataclass
class ObjectRecord:
    object_id: str
    group: str
    scenes: str
    source: str
    status: str
    verdict_cn: str
    evidence_cn: str
    render_kind: str
    shape: str = ""
    scale: str = ""
    urdf: str = ""
    note_cn: str = ""


COLORS = {
    "OK": (50, 155, 92),
    "OK*": (55, 126, 184),
    "SETUP": (220, 145, 35),
    "CHECK": (190, 68, 68),
}

PALETTE = [
    (0.88, 0.10, 0.08, 1.0),
    (0.08, 0.20, 0.85, 1.0),
    (0.95, 0.80, 0.08, 1.0),
    (0.12, 0.55, 0.20, 1.0),
    (0.05, 0.70, 0.80, 1.0),
    (0.55, 0.20, 0.75, 1.0),
]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def records() -> list[ObjectRecord]:
    rows: list[ObjectRecord] = []

    dynamic = [
        ("sphere_s", "Sphere", "0.18", "S1/S4"),
        ("sphere_m", "Sphere", "0.22", "S1/S3/S5-S8/S13/S14"),
        ("sphere_l", "Sphere", "0.28", "S1/S4"),
        ("cube_s", "Cube", "(0.18,0.18,0.18)", "S1/S2/S14"),
        ("cube_m", "Cube", "(0.24,0.24,0.24)", "S1-S3/S6/S8/S13/S14"),
        ("cube_l", "Cube", "(0.30,0.30,0.30)", "S1/S2/S14"),
        ("cylinder_s", "Cylinder", "(0.16,0.16,0.22)", "S1/S2"),
        ("cylinder_m", "Cylinder", "(0.20,0.20,0.28)", "S1-S3/S9"),
        ("cylinder_l", "Cylinder", "(0.24,0.24,0.34)", "S1/S2"),
    ]
    for object_id, shape, scale, scenes in dynamic:
        rows.append(ObjectRecord(
            object_id=object_id,
            group="basic_dynamic",
            scenes=scenes,
            source=f"kb.{shape} / PyBullet primitive",
            status="OK*",
            verdict_cn="物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。",
            evidence_cn="基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。",
            render_kind="primitive",
            shape=shape.lower(),
            scale=scale,
        ))

    static = [
        ("wall_x", "Cube", "(0.08,1.8,0.7)", "S4/S13"),
        ("wall_y", "Cube", "(1.8,0.08,0.7)", "optional"),
        ("ramp", "Cube", "(1.8,0.7,0.08), rotated", "S3"),
        ("occluder", "Cube", "(0.08,1.2,0.9)", "S13"),
        ("pillar", "Cylinder", "(0.16,0.16,0.8)", "S13"),
    ]
    for object_id, shape, scale, scenes in static:
        rows.append(ObjectRecord(
            object_id=object_id,
            group="basic_static",
            scenes=scenes,
            source=f"kb.{shape} / PyBullet primitive",
            status="OK*",
            verdict_cn="可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。",
            evidence_cn="可用 Cube/Cylinder 参数化生成，适合墙、斜面、遮挡板和柱体。",
            render_kind="primitive",
            shape=shape.lower(),
            scale=scale,
        ))

    for object_id in ["cone", "torus", "gear", "torus_knot", "sponge", "spot", "teapot", "suzanne"]:
        rows.append(ObjectRecord(
            object_id=object_id,
            group="kubasic",
            scenes="S9",
            source="gs://kubric-public/assets/KuBasic/KuBasic.json",
            status="OK*",
            verdict_cn="远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。",
            evidence_cn="已读取 KuBasic public manifest，8 个目标 id 均存在。",
            render_kind="card",
            note_cn="建议先补 Kubric 依赖，再跑小样本验证 mesh 尺寸、贴地高度和碰撞稳定性。",
        ))

    urdfs = [
        ("soccerball.urdf", "S10"),
        ("cube.urdf", "S10"),
        ("block.urdf", "S10"),
        ("lego/lego.urdf", "S10"),
        ("duck_vhacd.urdf", "S10"),
        ("teddy_vhacd.urdf", "S10"),
        ("objects/mug.urdf", "S10"),
        ("tray/tray.urdf", "S10"),
        ("domino/domino.urdf", "S11"),
        ("jenga/jenga.urdf", "S12"),
    ]
    for urdf, scenes in urdfs:
        rows.append(ObjectRecord(
            object_id=urdf,
            group="pybullet_urdf",
            scenes=scenes,
            source="pybullet_data",
            status="OK",
            verdict_cn="当前环境可直接加载。",
            evidence_cn="pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。",
            render_kind="urdf",
            urdf=urdf,
        ))

    rows.append(ObjectRecord(
        object_id="random_urdfs/000-999",
        group="pybullet_random_urdfs",
        scenes="S16",
        source="pybullet_data/random_urdfs",
        status="OK",
        verdict_cn="当前环境可用；本机检测到 1000 个随机 URDF。",
        evidence_cn="适合做杂物和静态/动态干扰物，但应抽样剔除极端尺寸或不稳定物体。",
        render_kind="card",
    ))
    rows.append(ObjectRecord(
        object_id="Google Scanned Objects",
        group="gso",
        scenes="S15/S16",
        source="gs://kubric-public/assets/GSO/GSO.json",
        status="CHECK",
        verdict_cn="当前工作区没有本地 GSO manifest；远端 manifest 本次请求失败，暂不算已验证可用。",
        evidence_cn="MOVi 脚本引用 GSO，但生成前需要下载/缓存资产并做尺寸、碰撞稳定性抽检。",
        render_kind="card",
    ))
    rows.append(ObjectRecord(
        object_id="bouncing_balls: sphere/cube/mixed",
        group="scenario_template",
        scenes="S14",
        source="kubric examples/bouncing_balls.py",
        status="OK*",
        verdict_cn="场景模板和物体类型设计可用；仍受当前 Kubric 依赖缺失影响。",
        evidence_cn="对象由 sphere/cube primitive 组成，逻辑上不依赖额外资产。",
        render_kind="card",
    ))
    return rows


def parse_scale(scale: str, shape: str) -> tuple:
    nums = []
    for token in scale.replace("(", "").replace(")", "").replace("rotated", "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            nums.append(float(token))
        except ValueError:
            pass
    if shape == "sphere":
        return (nums[0] if nums else 0.24,)
    if shape == "cylinder":
        return (nums[0] if nums else 0.2, nums[2] if len(nums) >= 3 else 0.4)
    return tuple(nums[:3]) if len(nums) >= 3 else (0.24, 0.24, 0.24)


def render_pybullet(record: ObjectRecord, idx: int, size: int = 480) -> Image.Image:
    cid = p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    color = PALETTE[idx % len(PALETTE)]

    if record.render_kind == "primitive":
        if record.shape == "sphere":
            radius = parse_scale(record.scale, record.shape)[0]
            collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            body = p.createMultiBody(1.0, collision, visual, [0, 0, radius])
        elif record.shape == "cylinder":
            radius, height = parse_scale(record.scale, record.shape)
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
            body = p.createMultiBody(1.0, collision, visual, [0, 0, height / 2])
        else:
            sx, sy, sz = parse_scale(record.scale, record.shape)
            half = [sx, sy, sz]
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
            body = p.createMultiBody(1.0, collision, visual, [0, 0, sz])
            if record.object_id == "ramp":
                quat = p.getQuaternionFromEuler([0, math.radians(25), 0])
                p.resetBasePositionAndOrientation(body, [0, 0, sz + 0.15], quat)
    else:
        body = p.loadURDF(record.urdf, [0, 0, 0], useFixedBase=False)

    aabb_min, aabb_max = p.getAABB(body)
    center = [(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)]
    extent = max(aabb_max[i] - aabb_min[i] for i in range(3))
    distance = max(1.0, extent * 3.0)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=center,
        distance=distance,
        yaw=38,
        pitch=-25,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1, nearVal=0.01, farVal=20)
    _, _, rgba, _, _ = p.getCameraImage(
        size,
        size,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[-3, -4, 6],
        shadow=1,
    )
    arr = np.reshape(np.array(rgba, dtype=np.uint8), (size, size, 4))
    p.disconnect(cid)
    return Image.fromarray(arr[:, :, :3], "RGB")


def render_card_thumbnail(record: ObjectRecord, idx: int, size: int = 480) -> Image.Image:
    img = Image.new("RGB", (size, size), (242, 244, 247))
    d = ImageDraw.Draw(img)
    accent = COLORS.get(record.status, (90, 90, 90))
    d.rounded_rectangle([28, 28, size - 28, size - 28], radius=18, fill=(255, 255, 255), outline=(210, 216, 224), width=2)
    d.ellipse([size // 2 - 86, 120, size // 2 + 86, 292], fill=accent, outline=(34, 40, 49), width=3)
    d.rectangle([size // 2 - 114, 260, size // 2 + 114, 322], fill=(34, 40, 49))
    d.text((48, 48), record.group.upper(), font=font(24), fill=(80, 88, 99))
    d.text((48, 356), record.status, font=font(58), fill=accent)
    wrap_ascii(d, record.object_id, (48, 425), font(25), (34, 40, 49), 31)
    return img


def wrap_ascii(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt, fill, max_chars: int, line_gap: int = 8) -> None:
    words = text.replace("/", "/ ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if len(test) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    x, y = xy
    for line in lines[:4]:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap


def make_thumbnail(record: ObjectRecord, idx: int) -> Path:
    try:
        if record.render_kind in {"primitive", "urdf"}:
            img = render_pybullet(record, idx)
        else:
            img = render_card_thumbnail(record, idx)
    except Exception:
        img = render_card_thumbnail(record, idx)
    path = THUMBS_DIR / f"{idx:03d}_{safe_name(record.object_id)}.png"
    img.save(path)
    return path


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:80]


def make_frame(record: ObjectRecord, idx: int, thumb_path: Path, frame_idx: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (248, 249, 251))
    d = ImageDraw.Draw(img)
    accent = COLORS.get(record.status, (90, 90, 90))
    d.rectangle([0, 0, W, 92], fill=(31, 38, 48))
    d.text((42, 26), "Object Availability Review", font=font(38), fill=(255, 255, 255))
    d.rounded_rectangle([W - 205, 21, W - 44, 70], radius=12, fill=accent)
    d.text((W - 177, 29), record.status, font=font(28), fill=(255, 255, 255))

    thumb = Image.open(thumb_path).convert("RGB").resize((450, 450))
    img.paste(thumb, (60, 145))

    x = 560
    d.text((x, 142), record.object_id, font=font(42), fill=(30, 36, 46))
    fields = [
        ("Group", record.group),
        ("Scenes", record.scenes),
        ("Source", record.source),
        ("Shape / Scale", f"{record.shape or record.urdf or 'asset collection'}  {record.scale}".strip()),
    ]
    y = 220
    for label, value in fields:
        d.text((x, y), label, font=font(21), fill=(105, 115, 128))
        wrap_ascii(d, value, (x, y + 30), font(27), (36, 43, 53), 42, 5)
        y += 92

    d.rectangle([560, 595, 1188, 598], fill=(218, 224, 232))
    d.text((560, 615), f"{idx + 1:02d}/{TOTAL_RECORDS:02d}    frame {frame_idx:03d}", font=font(22), fill=(88, 99, 113))
    return img


TOTAL_RECORDS = len(records())


def grouped_rows(rows: list[ObjectRecord]) -> dict[str, list[ObjectRecord]]:
    groups: dict[str, list[ObjectRecord]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    return groups


CATEGORY_DESCRIPTIONS = {
    "basic_dynamic": {
        "title_cn": "基础动态物体",
        "intro_cn": "这一类是数据集第一版最核心的可动物体，包括球、方块和圆柱。它们形状简单、物理稳定、属性容易控制，适合覆盖质量、摩擦、恢复系数、初始速度和碰撞结果等基础因果变量。",
        "recommendation_cn": "优先使用。建议先用这些物体完成 S1-S8 的基础可控物理场景。",
    },
    "basic_static": {
        "title_cn": "基础静态物体",
        "intro_cn": "这一类主要作为环境、约束和遮挡使用，包括墙、斜面、遮挡板和柱体。它们通常不主动运动，用来制造反弹、滑动、遮挡、障碍物绕行等物理情境。",
        "recommendation_cn": "优先使用。适合和基础动态物体组合，构成第一版可控场景。",
    },
    "kubasic": {
        "title_cn": "KuBasic 复杂几何体",
        "intro_cn": "这一类来自 Kubric 官方 KuBasic 资产，包含圆锥、圆环、齿轮、茶壶、猴头等复杂形状。它们比基础几何体更能测试模型对形状差异、旋转和复杂接触面的泛化能力。",
        "recommendation_cn": "第二批使用。当前已确认远端 asset id 存在，但本地还需要补齐 Kubric 环境并缓存资产后再批量生成。",
    },
    "pybullet_urdf": {
        "title_cn": "PyBullet 常用 URDF 物体",
        "intro_cn": "这一类是 pybullet_data 中已经实测可加载的日常物体和结构物，包括足球、乐高、杯子、托盘、多米诺和 Jenga。它们适合把数据集从抽象几何体扩展到更真实的刚体外观和复杂碰撞。",
        "recommendation_cn": "第二批使用。建议先抽小样本检查尺寸、贴地高度、质量和碰撞稳定性。",
    },
    "pybullet_random_urdfs": {
        "title_cn": "PyBullet 随机 URDF 物体库",
        "intro_cn": "这一类代表 pybullet_data 中的 1000 个随机几何组合物体。它们适合做杂物、干扰物和 OOD 测试，但单个物体的形状、尺度和稳定性差异较大。",
        "recommendation_cn": "后期使用。纳入数据集前应先做自动筛查，剔除尺寸异常、碰撞不稳定或视觉效果差的样本。",
    },
    "gso": {
        "title_cn": "Google Scanned Objects",
        "intro_cn": "这一类是真实扫描物体资产，适合构造更接近真实世界的碰撞和遮挡场景。它们的外观和几何复杂度高，但下载、缓存、尺寸归一化和碰撞稳定性都需要额外验证。",
        "recommendation_cn": "暂缓使用。本次当前工作区没有本地 GSO manifest，远端请求也失败，所以暂时标为 CHECK。",
    },
    "scenario_template": {
        "title_cn": "场景模板",
        "intro_cn": "这一类不是单个物体资产，而是可复用的生成脚本或场景配置。比如 bouncing_balls 可以快速生成多物体弹跳和碰撞视频，用来验证数据流程、轨迹标注和模型输入格式。",
        "recommendation_cn": "可作为调试模板使用。正式数据集仍建议把物体、属性和事件标注显式写入 manifest。",
    },
}


def object_description_cn(row: ObjectRecord) -> str:
    descriptions = {
        "sphere_s": "小球，适合自由落体、反弹、墙面碰撞和小尺度遮挡测试。",
        "sphere_m": "中球，基础场景主力物体，适合球撞球、球撞方块、负样本和遮挡场景。",
        "sphere_l": "大球，适合测试尺度变化对落体、反弹和墙面碰撞的影响。",
        "cube_s": "小方块，适合滑动停止、落体翻滚和多物体混合碰撞。",
        "cube_m": "中方块，球撞方块和遮挡场景中的主要目标物体。",
        "cube_l": "大方块，适合测试尺度、质量和摩擦变化下的滑动距离。",
        "cylinder_s": "小圆柱，适合测试滚动、侧翻和接触面变化。",
        "cylinder_m": "中圆柱，适合斜面滑动和复杂几何碰撞的基础对照。",
        "cylinder_l": "大圆柱，适合测试较大尺寸柱体的滚动和停止行为。",
        "wall_x": "沿 y 方向延展的竖直墙，用于墙面反弹和边界碰撞。",
        "wall_y": "沿 x 方向延展的竖直墙，可作为另一个方向的边界或反弹面。",
        "ramp": "斜面，用于生成重力驱动的滑动、滚动和底部停止事件。",
        "occluder": "遮挡板，用于让物体短暂消失，测试身份保持和遮挡后推理。",
        "pillar": "圆柱障碍物，用于制造绕行、碰撞、遮挡和复杂接触。",
        "cone": "圆锥体，接触面不对称，适合测试复杂形状碰撞和旋转。",
        "torus": "圆环体，适合测试非凸外观、滚动和孔洞形状带来的视觉泛化。",
        "gear": "齿轮形物体，边缘复杂，适合测试复杂接触和旋转。",
        "torus_knot": "扭结圆环，外形复杂，适合作为形状泛化和视觉分割难例。",
        "sponge": "多孔复杂几何体，适合作为复杂形状碰撞扩展。",
        "spot": "KuBasic 中的 Spot 模型，适合测试非基础几何体的外观泛化。",
        "teapot": "茶壶模型，典型复杂网格物体，适合真实物体前的过渡测试。",
        "suzanne": "Blender 猴头模型，常用复杂几何基准，适合测试视角和形状泛化。",
        "soccerball.urdf": "足球外观球体，适合替代抽象球做真实外观碰撞。",
        "cube.urdf": "标准方块 URDF，适合和原生方块做一致性对照。",
        "block.urdf": "细长块状物，适合测试小尺寸目标被撞后的位移和旋转。",
        "lego/lego.urdf": "乐高积木，适合引入轻量真实物体外观和凸起结构。",
        "duck_vhacd.urdf": "小黄鸭，带 VHACD 碰撞近似，适合日常物体碰撞扩展。",
        "teddy_vhacd.urdf": "泰迪熊，带 VHACD 碰撞近似，适合非规则外形测试。",
        "objects/mug.urdf": "杯子，典型日常容器物体，适合真实外观和非规则接触测试。",
        "tray/tray.urdf": "托盘，尺寸较大且有边缘结构，适合作为容器、障碍或被撞目标。",
        "domino/domino.urdf": "多米诺骨牌，适合连锁倒塌和接触传播场景。",
        "jenga/jenga.urdf": "Jenga 木块，适合堆叠稳定性和坍塌事件场景。",
        "random_urdfs/000-999": "1000 个随机 URDF 物体集合，适合作为杂物或 OOD 测试库。",
        "Google Scanned Objects": "真实扫描物体集合，适合高真实感场景，但当前尚未完成本地验证。",
        "bouncing_balls: sphere/cube/mixed": "多物体弹球场景模板，用于快速验证多物体碰撞和轨迹预测流程。",
    }
    return descriptions.get(row.object_id, row.verdict_cn)


def write_tables(rows: list[ObjectRecord]) -> None:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    OUT_JSON.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1

    lines = [
        "# task2 判断物体是否可用",
        "",
        "## 结论摘要",
        "",
        f"- 总计检查对象/对象集合：{len(rows)}",
        f"- OK：{counts.get('OK', 0)}，当前环境可直接加载或已有本地资源。",
        f"- OK*：{counts.get('OK*', 0)}，对象定义或远端 asset id 可用，但当前 Kubric 环境需补依赖/下载资产。",
        f"- CHECK：{counts.get('CHECK', 0)}，当前工作区未完成验证，不建议直接纳入第一批生成。",
        "",
        "当前 `python3` 导入 Kubric 失败，错误为缺少 `pyquaternion`；因此基础 Kubric primitive 和 KuBasic 标为 OK*，不是物体不可用，而是渲染环境还没完整跑通。",
        "",
        "## 建议",
        "",
        "- 第一批可以先用 `sphere/cube/cylinder/wall/ramp/occluder/pillar` 和 pybullet_data 中已加载成功的 URDF。",
        "- KuBasic 的 `cone/torus/gear/torus_knot/sponge/spot/teapot/suzanne` 可以作为第二批，先补 Kubric 依赖并缓存资产。",
        "- GSO 暂时放到第三批；本次没有本地 manifest，远端 manifest 请求失败，需要单独下载和抽样验证。",
        "- `random_urdfs/000-999` 本地存在 1000 个，但建议抽样剔除尺寸异常、重心怪、碰撞不稳定的物体。",
        "",
        "## 分类目录",
        "",
        "| group | 中文名称 | 数量 | 目录 |",
        "|---|---|---:|---|",
    ]
    for group, group_rows in grouped_rows(rows).items():
        title_cn = CATEGORY_DESCRIPTIONS.get(group, {}).get("title_cn", group)
        lines.append(f"| `{group}` | {title_cn} | {len(group_rows)} | `../by_category/{group}/` |")
    lines.extend([
        "",
        "## 明细",
        "",
        "| object_id | group | scenes | status | 结论 | 证据 |",
        "|---|---|---|---|---|---|",
    ])
    for r in rows:
        lines.append(
            f"| `{r.object_id}` | {r.group} | {r.scenes} | {r.status} | {r.verdict_cn} | {r.evidence_cn} |"
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_contact_sheet(rows: list[ObjectRecord], thumb_paths: list[Path], out_path: Path = OUT_CONTACT) -> None:
    cols = 5
    cell_w, cell_h = 300, 365
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(len(rows) / cols) * cell_h), (248, 249, 251))
    d = ImageDraw.Draw(sheet)
    for i, (row, path) in enumerate(zip(rows, thumb_paths)):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        thumb = Image.open(path).convert("RGB").resize((240, 240))
        sheet.paste(thumb, (x + 30, y + 18))
        d.text((x + 30, y + 268), row.status, font=font(24), fill=COLORS.get(row.status, (40, 40, 40)))
        wrap_ascii(d, row.object_id, (x + 30, y + 300), font(18), (33, 39, 48), 26, 4)
    sheet.save(out_path)


def write_category_outputs(rows: list[ObjectRecord], thumb_paths: list[Path]) -> None:
    CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[tuple[ObjectRecord, Path]]] = {}
    for row, thumb_path in zip(rows, thumb_paths):
        by_group.setdefault(row.group, []).append((row, thumb_path))

    for group, pairs in by_group.items():
        group_dir = CATEGORY_DIR / group
        group_thumbs = group_dir / "thumbnails"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_thumbs.mkdir(parents=True, exist_ok=True)
        for old_thumb in group_thumbs.glob("*.png"):
            old_thumb.unlink()

        group_rows = [row for row, _ in pairs]
        group_thumb_paths = []
        for _, thumb_path in pairs:
            target = group_thumbs / thumb_path.name
            shutil.copy2(thumb_path, target)
            group_thumb_paths.append(target)

        with (group_dir / "objects.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(group_rows[0]).keys()))
            writer.writeheader()
            for row in group_rows:
                writer.writerow(asdict(row))

        (group_dir / "objects.json").write_text(
            json.dumps([asdict(r) for r in group_rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        make_contact_sheet(group_rows, group_thumb_paths, group_dir / "contact_sheet.png")

        counts: dict[str, int] = {}
        for row in group_rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        status_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        category_info = CATEGORY_DESCRIPTIONS.get(group, {})
        lines = [
            f"# {group}",
            "",
            f"## 中文介绍",
            "",
            f"**{category_info.get('title_cn', group)}**",
            "",
            category_info.get("intro_cn", "这一类是当前物体可用性检查中的一个对象分组。"),
            "",
            f"**使用建议**：{category_info.get('recommendation_cn', '根据状态和场景需求选择使用。')}",
            "",
            "## 检查摘要",
            "",
            f"- 数量：{len(group_rows)}",
            f"- 状态统计：{status_text}",
            "- 缩略图目录：`thumbnails/`",
            "- 总览图：`contact_sheet.png`",
            "",
            "## 对象明细",
            "",
            "| object_id | 中文描述 | scenes | status | source | 结论 |",
            "|---|---|---|---|---|---|",
        ]
        for row in group_rows:
            lines.append(
                f"| `{row.object_id}` | {object_description_cn(row)} | {row.scenes} | {row.status} | {row.source} | {row.verdict_cn} |"
            )
        (group_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_legacy_outputs() -> None:
    for name in [
        "_build",
        "frames",
        "thumbnails",
        "object_availability_review.mp4",
        "object_availability_contact_sheet.png",
        "object_availability_report.md",
        "object_availability.csv",
        "object_availability.json",
        "object_intro_inventory.md",
        "object_intro_inventory.csv",
        "object_intro_inventory.json",
    ]:
        path = ROOT / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    for path in FRAMES_DIR.glob("*.png"):
        path.unlink()
    for path in THUMBS_DIR.glob("*.png"):
        path.unlink()

    rows = records()
    write_tables(rows)

    thumb_paths = [make_thumbnail(row, i) for i, row in enumerate(rows)]
    make_contact_sheet(rows, thumb_paths)
    write_category_outputs(rows, thumb_paths)

    frame_idx = 0
    for i, (row, thumb_path) in enumerate(zip(rows, thumb_paths)):
        for _ in range(FPS * SECONDS_PER_OBJECT):
            frame = make_frame(row, i, thumb_path, frame_idx)
            frame.save(FRAMES_DIR / f"frame_{frame_idx:04d}.png")
            frame_idx += 1

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(FRAMES_DIR / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
        str(OUT_MP4),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {OUT_MP4}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_CSV}")
    cleanup_legacy_outputs()


if __name__ == "__main__":
    main()
