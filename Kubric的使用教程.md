# Kubric 使用教程

## 结论

| 场景 | 用法 |
|---|---|
| 正式生成/渲染数据集 | 用 Docker 官方镜像 |
| 物体检查、URDF 检查、生成 CSV/JSON | 用 `kubric39` 普通 Python |
| 本地验证 `bpy`/Kubric renderer 能否导入 | 用 `kubric-blender-python` |

不要在普通 Python 里 `import bpy`。`bpy` 是 Blender 内嵌模块，必须通过 Blender 启动。

---

## 1. Docker：正式渲染推荐

启动 Docker：

```bash
sudo /usr/local/sbin/wsl-startup.sh
docker info
```

验证 Kubric 镜像：

```bash
docker run --rm kubricdockerhub/kubruntu \
  /usr/bin/python3 -c "import bpy, kubric as kb; from kubric.renderer.blender import Blender; print('bpy OK', bpy.app.version_string); print('kubric OK'); print('renderer OK')"
```

运行官方 Hello World：

```bash
cd /home/lzy/project/slot-datamaking/kubric-main

docker run --rm --interactive \
  --user $(id -u):$(id -g) \
  --volume "$(pwd):/kubric" \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 examples/helloworld.py
```

运行自己的脚本：

```bash
cd /home/lzy/project/slot-datamaking/kubric-main

docker run --rm --interactive \
  --user $(id -u):$(id -g) \
  --volume "$(pwd):/kubric" \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 your_script.py
```

如果脚本在项目根目录：

```bash
cd /home/lzy/project/slot-datamaking

docker run --rm --interactive \
  --user $(id -u):$(id -g) \
  --volume "/home/lzy/project/slot-datamaking:/workspace" \
  --volume "/home/lzy/project/slot-datamaking/kubric-main:/kubric" \
  --workdir /workspace \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 your_script.py
```

---

## 2. kubric39：辅助脚本推荐

激活环境：

```bash
source /home/lzy/miniconda3/etc/profile.d/conda.sh
conda activate kubric39
```

导入本仓库 Kubric 源码：

```bash
export PYTHONPATH=/home/lzy/project/slot-datamaking/kubric-main:$PYTHONPATH
```

验证：

```bash
python - <<'PY'
import pybullet
import pyquaternion
import kubric as kb

print("pybullet OK")
print("pyquaternion OK")
print("kubric", kb.__file__)
print("Sphere", hasattr(kb, "Sphere"))
print("Cube", hasattr(kb, "Cube"))
print("Cylinder", hasattr(kb, "Cylinder"))
PY
```

跑 task2 检查脚本：

```bash
cd /home/lzy/project/slot-datamaking
python task2-判断物体是否可用/make_object_availability_video.py
```

注意：当前 `kb.Cylinder` 为 `False`，圆柱正式渲染建议用 URDF/FileBasedObject，或后续补 Kubric 的 Cylinder 支持。

---

## 3. 本地 Blender 调试入口

已接入：

```text
/home/lzy/miniconda3/envs/kubric39/bin/blender
/home/lzy/miniconda3/envs/kubric39/bin/kubric-blender-python
```

验证导入：

```bash
source /home/lzy/miniconda3/etc/profile.d/conda.sh
conda activate kubric39

cat > /tmp/verify_kubric_blender.py <<'PY'
import bpy
import kubric as kb
from kubric.renderer.blender import Blender
from kubric.simulator import PyBullet

print("bpy", bpy.app.version_string)
print("kubric", kb.__file__)
print("renderer OK", Blender)
print("pybullet OK", PyBullet)
PY

kubric-blender-python /tmp/verify_kubric_blender.py
```

这个入口只建议做导入验证和轻量调试。正式渲染仍用 Docker。

---

## 4. 最小渲染脚本

保存为 `your_script.py`：

```python
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

scene = kb.Scene(resolution=(256, 256))
renderer = KubricRenderer(scene)

scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1))
scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
scene += kb.DirectionalLight(
    name="sun",
    position=(-1, -0.5, 3),
    look_at=(0, 0, 0),
    intensity=1.5,
)
scene.camera = kb.PerspectiveCamera(
    name="camera",
    position=(3, -1, 4),
    look_at=(0, 0, 1),
)

renderer.save_state("output/hello.blend")
frame = renderer.render_still()

kb.write_png(frame["rgba"], "output/hello.png")
kb.write_palette_png(frame["segmentation"], "output/hello_segmentation.png")
kb.write_scaled_png(frame["depth"], "output/hello_depth.png")
```

推荐用 Docker 跑这个脚本。
