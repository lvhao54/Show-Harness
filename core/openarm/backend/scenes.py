"""Resolve and assemble the vendored OpenArm MuJoCo scenes.

Two scenes ship, both assembled around the official OpenArm Pedestal model from the
``openarm-mujoco`` wheel:

* the default three-cube tabletop (``openarm2.xml``), fully self-contained; and
* ``insert_flower`` (``openarm2_insert_flower.xml``), the VLABench vase + three roses,
  whose meshes/textures are large and not redistributable with this repo -- fetch them
  once with ``scripts/model/download_vlabench_assets.sh``.

Both scene XMLs live next to this module; the pedestal and its meshes are resolved from
the installed wheel, exactly as upstream did. The assembled XML is written to a temporary
file because MuJoCo resolves relative asset paths against the MJCF file itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

#: Directory holding the scene XMLs (and, after a download, the VLABench assets).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

#: Scene name -> MJCF file in :data:`ASSETS_DIR`. ``None`` is the default cube tabletop.
OPENARM_SCENES: dict[str | None, str] = {
    None: "openarm2.xml",
    "insert_flower": "openarm2_insert_flower.xml",
}

#: Pedestal component of the official robot, resolved from the wheel.
_PEDESTAL_COMPONENT = "pedestal/openarm_pedestal.xml"


def vlabench_assets_dir() -> Path:
    """Where the VLABench meshes/textures must live once downloaded."""
    return ASSETS_DIR / "vlabench"


def is_vlabench_assets_present() -> bool:
    """Whether ``insert_flower`` can be built (its meshes have been downloaded)."""
    root = vlabench_assets_dir()
    return (root / "vase_0" / "visual" / "vase.obj").is_file() and (
        root / "rose_0" / "rose_0.obj"
    ).is_file()


def scene_file(scene: str | None) -> Path:
    """The MJCF file for ``scene`` (``None`` -> the default cube tabletop)."""
    try:
        name = OPENARM_SCENES[scene]
    except KeyError:
        available = [s for s in OPENARM_SCENES if s]
        raise ValueError(
            f"Unknown OpenArm scene {scene!r}; available: {available or ['(default cubes)']}"
        ) from None
    return ASSETS_DIR / name


@contextmanager
def openarm_scene(xml=None, *, scene=None):
    """Yield a loadable MJCF path for a scene or an explicit ``xml``.

    Every relative ``file=`` reference is rewritten to an absolute path before the XML
    goes to MuJoCo: the assembled document is written to a temporary file, so relative
    paths would otherwise resolve against the wrong directory. References already inside
    the bundled asset tree are made absolute in place; the pedestal component is resolved
    through the ``openarm-mujoco`` wheel's own asset API.
    """
    if xml is not None and scene is not None:
        raise ValueError("xml and scene are mutually exclusive")
    if xml is not None:
        yield str(xml)
        return

    source = scene_file(scene)
    if scene is not None and not is_vlabench_assets_present():
        raise FileNotFoundError(
            f"Scene {scene!r} needs the VLABench meshes, which are not bundled with this "
            f"repository. Fetch them once (fixed revision, sha256-verified) with:\n"
            f"  bash scripts/model/download_vlabench_assets.sh"
        )
    from openarm_mujoco.v2 import asset_path

    root = ET.fromstring(source.read_text(encoding="utf-8"))
    for element in root.iter():
        filename = element.get("file")
        if filename is None:
            continue
        if filename == _PEDESTAL_COMPONENT:
            element.set("file", asset_path(_PEDESTAL_COMPONENT))
        elif not Path(filename).is_absolute():
            element.set("file", str((source.parent / filename).resolve()))
    with NamedTemporaryFile(mode="w", suffix=".xml", encoding="utf-8") as temporary:
        temporary.write(ET.tostring(root, encoding="unicode"))
        temporary.flush()
        yield temporary.name
