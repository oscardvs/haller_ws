"""Compose SO-101 sim scenes by namespacing the vendored arm MJCF.

Strategy: parse the SO-101 MJCF as XML, prefix every `name="..."` attribute and
every reference to those names (`joint="..."`, `body="..."`, etc.) with an arm
prefix, then assemble the prefixed arm(s) plus workbench/cubes/overhead camera
into one parent MJCF. Keeps us off `dm_control` (heavy dep) while staying
deterministic and easy to inspect.

Upstream joint names (trs_so_arm100 / SO-100 MJCF, commit b846dd12):
    Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw

These differ from the LeRobot naming convention used elsewhere; SO101_JOINTS
here reflects the actual MJCF names so that prefixed lookups succeed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

# Canonical SO-101 joint names as they appear in the upstream trs_so_arm100 MJCF.
SO101_JOINTS = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
SO101_DIR = REPO_ROOT / "sim" / "assets" / "so101"
SCENES_DIR = REPO_ROOT / "sim" / "assets" / "scenes"

# Attributes that reference a named element by name (must be prefixed if the
# referenced element was prefixed). This list covers the elements used in the
# trs_so_arm100 MJCF; if upstream adds new reference attrs, extend it.
_NAME_REF_ATTRS = {
    "joint", "body1", "body2", "site", "geom", "mesh", "material",
    "tendon", "actuator", "class", "childclass", "target",
}


def _find_arm_xml() -> Path:
    """Return the SO-101 MJCF that can be parsed standalone (no <include>).

    The vendored layout has two files:
      scene.xml      — top-level scene that uses <include file="so_arm100.xml"/>
      so_arm100.xml  — the actual arm definition (no includes)

    We must use so_arm100.xml directly; parsing scene.xml via ElementTree
    would silently lose the included content.
    """
    # Prefer the arm-definition file that has no includes.
    preferred = SO101_DIR / "so_arm100.xml"
    if preferred.exists():
        return preferred

    # Fall back: any *.xml that has an <actuator section and no <include>.
    for p in sorted(SO101_DIR.glob("*.xml")):
        text = p.read_text()
        if "<actuator" in text and "<include" not in text:
            return p

    raise FileNotFoundError(f"no standalone SO-101 MJCF found in {SO101_DIR}")


def _collect_named_elements(root: ET.Element) -> set[str]:
    """Names that get prefixed: every element with a `name` OR `class` attribute.

    MuJoCo `<default class="...">` uses `class` as the primary identifier, not
    `name`.  We must prefix those too to avoid "repeated default class name"
    errors when two arms are composed.
    """
    names: set[str] = set()
    for el in root.iter():
        for attr in ("name", "class"):
            v = el.get(attr)
            if v is not None:
                names.add(v)
    return names


def _prefix_element_tree(root: ET.Element, prefix: str, names_to_prefix: set[str]) -> None:
    """In place: prefix every `name="..."` / `class="..."` and every reference
    attribute whose value is in `names_to_prefix`."""
    for el in root.iter():
        for decl_attr in ("name", "class"):
            v = el.get(decl_attr)
            if v is not None and v in names_to_prefix:
                el.set(decl_attr, f"{prefix}{v}")
        for attr in _NAME_REF_ATTRS:
            v = el.get(attr)
            if v is not None and v in names_to_prefix:
                el.set(attr, f"{prefix}{v}")


@dataclass
class _ArmSubtree:
    worldbody_inner: str
    asset_inner: str
    actuator_inner: str
    default_inner: str
    contact_inner: str
    sensor_inner: str
    tendon_inner: str
    equality_inner: str
    compiler_attrs: dict[str, str]  # mesh dir etc.; only the first arm's are used
    joint_names: list[str]


# Sections we extract from the upstream MJCF and recompose under the parent.
# We INTENTIONALLY drop <option>, <size>, <keyframe>, <visual>, <statistic>
# — the parent owns those, and MuJoCo errors on duplicates.
_EXTRACTED_SECTIONS = {
    "worldbody", "asset", "actuator", "default",
    "contact", "sensor", "tendon", "equality",
}


def _load_arm_subtree(prefix: str, x_offset: float) -> _ArmSubtree:
    """Parse the SO-101 MJCF, prefix every name + name-ref, return the per-section
    inner XML so the caller can recompose under a single parent <mujoco> root.
    Wraps the worldbody contents in a positioning body so multi-arm scenes don't
    overlap.
    """
    arm_path = _find_arm_xml()
    tree = ET.parse(arm_path)
    root = tree.getroot()
    names = _collect_named_elements(root)
    _prefix_element_tree(root, prefix, names)

    sections: dict[str, list[str]] = {s: [] for s in _EXTRACTED_SECTIONS}
    compiler_attrs: dict[str, str] = {}
    for child in list(root):
        if child.tag == "compiler":
            compiler_attrs = dict(child.attrib)
        elif child.tag in _EXTRACTED_SECTIONS:
            for sub in list(child):
                sections[child.tag].append(ET.tostring(sub, encoding="unicode"))

    wrapped_worldbody = (
        f'<body name="{prefix}root" pos="{x_offset} 0 0">\n'
        + "\n".join(sections["worldbody"])
        + "\n</body>"
    )

    return _ArmSubtree(
        worldbody_inner=wrapped_worldbody,
        asset_inner="\n".join(sections["asset"]),
        actuator_inner="\n".join(sections["actuator"]),
        default_inner="\n".join(sections["default"]),
        contact_inner="\n".join(sections["contact"]),
        sensor_inner="\n".join(sections["sensor"]),
        tendon_inner="\n".join(sections["tendon"]),
        equality_inner="\n".join(sections["equality"]),
        compiler_attrs=compiler_attrs,
        joint_names=[f"{prefix}{j}" for j in SO101_JOINTS],
    )


def build_scene(arms: list[str], cubes: int) -> tuple[str, dict[str, list[str]]]:
    """Compose a scene MJCF.

    `arms`: list of arm ids (e.g. ["right"] or ["left", "right"]).
    `cubes`: number of 4cm cubes on the workbench.

    Returns (mjcf_xml_string, arm_joint_map). arm_joint_map maps each arm id to
    its list of prefixed joint names — what `MuJoCoWorld` consumes.
    """
    if not arms:
        raise ValueError("scene needs at least one arm")

    # Horizontal offsets so arms don't overlap. Single arm: centered. Two arms:
    # +/- 0.20 m on x.
    if len(arms) == 1:
        offsets = [0.0]
    elif len(arms) == 2:
        offsets = [-0.20, 0.20]
    else:
        raise ValueError(f"only 1 or 2 arms supported, got {len(arms)}")

    arm_joint_map: dict[str, list[str]] = {}
    subtrees: list[_ArmSubtree] = []
    for arm_id, x in zip(arms, offsets):
        sub = _load_arm_subtree(prefix=f"{arm_id}_", x_offset=x)
        subtrees.append(sub)
        arm_joint_map[arm_id] = sub.joint_names

    # Workbench (always) + cubes.
    workbench_inner = _extract_worldbody_inner(SCENES_DIR / "workbench.xml")
    cube_chunks: list[str] = []
    cube_template = (SCENES_DIR / "cube.xml").read_text()
    # Match the full opening tag of the cube body, including any existing pos attr.
    cube_body_re = re.compile(
        r'<body\s+name="cube"(?:\s+pos="[^"]*")?', re.M
    )
    for i in range(cubes):
        x = -0.10 + 0.10 * i
        per_cube = cube_body_re.sub(
            f'<body name="cube_{i}" pos="{x} 0 0.05"', cube_template
        )
        cube_chunks.append(_extract_worldbody_inner_from_string(per_cube))

    # <compiler> — meshdir is relative to the MJCF on disk. We're producing an
    # in-memory string, so resolve meshdir to an absolute path so meshes load.
    compiler_attrs = dict(subtrees[0].compiler_attrs) if subtrees else {}
    meshdir = compiler_attrs.get("meshdir", "assets")
    if not Path(meshdir).is_absolute():
        compiler_attrs["meshdir"] = str((SO101_DIR / meshdir).resolve())
    texturedir = compiler_attrs.get("texturedir")
    if texturedir and not Path(texturedir).is_absolute():
        compiler_attrs["texturedir"] = str((SO101_DIR / texturedir).resolve())
    compiler_attr_str = " ".join(f'{k}="{v}"' for k, v in compiler_attrs.items())

    parts: list[str] = ['<mujoco model="haller-sim">']
    if compiler_attr_str:
        parts.append(f"<compiler {compiler_attr_str}/>")
    parts.append('<option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>')

    def _wrap(tag: str, inners: list[str]) -> None:
        joined = "\n".join(s for s in inners if s.strip())
        if joined:
            parts.append(f"<{tag}>\n{joined}\n</{tag}>")

    _wrap("default",  [s.default_inner  for s in subtrees])
    _wrap("asset",    [s.asset_inner    for s in subtrees])

    parts.append("<worldbody>")
    parts.append(workbench_inner)
    for s in subtrees:
        parts.append(s.worldbody_inner)
    parts.extend(cube_chunks)
    # Overhead camera looking straight down at the workbench.
    parts.append(
        '<camera name="overhead" pos="0 0 1.0" '
        'xyaxes="1 0 0  0 1 0" fovy="60"/>'
    )
    parts.append("</worldbody>")

    _wrap("contact",   [s.contact_inner  for s in subtrees])
    _wrap("equality",  [s.equality_inner for s in subtrees])
    _wrap("tendon",    [s.tendon_inner   for s in subtrees])
    _wrap("actuator",  [s.actuator_inner for s in subtrees])
    _wrap("sensor",    [s.sensor_inner   for s in subtrees])

    parts.append("</mujoco>")
    return "\n".join(parts), arm_joint_map


def _extract_worldbody_inner(path: Path) -> str:
    root = ET.parse(path).getroot()
    wb = root.find("worldbody")
    if wb is None:
        return ""
    return "\n".join(ET.tostring(c, encoding="unicode") for c in wb)


def _extract_worldbody_inner_from_string(xml: str) -> str:
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    if wb is None:
        return ""
    return "\n".join(ET.tostring(c, encoding="unicode") for c in wb)
