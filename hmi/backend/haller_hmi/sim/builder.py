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

import math
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


Vec3 = tuple[float, float, float]

#: Where cubes are dealt on the bench (x, y), metres. y<0 is in front of the
#: bases — the arms' reach direction. Each sits 0.23-0.35 m from the arm that
#: owns it: closer in and the arm must fold steeper than its elbow branch
#: allows, further out and it runs out of reach. One "home" cube per arm, a
#: contested midline cube by the place zone, then a back row for staging.
#: 4 cm cubes, one colour each, so tasks can name them.
_CUBE_SLOTS: list[tuple[float, float]] = [
    (-0.28, -0.22),  # left arm's home cube
    ( 0.28, -0.22),  # right arm's home cube
    ( 0.00, -0.08),  # contested midline cube, back between the bases
    (-0.12, -0.30),
    ( 0.12, -0.30),
]
_CUBE_COLORS: list[str] = [
    "0.85 0.20 0.20 1",  # red
    "0.20 0.70 0.25 1",  # green
    "0.95 0.75 0.15 1",  # amber
    "0.60 0.30 0.80 1",  # violet
    "0.20 0.60 0.80 1",  # teal
]


def _normalise(v: Vec3) -> Vec3:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-9:
        raise ValueError(f"cannot normalise degenerate vector {v!r}")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


#: Scene-level assets appended to the composed <asset> block. The checker on
#: the bench top is a teleop aid, not decoration: a featureless grey slab
#: gives the operator no lateral or depth reference at all, and fine placement
#: through a camera tile needs BOTH. 5 cm cells (texrepeat 24x18 over the
#: 1.2x0.9 m bench) sit near the gripper's jaw span, so "half a cell" is a
#: readable unit of error. Contrast is kept low so the cubes and place zone
#: still pop.
_SCENE_ASSETS = (
    '<texture name="bench_tex" type="2d" builtin="checker" '
    'rgb1="0.38 0.38 0.39" rgb2="0.32 0.32 0.33" width="256" height="256"/>'
    '<material name="bench_mat" texture="bench_tex" texrepeat="24 18" '
    'texuniform="false" reflectance="0.05"/>'
    # No steel material: fixture.xml and pin.xml colour their geoms with plain
    # rgba instead. Both approaches survive domain randomization in MuJoCo 3.10
    # (geom_rgba overrides a material's colour whenever it differs from the
    # compiler default 0.5 0.5 0.5 1), but plain rgba keeps the parts on the
    # same footing as the cubes, and a declared-but-unused material is a
    # standing invitation to write material= and rgba= on the same geom, which
    # silently bakes the rgba in and masks the material forever.
)

#: The prop sets `build_scene` knows how to deal.
_TASKS = ("cubes", "insertion")

#: Matches the whole `place_zone` geom element so the insertion scene can drop
#: it. Deliberately a regex over the extracted worldbody string rather than an
#: XML edit: `_extract_worldbody_inner` already hands back a string, and the
#: caller re-checks that the name is gone afterwards, so a formatting change
#: upstream fails loudly instead of silently leaving the pad in.
_PLACE_ZONE_RE = re.compile(r'<geom\s+name="place_zone".*?/>', re.S)

#: Per-arm gripper camera, injected inside the Fixed_Jaw body so it rides the
#: wrist exactly like the real rig's CSI wrist cams. This is the
#: fine-manipulation view: the operator cameras give context, this one gives
#: millimetres — the jaws centred in frame, the pinch point and its shadow in
#: the lower half, the bench checker for scale.
#:
#: The numbers were SOLVED, not hand-placed (guessing them cost five blind
#: renders): at a canonical grasp pose (Pitch -55, Elbow 85, Wrist_Pitch 55)
#: the desired camera sits 0.14 m behind the fingertip toward the base and
#: 0.11 m above it in WORLD frame, looking at the tip with world-up in frame;
#: that world pose transformed into the Fixed_Jaw frame gives the constants
#: below. Non-obvious result: "behind and above the hand" is local +x here —
#: the jaw's local frame has -y along the fingers and closes along x. Rigid
#: mount, so at other wrist pitches the view tilts with the hand, as a real
#: wrist cam does.
_WRISTCAM_FMT = (
    '<camera name="{prefix}wristcam" pos="0.1399 0.0168 0.0" '
    'xyaxes="0 0 -1  -0.684 0.7295 0" fovy="70"/>'
)


def _inject_wristcam(worldbody_inner: str, prefix: str) -> str:
    """Insert the wrist camera just inside the arm's Fixed_Jaw body."""
    pattern = rf'(<body name="{re.escape(prefix)}Fixed_Jaw"[^>]*>)'
    out, n = re.subn(pattern, r"\1" + _WRISTCAM_FMT.format(prefix=prefix),
                     worldbody_inner, count=1)
    if n != 1:
        raise ValueError(f"no Fixed_Jaw body found for prefix {prefix!r}")
    return out


def camera_xyaxes(pos: Vec3, target: Vec3) -> str:
    """MJCF `xyaxes` for a camera at `pos` aimed at `target`.

    A MuJoCo camera looks along its local -Z, with +X right and +Y up in the
    rendered image; `xyaxes` declares those first two axes and MuJoCo implies
    the third. Deriving them here keeps a viewpoint readable as (where it is,
    what it looks at) instead of six magic numbers that nobody can adjust
    later without re-deriving the basis by hand.
    """
    # +Z points from the target back toward the camera (opposite the view ray).
    z = _normalise((pos[0] - target[0], pos[1] - target[1], pos[2] - target[2]))
    # World up, unless we're looking straight down it — then the cross product
    # collapses and any perpendicular hint will do.
    up: Vec3 = (0.0, 0.0, 1.0)
    if abs(_cross(up, z)[0]) + abs(_cross(up, z)[1]) + abs(_cross(up, z)[2]) < 1e-6:
        up = (0.0, 1.0, 0.0)
    x = _normalise(_cross(up, z))
    y = _cross(z, x)
    return "  ".join(" ".join(f"{c:.6g}" for c in axis) for axis in (x, y))


def build_scene(arms: list[str], cubes: int,
                task: str = "cubes") -> tuple[str, dict[str, list[str]]]:
    """Compose a scene MJCF.

    `arms`: list of arm ids (e.g. ["right"] or ["left", "right"]).
    `cubes`: number of 4cm cubes dealt onto the workbench, in _CUBE_SLOTS
    order (front of the bases, spread across both arms' halves of the bench).
    `task`: which props to deal.
        "cubes"     — pick-and-place: cubes and the place zone (the default,
                      and what every pre-insertion config and test expects).
        "insertion" — bimanual construction: adds the steel `fixture` and
                      `pin`. Cubes are still honoured, but the insertion
                      configs pass 0; the bore, not a pad, is the target.

    The insertion scene also DROPS the pick-and-place pad — see the call site.
    A consequence worth knowing: `TaskMonitor` cannot be constructed against an
    insertion scene, because `place_zone_geom` will not resolve. That is the
    right failure. Scoring insertion with the cube predicate would mark every
    episode a failure, and a loud KeyError at startup beats a dataset of
    silent zeros.

    Returns (mjcf_xml_string, arm_joint_map). arm_joint_map maps each arm id to
    its list of prefixed joint names — what `MuJoCoWorld` consumes.
    """
    if not arms:
        raise ValueError("scene needs at least one arm")
    if task not in _TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_TASKS)}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"duplicate arm ids in {arms!r}")

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

    # Workbench (bench, backdrop, lights, and the pick-and-place pad) + props.
    workbench_inner = _extract_worldbody_inner(SCENES_DIR / "workbench.xml")
    if task == "insertion":
        # Drop the pick-and-place pad. It is the pad's own task's target and
        # means nothing here, but it is a large, saturated, perfectly flat
        # region sitting directly under the fixture's home slot — the single
        # most salient thing in the base camera after the arms. A VLA attends
        # to it and learns nothing, and the operator reads it as a target it
        # is not. Removing it also puts the pin's bottoming-out surface back
        # on the bench where the geometry was validated.
        workbench_inner = _PLACE_ZONE_RE.sub("", workbench_inner, count=1)
        if 'name="place_zone"' in workbench_inner:
            raise RuntimeError(
                "failed to strip place_zone from workbench.xml for the "
                "insertion scene — the geom's formatting changed and the "
                "regex no longer matches it")
    cube_chunks: list[str] = []
    cube_template = (SCENES_DIR / "cube.xml").read_text()
    # Match the full opening tag of the cube body, including any existing pos attr.
    cube_body_re = re.compile(
        r'<body\s+name="cube"(?:\s+pos="[^"]*")?', re.M
    )
    cube_rgba_re = re.compile(r'rgba="[^"]*"')
    for i in range(cubes):
        slot = _CUBE_SLOTS[i % len(_CUBE_SLOTS)]
        lap = i // len(_CUBE_SLOTS)  # more cubes than slots: deal a second row
        x, y, z = slot[0], slot[1] - 0.12 * lap, 0.025 + 0.05 * lap
        per_cube = cube_body_re.sub(
            f'<body name="cube_{i}" pos="{x} {y} {z}"', cube_template
        )
        per_cube = cube_rgba_re.sub(
            f'rgba="{_CUBE_COLORS[i % len(_CUBE_COLORS)]}"', per_cube, count=1)
        cube_chunks.append(_extract_worldbody_inner_from_string(per_cube))

    # Insertion props. Their home poses live in the XML rather than being
    # computed here, unlike the cubes': there are exactly two of them and where
    # they start is a task-design decision (the fixture in front of the left
    # arm, the pin in front of the right, so the demonstration begins with one
    # part per hand), not a slot to be dealt from a table.
    if task == "insertion":
        for part in ("fixture", "pin"):
            cube_chunks.append(
                _extract_worldbody_inner(SCENES_DIR / f"{part}.xml"))

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
    _wrap("asset",    [s.asset_inner    for s in subtrees] + [_SCENE_ASSETS])

    # Crisper shadows: the gripper's shadow on the bench is the operator's
    # main height cue, and at the default 1024 shadowmap it dissolves into a
    # blur by the time the camera is tight on the work.
    #
    # offwidth/offheight: MuJoCo's OFFSCREEN framebuffer defaults to 640x480,
    # and a Renderer larger than it refuses to construct — which would
    # silently kill the 960x720 operator cameras (SimCamera catches the
    # error and the stream just never produces a frame). Sized with headroom
    # over the largest configured camera, and pinned by a builder test.
    parts.append('<visual><quality shadowsize="4096"/>'
                 '<global offwidth="1280" offheight="960"/></visual>')

    parts.append("<worldbody>")
    parts.append(workbench_inner)
    for arm_id, s in zip(arms, subtrees):
        parts.append(_inject_wristcam(s.worldbody_inner, f"{arm_id}_"))
    parts.extend(cube_chunks)
    # Overhead camera looking straight down at the workbench. Good for reading
    # shoulder_pan; nearly useless for shoulder_lift / elbow_flex, which is why
    # the three-quarter view below exists alongside it.
    parts.append(
        f'<camera name="overhead" pos="0 0 1.0" '
        f'xyaxes="{camera_xyaxes((0, 0, 1.0), (0, 0, 0))}" fovy="60"/>'
    )
    # Three-quarter operator view: centred, from the front, raised enough to
    # look down INTO the bench. This is the view you teleop from and the one
    # the recorder saves, so it is framed on the workspace, not on the arms'
    # extremes.
    #
    # It used to sit at z=0.35 with fovy 55 — nearly eye-level, aimed almost
    # horizontally. That kept both arms in frame with a ±45° outward pan, but
    # such a pose holds the gripper off the bench entirely, over the void; it
    # is not a pose the pick-and-place task passes through. Paying for it cost
    # ~60% of the frame in every pose the task DOES use: empty space above the
    # bench's far edge and in front of its near one. Raising the camera and
    # aiming down spends those pixels on the bench instead — both arms read
    # bigger, and all three cubes clear the arms that used to hide them.
    #
    # Coverage now: |x| <= ~0.44 m across the working depth band, against
    # cubes at ±0.28 m and the place zone on the midline. A wide lateral swing
    # past that clips at the frame edge — the trade above, made on purpose.
    _tq_pos: Vec3 = (0.0, -0.72, 0.54)
    _tq_target: Vec3 = (0.0, -0.14, 0.04)
    parts.append(
        f'<camera name="threequarter" pos="{" ".join(str(c) for c in _tq_pos)}" '
        f'xyaxes="{camera_xyaxes(_tq_pos, _tq_target)}" fovy="46"/>'
    )
    # Over-the-shoulder view: behind the mounts, looking along the arms — the
    # same geometry as the passthrough operator's own eyes (the
    # "behind" stance, the default). In this frame the operator's right is
    # frame-right, so hand and gripper agree on screen; the threequarter view
    # above faces the mounts and is the "front" stance's counterpart. Pulled
    # up and back far enough that the arm towers don't eat the work area.
    _os_pos: Vec3 = (0.0, 0.44, 0.56)
    _os_target: Vec3 = (0.0, -0.16, 0.04)
    parts.append(
        f'<camera name="overshoulder" pos="{" ".join(str(c) for c in _os_pos)}" '
        f'xyaxes="{camera_xyaxes(_os_pos, _os_target)}" fovy="48"/>'
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
