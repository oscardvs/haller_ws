# SO-101 MJCF — vendor log

## Source

Vendored from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100).

- Upstream commit: `b846dd12bc459d776cccb3dee0b1d02acbf7a9c7`
- Date pulled: 2026-05-23

## SO-100 vs SO-101

The upstream model is `trs_so_arm100` (SO-100). SO-101 differs from SO-100
primarily in the gripper assembly; the 6-DOF arm chain is identical. We
ship the SO-100 MJCF unchanged for the arm; if gripper visuals/contact
matter for a future task, replace `assets/SO_ARM100_Gripper_*.stl` with
SO-101 meshes (TheRobotStudio/SO-ARM100 repo).

## Local edits

(none yet — record any future hand-edits here with rationale and a `git diff`-style summary)

## Refresh procedure

```bash
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$TMP/m"
rsync -a --delete --exclude='CHANGELOG.md' --exclude='README.md' \
      "$TMP/m/trs_so_arm100/" sim/assets/so101/
git -C "$TMP/m" rev-parse HEAD   # paste into the entry above
```
