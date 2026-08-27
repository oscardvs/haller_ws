"use client";

/**
 * Which camera the player shows.
 *
 * The kit hardwired `video_keys[0]`. That is invisible on a one-camera rig and
 * wrong on the three-key bimanual dataset, where a bad grasp is visible in a
 * wrist view and not in the mast view — the operator was reviewing a camera
 * they never chose. So the control is present at every key count: with one key
 * it is an inert chip reading that key rather than a hidden control, because
 * "this dataset has one camera" and "the picker was dropped" have to look
 * different.
 */
import { Segmented } from "@/components/lab/ui";

/** The backend strips this from `video_keys`; a raw `info.json` key still
 *  carries it. Only the LABEL is shortened — every request takes the raw key,
 *  because that is what `/lab/datasets/video` and `ep.videos[...]` are keyed
 *  by. */
const IMAGE_PREFIX = "observation.images.";

export function cameraLabel(key: string): string {
  return key.startsWith(IMAGE_PREFIX) ? key.slice(IMAGE_PREFIX.length) : key;
}

export function CameraPicker({
  keys,
  value,
  onChange,
  disabled = false,
}: {
  keys: string[];
  value: string | null;
  onChange: (k: string) => void;
  disabled?: boolean;
}): React.ReactElement {
  if (keys.length > 1) {
    return (
      <Segmented
        label="camera"
        className="shrink-0"
        disabled={disabled}
        // "" matches no option, so an unpicked camera reads as unpicked
        // rather than silently checking the first key.
        value={value ?? ""}
        onChange={onChange}
        options={keys.map((k) => ({ value: k, label: cameraLabel(k), hint: k }))}
      />
    );
  }

  // One key (or none): the same well and face as the segmented control above,
  // minus the interaction, so the header strip does not change shape between
  // the solo and the bimanual dataset.
  const only = keys[0] ?? null;
  return (
    <span className="inline-flex h-8 shrink-0 items-center rounded-md bg-muted p-1">
      <span
        title={only ?? undefined}
        className={
          "inline-flex h-6 items-center rounded-sm px-2 label-micro " +
          (only ? "bg-card text-foreground" : "text-muted-foreground")
        }
      >
        {only ? cameraLabel(only) : "no cameras"}
      </span>
    </span>
  );
}
