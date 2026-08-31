// hmi/frontend/__tests__/labUnits.test.tsx
//
// Whether the numbers on this page are this robot's degrees.
//
// Every value the Lab draws (a joint trace, a gripper guide, a sweep total,
// the thresholds a verdict was decided by) comes out of `observation.state`
// with no unit attached, and the page renders all of them identically. On a
// Haller recording they are degrees. On a corpus pulled off the Hub they may
// be normalised [-100, 100], and nothing in a plot separates the two: both are
// small signed numbers on joint-shaped trajectories.
//
// So the failure this file is about is not a crash. It is a foreign dataset
// that looks exactly like one of ours, gets marked against thresholds in the
// wrong unit, and is trained on. The backend already knows the difference
// (`lab/catalog.py::dataset_units`); the assertions here are that the operator
// is told, on the surfaces where the decision is actually made.
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { DatasetShelf } from "@/components/lab/DatasetShelf";
import { ReviewPane } from "@/components/lab/ReviewPane";
import {
  unitsAlert,
  type DatasetSummary,
  type DatasetUnits,
  type LabEpisode,
} from "@/lib/lab";

const FOREIGN = "armnet/armnetbench_v01_lerobot_bimanual_so101";
const OURS = "local/so101_pick_cube";

/** The backend's own sentence for an undeclared corpus, shortened. It is
 *  written server-side so the page and the co-training caller that refuses the
 *  same dataset cannot describe it two different ways, which is exactly why
 *  this fixture carries the server's words and not the component's. */
const SERVER_NOTE =
  "Units unknown: this dataset was not recorded by Haller and does not " +
  "declare what unit its joint columns are in. It may be degrees or " +
  "normalised [-100, 100]; the two look identical in a plot. Values are " +
  "shown exactly as recorded and must not be read as degrees.";

function summary(over: Partial<DatasetSummary> = {}): DatasetSummary {
  return {
    repo_id: OURS,
    task: "Pick up the battery and place it in the box",
    episodes: 3,
    frames: 1800,
    duration_s: 60,
    size_bytes: 1_000_000,
    marks: { keep: 3, reject: 0, unset: 0, train: 3 },
    is_backup: false,
    rig: "solo",
    stale: false,
    units: { declared: true, state_unit: "deg", convertible: true },
    ...over,
  };
}

function ep(over: Partial<LabEpisode> = {}): LabEpisode {
  return {
    episode_index: 0, label: 1, frames: 372, duration_s: 12.4, share: 0.03,
    task: "Pick up the cube", verdict: "PASS", reasons: [],
    mark: "unset", note: null, tags: [], ...over,
  };
}

function routeFetch(routes: Record<string, unknown>) {
  vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL) => {
      const url = String(input);
      for (const [fragment, body] of Object.entries(routes)) {
        if (url.includes(fragment)) {
          return new Response(JSON.stringify(body), { status: 200 });
        }
      }
      return new Response(JSON.stringify({}), { status: 200 });
    },
  );
}

/** Every element wearing the units warning, whichever surface drew it. */
const alerts = () => [...document.querySelectorAll("[data-units-alert]")];

/** The shelf card for a repo. Found by its accessible name because the whole
 *  card is one button and the chips inside it are decorative. */
function cardFor(repoId: string): HTMLElement {
  const card = screen.getAllByRole("button").find(
    (b) => (b.getAttribute("aria-label") ?? "").includes(repoId),
  );
  if (!card) throw new Error(`no card for ${repoId}`);
  return card;
}

beforeEach(() => {
  vi.restoreAllMocks();
  Element.prototype.scrollIntoView = function scrollIntoView() {};
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

/* ─── the rule ────────────────────────────────────────────────────────── */

describe("unitsAlert: what is worth interrupting an operator for", () => {
  it("says nothing about a dataset whose every joint is calibrated", () => {
    // The map to normalised is exact and reversible on this one, so there is
    // nothing to be careful about. A chip that appeared here too would be a
    // chip nobody reads by the third dataset.
    expect(
      unitsAlert({ declared: true, state_unit: "deg", convertible: true }),
    ).toBeNull();
  });

  it("says nothing when the backend never sent the field", () => {
    // An older build, not a bad dataset. Every optional field on this surface
    // degrades to "hide me" rather than to a warning, because a warning that
    // fires on a backend that simply predates the check trains the operator to
    // dismiss the real one.
    expect(unitsAlert(undefined)).toBeNull();
    expect(unitsAlert(null)).toBeNull();
  });

  it("warns that an undeclared corpus is not in this robot's degrees", () => {
    const alert = unitsAlert({
      declared: false, state_unit: null, convertible: false,
    });

    expect(alert?.label).toBe("units unknown");
    expect(alert?.note).toMatch(/must not be read as this robot's degrees/i);
  });

  it("distinguishes a half-calibrated dataset from an undeclared one", () => {
    // Different problems with different fixes: one was never recorded here,
    // the other was recorded across a recalibration and can be repaired. A
    // single "units?" for both sends the operator to the wrong one.
    const alert = unitsAlert({
      declared: true, state_unit: "deg", convertible: false,
    });

    expect(alert?.label).toBe("units partial");
  });

  it("prefers the server's sentence over its own copy", () => {
    // The detail endpoint's note counts the joints and names the ones with no
    // calibrated range. Composing a second sentence in the browser is how the
    // page and the backend end up describing one dataset differently.
    const units: DatasetUnits = {
      declared: true,
      source: "haller_joint_calibration",
      state_unit: "deg",
      convertible: false,
      joints_total: 12,
      joints_calibrated: 11,
      uncalibrated: ["right_gripper"],
      reason: "1 of 12 joints have no usable calibrated range",
      note: "Partly calibrated, so NOT convertible: right_gripper.",
    };

    expect(unitsAlert(units)?.note).toBe(units.note);
  });
});

/* ─── the shelf ───────────────────────────────────────────────────────── */

describe("DatasetShelf: the warning before the dataset is opened", () => {
  it("marks the foreign corpus and leaves the calibrated one alone", async () => {
    // Both cards are drawn by the same component from the same list, so the
    // assertion has to be that exactly ONE of them is flagged. A card that
    // warns about everything says nothing.
    routeFetch({
      "/lab/datasets": {
        datasets: [
          summary({
            repo_id: FOREIGN,
            rig: "bimanual",
            units: { declared: false, state_unit: null, convertible: false },
          }),
          summary(),
        ],
      },
    });

    render(<DatasetShelf onOpen={() => {}} />);

    await screen.findByText("units unknown");
    expect(alerts()).toHaveLength(1);
    // On the FOREIGN card, not merely somewhere on the shelf, and in the
    // card's accessible name rather than only in its colour: the chip is
    // decorative and out of the tab order, so the name is the only way this
    // reaches an operator who is not looking at the pixels.
    const card = cardFor(FOREIGN);
    expect(within(card).getByText("units unknown")).toBeInTheDocument();
    expect(card.getAttribute("aria-label")).toContain("units unknown");
  });

  it("hangs the explanation off the card the operator can hover", async () => {
    // The chips inside a card are pointer-events-none so the card stays one
    // control for the keyboard, which means the card owns the tooltip. The
    // repo-id has to survive alongside it: it is what the operator hovers for
    // on a shelf of truncated names.
    routeFetch({
      "/lab/datasets": {
        datasets: [
          summary({
            repo_id: FOREIGN,
            units: { declared: false, state_unit: null, convertible: false },
          }),
        ],
      },
    });

    render(<DatasetShelf onOpen={() => {}} />);

    await screen.findByText("units unknown");
    const card = cardFor(FOREIGN);
    expect(card.title).toContain(FOREIGN);
    expect(card.title).toMatch(/must not be read as this robot's degrees/i);
  });
});

/* ─── review ──────────────────────────────────────────────────────────── */

describe("ReviewPane: the warning beside the numbers it is about", () => {
  const EPISODES = [ep({ episode_index: 0, label: 1 })];

  function mount(units: unknown) {
    routeFetch({
      "/lab/datasets/detail": {
        repo_id: FOREIGN, root: "/d", fps: 30, robot_type: "so_follower",
        video_keys: ["top"], features: {}, rig: "bimanual", units,
        episodes: EPISODES,
      },
      "/lab/datasets/episodes": { total: 1, episodes: EPISODES },
      "/lab/datasets/split": { order: [], train_episodes: [], eval_episodes: [] },
      "/lab/datasets/trace": {
        names: ["shoulder_pan.pos", "gripper.pos"],
        t: [0, 0.03], state: [[1, 2], [90, 40]], action: [[1, 2], [90, 40]],
        gripper: { "gripper.pos": [90, 40] },
      },
      // The pane looks its own row up in the listing for the card-shaped
      // fields; this one is deliberately silent about units so the assertion
      // below can only pass off the detail's block.
      "/lab/datasets": { datasets: [] },
    });
    render(<ReviewPane repoId={FOREIGN} onPickDataset={() => {}} />);
  }

  it("shows the server's sentence while an undeclared corpus is triaged", async () => {
    // This is the pane where a take is marked keep or reject, and the grader's
    // thresholds behind those marks are floats in the dataset's own unit. If
    // that unit is undeclared, the operator has to be able to see it without
    // leaving the page they are deciding on.
    mount({
      declared: false, source: "undeclared", state_unit: null,
      convertible: false, joints_total: 12, joints_calibrated: 0,
      uncalibrated: [], reason: "no haller_joint_calibration block",
      note: SERVER_NOTE,
    });

    const chip = await screen.findByText("units unknown");

    expect(chip.title).toBe(SERVER_NOTE);
  });

  it("stays quiet on a dataset the backend can convert exactly", async () => {
    mount({
      declared: true, source: "haller_joint_calibration", state_unit: "deg",
      convertible: true, joints_total: 12, joints_calibrated: 12,
      uncalibrated: [], reason: null, note: "Haller-calibrated.",
    });

    // Waited for rather than asserted immediately: the pane paints before its
    // detail lands, so an absent chip is not evidence until the response has
    // been rendered.
    await waitFor(() => expect(screen.getByText("bimanual")).toBeInTheDocument());
    expect(alerts()).toHaveLength(0);
  });
});
