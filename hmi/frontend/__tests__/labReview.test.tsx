// hmi/frontend/__tests__/labReview.test.tsx
//
// The Lab's destructive and guarded paths.
//
// Everything under review here fails the same way: quietly, and against a
// demonstration that cannot be re-recorded. A prune that sends the wrong
// indices removes takes nobody agreed to lose; a delete gate that opens one
// keystroke early destroys a dataset the operator had not finished reading;
// an autoclassify that applies without its diff on screen rewrites the only
// record of which episodes are worth training on. None of those throw, and
// none of them look wrong afterwards — the dataset simply has fewer takes in
// it than it should.
import {
  act, cleanup, fireEvent, render, screen, waitFor, within,
} from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { AutoclassifyDialog } from "@/components/lab/AutoclassifyDialog";
import { BulkBar } from "@/components/lab/BulkBar";
import { DeleteDatasetDialog } from "@/components/lab/DeleteDatasetDialog";
import {
  DEFAULT_FILTERS, EpisodeFilters, type EpisodeFilterState,
} from "@/components/lab/EpisodeFilters";
import { EpisodeRow } from "@/components/lab/EpisodeRow";
import { PruneDialog } from "@/components/lab/PruneDialog";
import { ReviewPane } from "@/components/lab/ReviewPane";
import type { AutoclassPreview, LabEpisode, Mark } from "@/lib/lab";

const REPO = "local/so101_pick_cube";

/* ─── harness ─────────────────────────────────────────────────────────── */

/** A refusal carrying the backend's OWN sentence, so a test can assert the
 *  exact words that reach the operator rather than a placeholder. */
type Refusal = { __status: number; detail: string };

function refuse(status: number, detail: string): Refusal {
  return { __status: status, detail };
}

function isRefusal(b: unknown): b is Refusal {
  return typeof b === "object" && b !== null && "__status" in b;
}

/** Routes fetch by path so a component's boot calls do not have to be ordered.
 *  Returns the calls for assertions. Mirrors __tests__/cockpitTabs.test.tsx. */
function routeFetch(routes: Record<string, unknown>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      for (const [fragment, body] of Object.entries(routes)) {
        if (!url.includes(fragment)) continue;
        if (typeof body === "number") {
          return new Response(JSON.stringify({ detail: "nope" }), { status: body });
        }
        if (isRefusal(body)) {
          return new Response(
            JSON.stringify({ detail: body.detail }), { status: body.__status },
          );
        }
        return new Response(JSON.stringify(body), { status: 200 });
      }
      return new Response(JSON.stringify({}), { status: 200 });
    },
  );
  return { calls, spy };
}

const sent = (c: { init?: RequestInit }): unknown =>
  JSON.parse(c.init!.body as string);

function ep(over: Partial<LabEpisode> = {}): LabEpisode {
  return {
    index: 0, label: 1, frames: 372, duration_s: 12.4, share: 0.03,
    task: "Pick up the cube", verdict: "PASS", reasons: [],
    mark: "unset", note: null, tags: [], ...over,
  };
}

beforeEach(() => {
  vi.restoreAllMocks();
  // jsdom does not implement it, and ReviewPane calls it to keep the selected
  // row in view. Stubbed here rather than guarded in the component: the guard
  // would be dead code in every browser that matters.
  Element.prototype.scrollIntoView = function scrollIntoView() {};
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

/* ─── prune ───────────────────────────────────────────────────────────── */

describe("PruneDialog — the one that destroys episodes", () => {
  // Indices are NOT array positions: this is a filtered page of a dataset, so
  // idx 4 is simply not here. `expect_episodes` has to carry 2 and 5.
  const EPISODES: LabEpisode[] = [
    ep({ index: 0, label: 1, mark: "keep" }),
    ep({ index: 1, label: 2, mark: "unset" }),
    ep({
      index: 2, label: 3, mark: "reject", duration_s: 4.2,
      reasons: ["left: gripper never closed"],
    }),
    ep({ index: 3, label: 4, mark: "keep" }),
    ep({ index: 5, label: 6, mark: "reject", duration_s: 9.8, note: "arm stalled" }),
  ];

  function open(episodes: LabEpisode[] = EPISODES) {
    const onClose = vi.fn();
    const onPruned = vi.fn();
    render(
      <PruneDialog
        repoId={REPO} episodes={episodes} onClose={onClose} onPruned={onPruned}
      />,
    );
    return { onClose, onPruned };
  }

  it("asks for nothing but the click while a backup is kept", () => {
    // The backup IS the undo. Demanding a typed word on top of it teaches the
    // operator to type it by reflex, and that reflex is what carries them
    // through the one dialog where there is nothing to go back to.
    routeFetch({});
    open();

    expect(
      screen.getByRole("checkbox", { name: /keep the previous version as a backup/i }),
    ).toBeChecked();
    expect(screen.getByRole("button", { name: "remove 2 episodes" })).toBeEnabled();
    // No box to type in at all — not an empty one that is ignored.
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("holds the confirm shut until DELETE is typed exactly, with no backup", () => {
    // Without a backup this click is the end of those takes. A near-miss that
    // opens the gate means the gate is a formality.
    routeFetch({});
    open();

    fireEvent.click(
      screen.getByRole("checkbox", { name: /keep the previous version as a backup/i }),
    );
    const confirm = screen.getByRole("button", { name: "remove 2 episodes" });
    expect(confirm).toBeDisabled();

    const box = screen.getByRole("textbox", { name: /type DELETE to confirm/i });
    for (const near of ["delete", "DELETEX", "DELET", "", "  "]) {
      fireEvent.change(box, { target: { value: near } });
      expect(confirm).toBeDisabled();
    }

    fireEvent.change(box, { target: { value: "DELETE" } });
    expect(confirm).toBeEnabled();
  });

  it("trims the typed word, the same way the dataset delete does", () => {
    // Both typed gates trim, deliberately and identically: it is the same
    // operator at the same keyboard, and two rules is one of them being wrong
    // at any given moment. Trimming is safe on both — no wrong value trims to
    // a right one — and a pasted trailing space blocking someone who typed the
    // right word is friction that teaches nothing.
    routeFetch({});
    open();
    fireEvent.click(
      screen.getByRole("checkbox", { name: /keep the previous version as a backup/i }),
    );
    const confirm = screen.getByRole("button", { name: "remove 2 episodes" });
    const box = screen.getByRole("textbox", { name: /type DELETE to confirm/i });
    fireEvent.change(box, { target: { value: "  DELETE " } });
    expect(confirm).toBeEnabled();
  });

  it("re-shuts the gate when the backup is turned back off", () => {
    // Typing DELETE, changing your mind about the backup, then changing it
    // back must not leave a still-armed confirm behind the operator.
    routeFetch({});
    open();

    const backup = screen.getByRole("checkbox", {
      name: /keep the previous version as a backup/i,
    });
    fireEvent.click(backup);
    fireEvent.change(
      screen.getByRole("textbox", { name: /type DELETE to confirm/i }),
      { target: { value: "DELETE" } },
    );
    fireEvent.click(backup);
    fireEvent.click(backup);

    expect(screen.getByRole("button", { name: "remove 2 episodes" })).toBeDisabled();
  });

  it("names every episode it would destroy, and no episode it would keep", () => {
    // A count is something you agree to; a list is something you read. If the
    // list is wrong — or is only a number — the operator approves a set they
    // were never shown.
    routeFetch({});
    open();

    expect(screen.getByText("removing 2 episodes · 3 kept")).toBeInTheDocument();

    expect(screen.getByText("Ep 3")).toBeInTheDocument();
    expect(screen.getByText("(idx 2)")).toBeInTheDocument();
    expect(screen.getByText("4.2s")).toBeInTheDocument();
    expect(screen.getByText("left: gripper never closed")).toBeInTheDocument();

    expect(screen.getByText("Ep 6")).toBeInTheDocument();
    expect(screen.getByText("(idx 5)")).toBeInTheDocument();
    expect(screen.getByText("9.8s")).toBeInTheDocument();
    // No grader reason, so the operator's own note is what stands in.
    expect(screen.getByText("arm stalled")).toBeInTheDocument();

    for (const kept of ["Ep 1", "Ep 2", "Ep 4"]) {
      expect(screen.queryByText(kept)).not.toBeInTheDocument();
    }
  });

  it("posts the STORED indices of exactly the rejected takes", async () => {
    // `expect_episodes` is the guard against the dataset moving between this
    // dialog opening and this click. Sending labels (3, 6) or array positions
    // (2, 4) instead of stored indices deletes different demonstrations.
    const { calls } = routeFetch({ "/lab/datasets/prune": { run_id: "run-42" } });
    const { onClose, onPruned } = open();

    fireEvent.click(screen.getByRole("button", { name: "remove 2 episodes" }));

    await waitFor(() => {
      const post = calls.find((c) => c.url.includes("/lab/datasets/prune"));
      expect(post).toBeTruthy();
      expect(sent(post!)).toEqual({
        repo_id: REPO, backup: true, expect_episodes: [2, 5],
      });
    });
    await waitFor(() => expect(onPruned).toHaveBeenCalled());
    expect(onClose).toHaveBeenCalled();
  });

  it("carries the backup checkbox to the wire, not just to the gate", async () => {
    // A dialog that asks for DELETE and then prunes WITH a backup is only
    // annoying; one that shows the backup box ticked and prunes without is
    // how the undo turns out not to exist.
    const { calls } = routeFetch({ "/lab/datasets/prune": { run_id: "run-43" } });
    open();

    fireEvent.click(
      screen.getByRole("checkbox", { name: /keep the previous version as a backup/i }),
    );
    fireEvent.change(
      screen.getByRole("textbox", { name: /type DELETE to confirm/i }),
      { target: { value: "DELETE" } },
    );
    fireEvent.click(screen.getByRole("button", { name: "remove 2 episodes" }));

    await waitFor(() => {
      const post = calls.find((c) => c.url.includes("/lab/datasets/prune"));
      expect(sent(post!)).toEqual({
        repo_id: REPO, backup: false, expect_episodes: [2, 5],
      });
    });
  });

  it("offers no destructive control when nothing is marked reject", () => {
    // A live remove button over an empty set is a button that can only ever
    // do something nobody asked for.
    routeFetch({});
    open([ep({ index: 0, label: 1, mark: "keep" })]);

    expect(screen.getByText(/no episodes are marked reject/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^remove/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "close" })).toBeInTheDocument();
  });

  it("refuses to empty a dataset and says which door to use instead", () => {
    // Pruning every episode leaves a dataset lerobot cannot open. The operator
    // wanted the dataset gone, so send them to the delete that says so.
    routeFetch({});
    open([
      ep({ index: 0, label: 1, mark: "reject" }),
      ep({ index: 1, label: 2, mark: "reject" }),
    ]);

    expect(screen.getByText(/delete the whole dataset instead/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^remove/ })).not.toBeInTheDocument();
  });

  it("keeps the dialog open on a 409 and quotes the backend verbatim", async () => {
    // 409 is a refusal, not a failure: a run holds the dataset, which the
    // operator can clear. Closing on it loses both the reason and the list.
    const { calls } = routeFetch({
      "/lab/datasets/prune": refuse(409, "a training run is reading this dataset"),
    });
    const { onClose, onPruned } = open();

    fireEvent.click(screen.getByRole("button", { name: "remove 2 episodes" }));

    expect(
      await screen.findByText("a training run is reading this dataset"),
    ).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "prune rejected" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "remove 2 episodes" })).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(onPruned).not.toHaveBeenCalled();
    expect(calls.filter((c) => c.url.includes("/prune"))).toHaveLength(1);
  });
});

/* ─── delete dataset ──────────────────────────────────────────────────── */

describe("DeleteDatasetDialog — no backup of any kind", () => {
  function open() {
    const onClose = vi.fn();
    const onDeleted = vi.fn();
    render(
      <DeleteDatasetDialog
        repoId={REPO}
        episodes={46}
        durationS={3705}
        sizeBytes={3 * 1024 * 1024 * 1024}
        onClose={onClose}
        onDeleted={onDeleted}
      />,
    );
    return { onClose, onDeleted };
  }

  it("keeps delete shut until the typed name is the repo id", () => {
    // The mistake this guards is not "I did not mean to delete" — it is "I did
    // not notice WHICH dataset was selected". A prefix or a case-fold opening
    // the gate defeats the only thing that makes you read the name.
    routeFetch({});
    open();

    const confirm = screen.getByRole("button", { name: "delete this dataset" });
    expect(confirm).toBeDisabled();

    const box = screen.getByRole("textbox", { name: /type the dataset name to confirm/i });
    for (const near of [
      "local/so101_pick_cub",        // one character short
      "LOCAL/SO101_PICK_CUBE",       // same name, wrong case
      "local/so101_pick_cube_old",   // the prune backup — a different dataset
      "so101_pick_cube",             // no namespace
    ]) {
      fireEvent.change(box, { target: { value: near } });
      expect(confirm).toBeDisabled();
    }

    fireEvent.change(box, { target: { value: REPO } });
    expect(confirm).toBeEnabled();
  });

  it("accepts surrounding whitespace, because a pasted name carries it", () => {
    // Deliberate: the gate trims. Documented here so that turning it into a
    // byte-exact compare is a decision someone makes, not a silent edit that
    // starts rejecting every copy-pasted repo id.
    routeFetch({});
    open();

    fireEvent.change(
      screen.getByRole("textbox", { name: /type the dataset name to confirm/i }),
      { target: { value: `  ${REPO} ` } },
    );
    expect(screen.getByRole("button", { name: "delete this dataset" })).toBeEnabled();
  });

  it("states the cost in takes, hours and bytes", () => {
    // A byte count alone reads as housekeeping. What is destroyed is hours of
    // demonstration, and the dialog has to say so before it is agreed to.
    routeFetch({});
    open();

    expect(screen.getByText("46")).toBeInTheDocument();
    expect(screen.getByText("1h 01m")).toBeInTheDocument();
    expect(screen.getByText("3.0 GB")).toBeInTheDocument();
    expect(screen.getByText(REPO)).toBeInTheDocument();
  });

  it("puts the typed gate on the wire, escaped, next to the repo id", async () => {
    // The server compares `confirm` to `repo_id` byte for byte. A client that
    // checks only in the dialog can delete a dataset the operator never named.
    const { calls } = routeFetch({
      "/lab/datasets": { repo_id: REPO, root: "/data/x", freed_bytes: 1024 },
    });
    const { onDeleted } = open();

    fireEvent.change(
      screen.getByRole("textbox", { name: /type the dataset name to confirm/i }),
      { target: { value: REPO } },
    );
    fireEvent.click(screen.getByRole("button", { name: "delete this dataset" }));

    await waitFor(() => {
      const del = calls.find((c) => c.init?.method === "DELETE");
      expect(del).toBeTruthy();
      expect(del!.url).toContain("/lab/datasets?");
      expect(del!.url).toContain("repo_id=local%2Fso101_pick_cube");
      expect(del!.url).toContain("confirm=local%2Fso101_pick_cube");
    });
    await waitFor(() => expect(onDeleted).toHaveBeenCalled());
  });
});

/* ─── autoclassify ────────────────────────────────────────────────────── */

/** The move cell as it reads end to end: `from`, the arrow, the screen-reader
 *  "to", then the destination. Asserting it whole is what pins the DIRECTION —
 *  `keep → reject` and `reject → keep` are the same two words reordered, and
 *  they are opposite decisions about a demonstration. */
function moveText(from: Mark, to: Mark): string {
  return `${from}→to${to}`;
}

describe("AutoclassifyDialog — bulk rewrite of the review", () => {
  const PREVIEW: AutoclassPreview = {
    token: "tok-7f3a2c",
    diff: [
      {
        episode: 2, from: "unset", to: "reject",
        why: "gripper never closed", confidence: 0.94,
      },
      {
        episode: 6, from: "keep", to: "reject",
        why: "tracking error 14.2 deg", confidence: 0.71,
      },
    ],
  };

  function open() {
    const onClose = vi.fn();
    const onApplied = vi.fn();
    render(
      <AutoclassifyDialog repoId={REPO} onClose={onClose} onApplied={onApplied} />,
    );
    return { onClose, onApplied };
  }

  it("shows the diff before any apply control exists", async () => {
    // THE central guarantee. An apply that is reachable while the traces are
    // still being read is a button that rewrites a review nobody has seen —
    // and the review is the only record of which takes are worth training on.
    routeFetch({ "/lab/datasets/autoclass/preview": PREVIEW });
    open();

    expect(screen.getByText(/reading the traces/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /apply/i })).not.toBeInTheDocument();

    // Both spellings of the episode, and the direction, and the reason.
    expect(await screen.findByText("Ep 3")).toBeInTheDocument();
    expect(screen.getByText("(idx 2)")).toBeInTheDocument();
    expect(screen.getByText("Ep 7")).toBeInTheDocument();
    expect(screen.getByText("(idx 6)")).toBeInTheDocument();

    for (const c of PREVIEW.diff) {
      const row = screen.getByText(c.why).closest("div")!;
      expect(within(row).getByText(c.from).parentElement!)
        .toHaveTextContent(moveText(c.from, c.to));
    }
    expect(screen.getByText("94%")).toBeInTheDocument();
    expect(screen.getByText("71%")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "apply 2 changes" })).toBeEnabled();
  });

  it("applies by TOKEN and sends no diff", async () => {
    // The token binds the diff to the dataset state it was computed against;
    // the server recomputes and 409s if it moved. Re-sending the diff would
    // apply decisions against a state the operator never saw.
    const { calls } = routeFetch({
      "/lab/datasets/autoclass/preview": PREVIEW,
      "/lab/datasets/autoclass/apply": { applied: 2, batch: "batch-91" },
    });
    const { onApplied } = open();

    fireEvent.click(await screen.findByRole("button", { name: "apply 2 changes" }));

    await waitFor(() => {
      const post = calls.find((c) => c.url.includes("/autoclass/apply"));
      expect(post).toBeTruthy();
      expect(sent(post!)).toEqual({ repo_id: REPO, token: "tok-7f3a2c" });
    });
    await waitFor(() => expect(onApplied).toHaveBeenCalled());
  });

  it("stays open after applying and offers the batch back", async () => {
    // Revert is the SAFE direction and it is only reachable from this screen.
    // A dialog that closes on success leaves the operator with a rewritten
    // review and the undo handle gone.
    const { calls } = routeFetch({
      "/lab/datasets/autoclass/preview": PREVIEW,
      "/lab/datasets/autoclass/apply": { applied: 2, batch: "batch-91" },
      "/lab/datasets/autoclass/revert": { reverted: 2 },
    });
    const { onClose } = open();

    fireEvent.click(await screen.findByRole("button", { name: "apply 2 changes" }));

    const revert = await screen.findByRole("button", { name: "revert this batch" });
    expect(screen.getByRole("dialog", { name: "autoclassify" })).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();

    fireEvent.click(revert);
    await waitFor(() => {
      const post = calls.find((c) => c.url.includes("/autoclass/revert"));
      expect(post).toBeTruthy();
      // The batch from the APPLY response, not the token and not the diff.
      expect(sent(post!)).toEqual({ repo_id: REPO, batch: "batch-91" });
    });
    expect(await screen.findByText(/batch reverted/i)).toBeInTheDocument();
  });

  it("counts the DIFF, so an untouched SUSPECT is never implied to be handled", async () => {
    // `grade` leaves SUSPECT alone by design. If the header or the button
    // counted episodes considered rather than rows shown, the operator would
    // read "3 changes", see two, and assume the third was dealt with.
    routeFetch({ "/lab/datasets/autoclass/preview": PREVIEW });
    open();

    await screen.findByText("Ep 3");
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "apply 2 changes" })).toBeInTheDocument();
    expect(
      screen.getByText(/everything not listed is left alone/i),
    ).toBeInTheDocument();

    // Episode 4 is the SUSPECT one. It is not in the diff, so it must not be
    // on screen under any spelling.
    expect(screen.queryByText("Ep 5")).not.toBeInTheDocument();
    expect(screen.queryByText("(idx 4)")).not.toBeInTheDocument();
  });

  it("says the classifier agrees, and offers no apply, on an empty diff", async () => {
    // "Nothing to do" and "0 changes applied" are different sentences, and
    // only one of them leaves an armed button over an empty set.
    routeFetch({
      "/lab/datasets/autoclass/preview": { token: "tok-empty", diff: [] },
    });
    open();

    expect(
      await screen.findByText(/agrees with every mark already set/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /apply/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "close" })).toBeInTheDocument();
  });
});

/* ─── episode row ─────────────────────────────────────────────────────── */

describe("EpisodeRow — the off-by-one that deletes the wrong take", () => {
  // index and label deliberately differ, so nothing here can pass by reading
  // the wrong field and landing on the right number.
  const EP = ep({
    index: 2, label: 3, duration_s: 12.4, share: 0.03, mark: "keep",
    task: "Pick up the cube",
  });

  function handlers() {
    return {
      onSelect: vi.fn(),
      onToggleSelect: vi.fn(),
      onMark: vi.fn(),
      onNote: vi.fn(),
    };
  }

  it("shows the spoken number and the stored one, together", () => {
    // Oscar says "episode 3" and the endpoints take 2. A row that shows one of
    // the two is a row that gets the wrong index typed into a prune.
    const h = handlers();
    render(
      <EpisodeRow episode={EP} selected={false} checked={false} inEval={false} {...h} />,
    );

    expect(screen.getByText("Ep 3")).toBeInTheDocument();
    expect(screen.getByText("idx 2")).toBeInTheDocument();
    // Neither number may appear under the other's label.
    expect(screen.queryByText("Ep 2")).not.toBeInTheDocument();
    expect(screen.queryByText("idx 3")).not.toBeInTheDocument();
  });

  it("returns the active mark to unset instead of re-asserting it", () => {
    // "I have not judged this" is a real state and a different fact from "I
    // watched it and it is fine". A chip that cannot be un-clicked makes the
    // first mis-click permanent.
    const h = handlers();
    render(
      <EpisodeRow episode={EP} selected={false} checked={false} inEval={false} {...h} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "keep", pressed: true }));
    expect(h.onMark).toHaveBeenCalledWith("unset");

    fireEvent.click(screen.getByRole("button", { name: "reject", pressed: false }));
    expect(h.onMark).toHaveBeenLastCalledWith("reject");
  });

  it("does not re-select the row when the note box is clicked", () => {
    // Clicking into a note re-selects the row, which swaps the video under the
    // player — so the note gets typed against whatever loaded next.
    const h = handlers();
    render(
      <EpisodeRow episode={EP} selected={false} checked={false} inEval={false} {...h} />,
    );

    // The row does select on a plain click, or the assertion below proves
    // nothing.
    fireEvent.click(screen.getByText("12.4s · 3%"));
    expect(h.onSelect).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("textbox", { name: "note for episode 3" }));
    expect(h.onSelect).toHaveBeenCalledTimes(1);
  });

  it("badges a held-out episode, and only a held-out one", () => {
    // The val badge says the policy never sees this take. Wrong either way it
    // misreads an eval number as generalisation or as overfitting.
    const h = handlers();
    const { rerender } = render(
      <EpisodeRow episode={EP} selected={false} checked={false} inEval {...h} />,
    );
    expect(screen.getByRole("button", { name: "val" })).toBeInTheDocument();

    rerender(
      <EpisodeRow episode={EP} selected={false} checked={false} inEval={false} {...h} />,
    );
    expect(screen.queryByRole("button", { name: "val" })).not.toBeInTheDocument();
  });

  it("reports the shift modifier so the list can do the range maths", () => {
    // The row only reports the gesture; the LIST owns what a range means. A
    // row that swallows the modifier turns a shift-click into a single select
    // and the operator bulk-marks one take instead of forty.
    const h = handlers();
    render(
      <EpisodeRow episode={EP} selected={false} checked={false} inEval={false} {...h} />,
    );

    const box = screen.getByRole("checkbox", { name: "select episode 3" });
    fireEvent.click(box);
    expect(h.onToggleSelect).toHaveBeenLastCalledWith(false);

    fireEvent.click(box, { shiftKey: true });
    expect(h.onToggleSelect).toHaveBeenLastCalledWith(true);
    // Ticking a box must not also move the player.
    expect(h.onSelect).not.toHaveBeenCalled();
  });
});

/* ─── filters and bulk bar ────────────────────────────────────────────── */

describe("EpisodeFilters", () => {
  const COUNTS = { keep: 12, reject: 4, unset: 30 };

  it("clears the mark filter when its lit chip is clicked again", () => {
    // `filter_mark` takes ONE value on the wire. A chip that only ever
    // re-applies itself leaves the operator with no way back to the whole
    // list except reloading the tab.
    const onChange = vi.fn();
    const value: EpisodeFilterState = { ...DEFAULT_FILTERS, mark: "reject" };
    render(
      <EpisodeFilters value={value} onChange={onChange} tags={["grasp"]} counts={COUNTS} />,
    );

    // Matched by prefix: the chip's accessible name runs its count straight
    // onto the label ("reject4"), which is a separate nit from this behaviour.
    fireEvent.click(screen.getByRole("button", { name: /^reject/, pressed: true }));
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, mark: null });

    fireEvent.click(screen.getByRole("button", { name: /^keep/, pressed: false }));
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_FILTERS, mark: "keep" });
  });

  it("debounces the search instead of asking per keystroke", () => {
    // One request per keystroke over the USB tether is a list that lags the
    // box it is typed into, and the answers arrive out of order.
    vi.useFakeTimers();
    const onChange = vi.fn();
    render(<EpisodeFilters value={DEFAULT_FILTERS} onChange={onChange} tags={[]} />);

    const box = screen.getByRole("textbox", { name: "search task or note" });
    fireEvent.change(box, { target: { value: "c" } });
    expect(onChange).not.toHaveBeenCalled();

    act(() => { vi.advanceTimersByTime(200); });
    expect(onChange).not.toHaveBeenCalled();

    // A second key restarts the clock rather than adding a second request.
    fireEvent.change(box, { target: { value: "cu" } });
    act(() => { vi.advanceTimersByTime(200); });
    expect(onChange).not.toHaveBeenCalled();

    act(() => { vi.advanceTimersByTime(100); });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, q: "cu" });
  });
});

describe("BulkBar", () => {
  function props(over: Partial<React.ComponentProps<typeof BulkBar>> = {}) {
    return {
      count: 3,
      knownTags: ["grasp"],
      onMark: vi.fn(),
      onTag: vi.fn(),
      onUntag: vi.fn(),
      onClear: vi.fn(),
      onSelectAll: vi.fn(),
      ...over,
    };
  }

  it("is not on screen at all when nothing is selected", () => {
    // A permanently docked bar reading "0 selected" is a row of destructive
    // verbs sitting under the list at all times.
    const { container } = render(<BulkBar {...props({ count: 0 })} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("goes dead while a bulk write is in flight", () => {
    // A bulk mark is one request over a tether. A second one sent on top of an
    // unfinished one is two answers racing to become the stored mark.
    const { rerender } = render(<BulkBar {...props({ busy: true })} />);

    expect(screen.getByText("3")).toBeInTheDocument();
    const names = ["keep", "reject", "unset", "select all", "clear", "add tag",
      "remove tag grasp"];
    for (const n of names) {
      expect(screen.getByRole("button", { name: n })).toBeDisabled();
    }

    rerender(<BulkBar {...props({ busy: false })} />);
    for (const n of names) {
      expect(screen.getByRole("button", { name: n })).toBeEnabled();
    }
  });
});

/* ─── shift-range selection ───────────────────────────────────────────── */

describe("ReviewPane — shift-range selection", () => {
  // The list is sorted SERVER-side, so the visible order and the numeric order
  // are different things. Every fixture here is deliberately out of numeric
  // order, because a range implementation that walks indices instead of rows
  // passes on a sorted list and only fails on a sorted-by-duration one — which
  // is the list an operator triaging by length is actually looking at.
  const ORDER = [5, 2, 9, 1, 7];

  const EPISODES = ORDER.map((i, n) =>
    ep({ index: i, label: i + 1, duration_s: 30 - n, mark: "unset" }),
  );

  function mount() {
    const r = routeFetch({
      "/lab/datasets/detail": {
        repo_id: REPO, root: "/d", fps: 30, robot_type: "so_follower",
        video_keys: ["top"], features: {}, rig: "solo", episodes: EPISODES,
      },
      "/lab/datasets/episodes": { total: EPISODES.length, episodes: EPISODES },
      "/lab/datasets/split": { order: [], train_episodes: [], eval_episodes: [] },
      "/lab/datasets/bulk": { updated: 0 },
      "/lab/datasets/trace": {
        names: ["shoulder_pan.pos", "gripper.pos"],
        t: [0, 0.03], state: [[1, 2], [90, 40]], action: [[1, 2], [90, 40]],
        gripper: { "gripper.pos": [90, 40] },
      },
    });
    render(<ReviewPane repoId={REPO} onPickDataset={() => {}} />);
    return r;
  }

  /** The row checkboxes, in the order the list renders them. */
  async function boxes() {
    await waitFor(() =>
      expect(document.querySelectorAll("[data-episode-index]").length)
        .toBe(EPISODES.length),
    );
    return [...document.querySelectorAll<HTMLElement>("[data-episode-index]")].map(
      (row) => ({
        index: Number(row.getAttribute("data-episode-index")),
        box: within(row).getByRole("checkbox"),
      }),
    );
  }

  const selectedCount = () => {
    const strip = [...document.querySelectorAll("span")].find(
      (n) => /^\s*\d+\s+selected\s*$/.test(n.textContent ?? ""),
    );
    return strip ? Number(/(\d+)/.exec(strip.textContent ?? "")?.[1] ?? 0) : 0;
  };

  it("takes the range from the VISIBLE order, not the numeric indices", async () => {
    // Anchor on the first row (idx 5), shift-click the third (idx 9). What the
    // operator dragged across is three rows. Walking numeric indices instead
    // would sweep 5..9 — five episodes, two of which are not even on screen in
    // that span — and bulk-marking those is a silent, unrecoverable edit.
    mount();
    const rows = await boxes();
    expect(rows.map((r) => r.index)).toEqual(ORDER);

    fireEvent.click(rows[0].box);
    fireEvent.click(rows[2].box, { shiftKey: true });

    await waitFor(() => expect(selectedCount()).toBe(3));
  });

  it("ranges backwards from the anchor as readily as forwards", async () => {
    // The anchor is a position, not a lower bound. An implementation that
    // assumes the shift-click is always below it selects nothing here.
    mount();
    const rows = await boxes();

    fireEvent.click(rows[3].box);
    fireEvent.click(rows[1].box, { shiftKey: true });

    await waitFor(() => expect(selectedCount()).toBe(3));
  });

  it("keeps ranging from a live anchor across repeated shift-clicks", async () => {
    // What this actually guards: the anchor survives a shift-click at all. Two
    // shift-clicks in a row must both range, rather than the second degrading
    // to a single toggle because the anchor was lost.
    //
    // What it CANNOT guard, and why there is no test for it: whether the
    // anchor STAYS at row 0 or moves to the last shift-clicked row is
    // unobservable while the range is add-only, because both produce the same
    // selection set — the anchor is always already inside the selection, so
    // every subsequent range unions into the same span. `if (!shiftKey)` is
    // still the right code (it matches what the operator thinks the anchor is)
    // and it becomes observable the day the range replaces the selection
    // instead of adding to it. Whoever makes that change owes this a real test.
    mount();
    const rows = await boxes();

    fireEvent.click(rows[0].box);
    fireEvent.click(rows[2].box, { shiftKey: true });
    await waitFor(() => expect(selectedCount()).toBe(3));

    fireEvent.click(rows[4].box, { shiftKey: true });
    await waitFor(() => expect(selectedCount()).toBe(5));
  });

  it("plain-clicks toggle one row and re-anchor", async () => {
    // The un-shifted click is both the toggle and the anchor set. A row that
    // toggles without anchoring makes the next shift-click range from
    // wherever the operator last happened to be.
    mount();
    const rows = await boxes();

    fireEvent.click(rows[0].box);
    await waitFor(() => expect(selectedCount()).toBe(1));
    fireEvent.click(rows[0].box);
    await waitFor(() => expect(selectedCount()).toBe(0));

    fireEvent.click(rows[3].box);
    fireEvent.click(rows[4].box, { shiftKey: true });
    await waitFor(() => expect(selectedCount()).toBe(2));
  });
});
