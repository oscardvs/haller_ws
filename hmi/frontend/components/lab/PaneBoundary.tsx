"use client";

/**
 * A widget that throws must not take the workspace with it.
 *
 * There is no error boundary anywhere else in this app, which means a
 * render-phase throw anywhere under the cockpit unmounts the whole tab tree.
 * That blast radius is wildly out of proportion to most causes: a
 * `/lab/datasets/trace` response missing `names` made `trace.names.map` throw
 * inside a chart and cost the operator the episode list, the mark buttons and
 * the dialogs — over a chart they were not looking at.
 *
 * Guarding each field is the same fix written twenty times and remembered
 * nineteen. This is the general form, applied where the review and train panes
 * compose their parts: a subordinate widget fails in its own box, says so, and
 * leaves everything around it working.
 *
 * It deliberately does NOT swallow the error. The message is shown in place
 * and logged, because a boundary that renders a tidy blank is how a broken
 * backend goes unnoticed for a week. This is damage control, not a fix, and it
 * should look like one.
 */
import { Component, type ErrorInfo, type ReactNode } from "react";

import { Refusal } from "@/components/lab/ui";

type Props = {
  /** What failed, in the operator's words: "the trace charts", "the run log". */
  what: string;
  children: ReactNode;
};

type State = { error: Error | null };

export class PaneBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept loud. The operator sees the box below; whoever is debugging needs
    // the component stack, and there is nowhere else it would go.
    console.error(`lab: ${this.props.what} failed to render`, error, info);
  }

  /** Called by the pane when the thing that failed has been replaced — a new
   *  episode, a new run — so a transient bad response is not permanent. */
  componentDidUpdate(prev: Props) {
    if (this.state.error !== null && prev.children !== this.props.children) {
      this.setState({ error: null });
    }
  }

  render() {
    const { error } = this.state;
    if (error === null) return this.props.children;
    return (
      <Refusal tone="fault">
        {this.props.what} could not be drawn — {error.message}. everything else
        on this pane still works; the backend sent something this build did not
        expect.
      </Refusal>
    );
  }
}
