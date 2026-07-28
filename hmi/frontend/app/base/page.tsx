// hmi/frontend/app/base/page.tsx
// Unlinked deep link — the cockpit's Operate tab is the everyday route to the
// base. This one survives for bookmarks and second-monitor use.
import { BasePanel } from "@/components/BasePanel";
import { DeepLinkChrome } from "@/components/DeepLinkChrome";

export default function BasePage() {
  return (
    <>
      <DeepLinkChrome label="Base" />
      <main className="p-3">
        <BasePanel />
      </main>
    </>
  );
}
