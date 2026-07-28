// hmi/frontend/app/arm/[id]/page.tsx
// Unlinked deep link — one arm, full width. The cockpit shows both arms side
// by side; this is the route for putting a single arm on its own screen.
import { ArmPanel } from "@/components/ArmPanel";
import { DeepLinkChrome } from "@/components/DeepLinkChrome";

export default async function ArmDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <>
      <DeepLinkChrome label={`Arm · ${id}`} />
      <main className="p-3">
        <ArmPanel armId={id} />
      </main>
    </>
  );
}
