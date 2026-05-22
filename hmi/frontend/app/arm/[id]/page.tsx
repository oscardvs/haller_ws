// hmi/frontend/app/arm/[id]/page.tsx
import { ArmPanel } from "@/components/ArmPanel";

export default async function ArmDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <main className="p-3">
      <ArmPanel armId={id} />
    </main>
  );
}
