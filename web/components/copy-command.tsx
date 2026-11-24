"use client";

import { useState } from "react";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";

export function CopyCommand({ command }: { command: string }) {
  const [message, setMessage] = useState("");
  const copied = message === "Commands copied.";

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setMessage("Commands copied.");
    } catch {
      setMessage("Select the commands to copy them.");
    }
  }

  return (
    <div className="copy-control">
      <span role="status">{message}</span>
      <Button variant="outline" onClick={copy} aria-label="Copy commands">
        <HugeiconsIcon icon={copied ? Tick02Icon : Copy01Icon} aria-hidden="true" data-icon="inline-start" />
        {copied ? "Copied" : "Copy"}
      </Button>
    </div>
  );
}
