"use client";

import { useState } from "react";
import { AlertCircleIcon, CheckmarkCircle02Icon, ComputerTerminal01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const examples = {
  tickets: {
    label: "Ticket ID extraction",
    task: "extract_ticket_id",
    verdict: "PASS",
    note: "Review the Python before shipping.",
    result: "4 / 4",
    evidence: "Original 2 / 4",
    fee: "$0.0048",
    replacement: "$0",
  },
  support: {
    label: "Support routing",
    task: "classify_support",
    verdict: "UNVERIFIED",
    note: "Verify 5 more distinct inputs to meet this policy.",
    result: "1 available",
    evidence: "4 required",
    fee: "$0.0028",
    replacement: "Unavailable",
  },
} as const;

type Example = keyof typeof examples;

export function Demo() {
  const [selected, setSelected] = useState<Example>("tickets");
  const report = examples[selected];
  const passed = selected === "tickets";

  return (
    <figure className="report">
      <div className="report-header">
        <HugeiconsIcon icon={ComputerTerminal01Icon} size={19} aria-hidden="true" />
        <span>Verification report</span>
        <span className="report-fixture">Fixture data</span>
      </div>
      <div className="report-controls" role="group" aria-label="Example task">
        {(Object.keys(examples) as Example[]).map((key) => (
          <Button
            key={key}
            variant="ghost"
            aria-pressed={selected === key}
            className="report-select"
            onClick={() => setSelected(key)}
          >
            {examples[key].label}
          </Button>
        ))}
      </div>
      <div className="report-body" aria-live="polite">
        <p className="report-command"><span aria-hidden="true">$</span> promptectomy test</p>
        <div className="report-task">
          <code>{report.task}</code>
          <Badge className={passed ? "verdict verdict-pass" : "verdict verdict-pending"}>
            <HugeiconsIcon icon={passed ? CheckmarkCircle02Icon : AlertCircleIcon} aria-hidden="true" />
            {report.verdict}
          </Badge>
        </div>
        <dl className="readouts">
          <div>
            <dt>{passed ? "Exact matches" : "Holdout examples"}</dt>
            <dd>{report.result}</dd>
            <dd className="readout-detail">{report.evidence}</dd>
          </div>
          <div>
            <dt>Monthly model fees</dt>
            <dd>{report.replacement}</dd>
            <dd className="readout-detail">Original <span>{report.fee}</span></dd>
          </div>
        </dl>
        <p className="report-note">{report.note}</p>
      </div>
      <figcaption>Same log, two tasks.</figcaption>
    </figure>
  );
}
