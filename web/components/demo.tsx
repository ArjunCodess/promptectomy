"use client";

import { useState } from "react";
import { ComputerTerminal01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";

const examples = {
  tickets: {
    label: "Ticket ID extraction",
    task: "extract_ticket_id",
    verdict: "PASS",
    note: "Review the Python before shipping",
    fee: "$0.0048",
    replacement: "$0.0000",
  },
  support: {
    label: "Support routing",
    task: "classify_support",
    verdict: "UNVERIFIED",
    note: "Verify 5 more distinct inputs to meet this policy",
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
        <HugeiconsIcon icon={ComputerTerminal01Icon} size={16} strokeWidth={1.5} aria-hidden="true" />
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
          <span className={passed ? "verdict verdict-pass" : "verdict"}>
            {report.verdict}
          </span>
        </div>
        <table className="report-table" aria-label="Original and replacement results">
          <thead>
            <tr>
              <th scope="col"><span className="sr-only">Metric</span></th>
              <th scope="col">Original</th>
              <th scope="col">Replacement</th>
            </tr>
          </thead>
          <tbody>
            {passed && (
              <tr>
                <th scope="row">Exact matches</th>
                <td>2 / 4</td>
                <td className="verified-value">4 / 4</td>
              </tr>
            )}
            <tr>
              <th scope="row">Monthly model fees</th>
              <td>{report.fee}</td>
              <td className={passed ? "verified-value" : "unavailable-value"}>{report.replacement}</td>
            </tr>
          </tbody>
        </table>
        {!passed && (
          <dl className="report-holdout">
            <dt>Holdout examples</dt>
            <dd><span>1</span> available, <span>4</span> required</dd>
          </dl>
        )}
        <p className="report-note">{report.note}</p>
      </div>
      <figcaption>Same log, two tasks</figcaption>
    </figure>
  );
}
