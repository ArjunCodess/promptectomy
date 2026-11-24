import {
  ArrowDown01Icon,
  ArrowUpRight01Icon,
  ComputerTerminal01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { CopyCommand } from "@/components/copy-command";
import { Demo } from "@/components/demo";
import { SiteNav } from "@/components/site-nav";

const repository = "https://github.com/ArjunCodess/promptectomy";
const commands = [
  "git clone https://github.com/ArjunCodess/promptectomy",
  "cd promptectomy/core",
  "uv sync",
  "uv run python examples/demo.py",
].join("\n");

const workflow = [
  { command: "ingest", detail: "Import JSONL and mask sensitive text" },
  { command: "analyze", detail: "Group calls. Rank tasks by cost and frequency." },
  { command: "suggest", detail: "Get Python and assumptions from your task description" },
  { command: "test", detail: "Score reviewed code on verified examples" },
];

const questions = [
  ["Do I need a Groq key?", "Only if you use a Groq model for suggest. The local rules, demo, analysis, and verification work without a key."],
  ["What gets sent to Groq?", "Only the task description you write for suggest. Keep it generic. Your trace and verified answers stay on your machine."],
  ["What makes an example verified?", "A person or a trusted source checks the expected answer independently. The original model output has its own column in the report."],
  ["What does PASS mean?", "The reviewed Python met the confidence threshold on independently checked prompts from the task. Use the report and its assumptions when reviewing a production change."],
  ["What happens when there’s too little evidence?", "UNVERIFIED tells you what evidence is missing. KEEP_AI means the task should stay with the model because it depends on context or judgment."],
  ["Can I use logs from any provider?", "Yes. Export the calls as JSONL with the prompt, output, model, and token counts. Include a step label if your logger has one."],
];

export default function Home() {
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="masthead">
        <div className="shell masthead-inner">
          <a href="#main" className="wordmark">
            <HugeiconsIcon icon={ComputerTerminal01Icon} size={16} strokeWidth={1.5} aria-hidden="true" />
            Promptectomy
          </a>
          <SiteNav repository={repository} />
        </div>
      </header>
      <main id="main" tabIndex={-1}>
        <section className="shell hero" aria-labelledby="hero-title">
          <div>
            <h1 id="hero-title">Cut the calls <br />you can replace</h1>
            <p className="hero-lede">Replace repetitive LLM calls with Python tested against independently verified answers</p>
            <p className="hero-meta">Open source under MIT. Runs on your machine.</p>
            <a href="#get-started" className="action">
              Run the local demo
              <HugeiconsIcon icon={ArrowDown01Icon} size={16} strokeWidth={1.5} aria-hidden="true" />
            </a>
          </div>
          <Demo />
        </section>

        <section id="workflow" className="workflow-band" aria-labelledby="workflow-title">
          <div className="shell workflow">
            <h2 id="workflow-title">From trace to verdict</h2>
            <ol className="workflow-list">
              {workflow.map((step) => (
                <li key={step.command}>
                  <h3><code>{step.command}</code></h3>
                  <p>{step.detail}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className="shell evidence" aria-labelledby="evidence-title">
          <h2 id="evidence-title">Your logs show what the model said.<br />Verified examples show what it got right</h2>
        </section>

        <section id="questions" className="shell questions" aria-labelledby="questions-title">
          <h2 id="questions-title">FAQ</h2>
          <div className="question-list">
            {questions.map(([question, answer]) => (
              <details key={question}>
                <summary>{question}<HugeiconsIcon icon={ArrowDown01Icon} size={16} strokeWidth={1.5} aria-hidden="true" /></summary>
                <p>{answer}</p>
              </details>
            ))}
          </div>
        </section>

        <section id="get-started" className="start-band" aria-labelledby="start-title">
          <div className="shell start">
            <div>
              <h2 id="start-title">Try it with a sample log</h2>
              <p>With Python 3.11 or newer and uv, you can rank the sample calls and rerun both verdicts locally. The README shows how to bring your own logs.</p>
              <a href={`${repository}#try-it`} className="action action-secondary">
                Open the demo guide
                <HugeiconsIcon icon={ArrowUpRight01Icon} size={16} strokeWidth={1.5} aria-hidden="true" />
              </a>
            </div>
            <div className="quickstart">
              <div className="quickstart-head"><span>Terminal</span><CopyCommand command={commands} /></div>
              <pre tabIndex={0} aria-label="Local demo commands"><code>{commands}</code></pre>
            </div>
          </div>
        </section>
      </main>
      <footer className="shell footer">
        <a href="#main" className="wordmark"><HugeiconsIcon icon={ComputerTerminal01Icon} size={16} strokeWidth={1.5} aria-hidden="true" /> Promptectomy</a>
        <div>
          <a href={`${repository}#use-your-own-logs`}>Data and privacy</a>
          <a href={`${repository}/blob/main/LICENSE`}>MIT license</a>
          <a href={repository}>Source code</a>
        </div>
      </footer>
    </>
  );
}
