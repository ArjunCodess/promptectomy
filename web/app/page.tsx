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
  { command: "ingest", detail: "Import native JSONL, OpenAI, LangSmith, or OpenTelemetry logs" },
  { command: "analyze", detail: "Group calls. Rank tasks by cost and frequency." },
  { command: "suggest", detail: "Get Python and assumptions from your task description" },
  { command: "review", detail: "Sample task inputs with model answers hidden" },
  { command: "attest", detail: "Sign checked answers with a reviewer key" },
  { command: "test", detail: "Score the Python against signed holdout examples" },
];

const questions = [
  ["Do I need a Groq key?", "Only if you use a Groq model for suggest. The local rules, demo, analysis, and verification work without a key."],
  ["What gets sent to Groq?", "Suggest sends the task description and, if needed, asks Groq to repair its own code. Your traces and reviewed answers stay on your machine."],
  ["What does a reviewer sign?", "Review prepares task inputs with model answers hidden. After checking the answers against a reference, a reviewer signs them with attest."],
  ["What does PASS mean?", "The replacement met the confidence threshold on at least 20 distinct holdout inputs with signed answers. Check the report before deploying it."],
  ["What happens when there’s too little evidence?", "UNVERIFIED tells you what evidence is missing. KEEP_AI means the task should stay with the model because it depends on context or judgment."],
  ["Which log formats can I import?", "Use native JSONL, OpenAI Chat Completions or Batch exports, LangSmith runs, or OpenTelemetry GenAI spans. Step labels and token counts can be omitted."],
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
            <p className="hero-lede">Replace repetitive LLM calls with Python tested against signed reference answers</p>
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
          <h2 id="evidence-title">Your logs show what the model said.<br />Reference answers give you something to test</h2>
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
