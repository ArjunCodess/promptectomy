import type { Metadata } from "next";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./globals.css";

const title = "Promptectomy | Replace LLM calls with tested Python";
const description =
  "Find repetitive LLM calls in your logs, test a Python replacement against verified examples, and keep a report you can rerun.";
const productionUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL;

export const metadata: Metadata = {
  title,
  description,
  metadataBase: URL.parse(productionUrl ? `https://${productionUrl}` : "http://localhost:3000") ?? undefined,
  openGraph: { title, description, type: "website" },
  twitter: { card: "summary_large_image", title, description },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
