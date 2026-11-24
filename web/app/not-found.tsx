import Link from "next/link";

export default function NotFound() {
  return (
    <main className="shell missing">
      <h1>Page not found</h1>
      <p>The address you opened doesn’t match a page on this site.</p>
      <Link href="/" className="action">Back to Promptectomy</Link>
    </main>
  );
}
