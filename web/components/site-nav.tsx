"use client";

import { useRef } from "react";
import { GithubIcon, Menu01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function SiteNav({ repository }: { repository: string }) {
  const menu = useRef<HTMLDivElement>(null);
  const links = [
    { href: "#workflow", label: "Workflow" },
    { href: "#questions", label: "FAQ" },
    { href: repository, label: "GitHub" },
  ];

  return (
    <nav aria-label="Main navigation">
      <div className="desktop-nav">
        {links.map(({ href, label }) => (
          <a key={href} href={href}>
            {href === repository && (
              <HugeiconsIcon icon={GithubIcon} size={16} strokeWidth={1.5} aria-hidden="true" />
            )}
            {label}
          </a>
        ))}
      </div>
      <div className="mobile-nav">
        <button type="button" className="menu-toggle" popoverTarget="site-menu">
          <HugeiconsIcon icon={Menu01Icon} size={16} strokeWidth={1.5} aria-hidden="true" />
          Menu
        </button>
        <div ref={menu} id="site-menu" className="menu-links" popover="auto">
          {links.map(({ href, label }) => (
            <a key={href} href={href} onClick={() => menu.current?.hidePopover()}>
              {label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
