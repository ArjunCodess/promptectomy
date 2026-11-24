import { ImageResponse } from "next/og";

export const alt = "Promptectomy. Cut the calls you can replace.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const colors = {
  paper: "#101216",
  ink: "#f1f2f3",
  accent: "#e9b858",
  muted: "#c0c5cb",
};

export default function SocialImage() {
  return new ImageResponse(
    (
      <div style={{
        background: colors.paper, color: colors.ink, display: "flex", flexDirection: "column",
        justifyContent: "center", width: "100%", height: "100%", padding: "80px",
        borderTop: `10px solid ${colors.accent}`, gap: "32px",
      }}>
        <div style={{ color: colors.accent, fontSize: 32 }}>Promptectomy</div>
        <div style={{ fontSize: 88, letterSpacing: "-5px", lineHeight: 1.02 }}>
          Cut the calls you can replace.
        </div>
        <div style={{ color: colors.muted, fontSize: 30 }}>Replace LLM calls with tested Python.</div>
      </div>
    ),
    size,
  );
}
