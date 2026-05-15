"""Visual helpers for the probabilistic shielding tutorial."""

from __future__ import annotations


def render_simplex_projection_svg() -> str:
    """Render a tiny-MDP simplex projection diagram as an SVG string."""
    return """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 600" width="1040" height="600" role="img" aria-label="Toy MDP and probability simplex projection for probabilistic shielding">
  <defs>
    <marker id="simplex-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#374151" />
    </marker>
    <marker id="simplex-red-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#b91c1c" />
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>

  <text x="28" y="34" font-family="sans-serif" font-size="18" font-weight="700" fill="#111827">Projection geometry on a tiny MDP</text>
  <text x="28" y="58" font-family="sans-serif" font-size="12" fill="#4b5563">A distribution over actions is safe when its expected reach-unsafe risk stays within the budget q.</text>

  <rect x="28" y="82" width="340" height="486" rx="8" fill="#f8fafc" stroke="#d1d5db"/>
  <text x="48" y="113" font-family="sans-serif" font-size="14" font-weight="700" fill="#111827">Toy safety values</text>

  <circle cx="96" cy="170" r="34" fill="#eff6ff" stroke="#2563eb" stroke-width="2.5"/>
  <text x="96" y="166" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#111827">s0</text>
  <text x="96" y="185" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1d4ed8">beta=0.02</text>

  <circle cx="264" cy="134" r="30" fill="#dcfce7" stroke="#16a34a" stroke-width="2.5"/>
  <text x="264" y="130" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">safe</text>
  <text x="264" y="148" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#166534">beta=0</text>

  <circle cx="264" cy="206" r="30" fill="#fee2e2" stroke="#dc2626" stroke-width="2.5"/>
  <text x="264" y="202" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">unsafe</text>
  <text x="264" y="220" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#991b1b">beta=1</text>

  <path d="M129 162 C171 150, 196 140, 233 135" fill="none" stroke="#374151" stroke-width="1.7" marker-end="url(#simplex-arrow)"/>
  <path d="M129 178 C171 190, 196 199, 233 205" fill="none" stroke="#374151" stroke-width="1.7" marker-end="url(#simplex-arrow)"/>

  <text x="52" y="260" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">Action risks from s0</text>
  <rect x="50" y="276" width="288" height="142" rx="7" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="50" y1="312" x2="338" y2="312" stroke="#e5e7eb"/>
  <line x1="50" y1="348" x2="338" y2="348" stroke="#e5e7eb"/>
  <line x1="50" y1="384" x2="338" y2="384" stroke="#e5e7eb"/>
  <text x="70" y="300" font-family="sans-serif" font-size="11" font-weight="700" fill="#4b5563">action</text>
  <text x="156" y="300" font-family="sans-serif" font-size="11" font-weight="700" fill="#4b5563">risk</text>
  <text x="240" y="300" font-family="sans-serif" font-size="11" font-weight="700" fill="#4b5563">pure action</text>
  <text x="70" y="335" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">a0</text>
  <text x="156" y="335" font-family="sans-serif" font-size="12" fill="#111827">0.02</text>
  <text x="240" y="335" font-family="sans-serif" font-size="12" fill="#166534">safe</text>
  <text x="70" y="371" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">a1</text>
  <text x="156" y="371" font-family="sans-serif" font-size="12" fill="#111827">0.08</text>
  <text x="240" y="371" font-family="sans-serif" font-size="12" fill="#166534">safe</text>
  <text x="70" y="407" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">a2</text>
  <text x="156" y="407" font-family="sans-serif" font-size="12" fill="#111827">0.20</text>
  <text x="240" y="407" font-family="sans-serif" font-size="12" fill="#991b1b">unsafe</text>

  <rect x="50" y="452" width="288" height="74" rx="7" fill="#ffffff" stroke="#d1d5db"/>
  <text x="70" y="477" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">Budget at s0: q=0.10</text>
  <text x="70" y="498" font-family="sans-serif" font-size="11" fill="#4b5563">Mixed actions are safe when</text>
  <text x="70" y="515" font-family="sans-serif" font-size="11" fill="#4b5563">their expected risk is at most q.</text>

  <rect x="396" y="82" width="616" height="486" rx="8" fill="#f8fafc" stroke="#d1d5db"/>
  <text x="416" y="113" font-family="sans-serif" font-size="14" font-weight="700" fill="#111827">Action probability simplex</text>
  <text x="416" y="136" font-family="sans-serif" font-size="12" fill="#4b5563">0.02*pi0 + 0.08*pi1 + 0.20*pi2 &lt;= q = 0.10</text>
  <text x="416" y="156" font-family="sans-serif" font-size="11" fill="#4b5563">Dashed boundary: expected risk = q.</text>

  <rect x="792" y="104" width="196" height="62" rx="7" fill="#ffffff" stroke="#d1d5db"/>
  <rect x="810" y="122" width="14" height="14" fill="#dcfce7" stroke="#16a34a"/>
  <text x="834" y="134" font-family="sans-serif" font-size="11" fill="#166534">safe distributions</text>
  <rect x="810" y="144" width="14" height="14" fill="#fee2e2" stroke="#ef4444"/>
  <text x="834" y="156" font-family="sans-serif" font-size="11" fill="#991b1b">clipped away by budget</text>

  <polygon points="455,520 680,180 930,520" fill="#ffffff" stroke="#111827" stroke-width="2"/>
  <polygon points="455,520 680,180 722,237 666,520" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5" opacity="0.9"/>
  <polygon points="666,520 722,237 930,520" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" opacity="0.9"/>
  <line x1="666" y1="520" x2="722" y2="237" stroke="#b91c1c" stroke-width="2.3" stroke-dasharray="7 5"/>

  <text x="455" y="548" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">pi(a0)</text>
  <text x="680" y="164" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">pi(a1)</text>
  <text x="930" y="548" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#111827">pi(a2)</text>

  <circle cx="666" cy="520" r="5" fill="#b91c1c"/>
  <text x="666" y="541" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#7f1d1d">P02=(0.556,0,0.444)</text>
  <circle cx="722" cy="237" r="5" fill="#b91c1c"/>
  <text x="736" y="230" font-family="sans-serif" font-size="10" fill="#7f1d1d">P12=(0,0.833,0.167)</text>

  <circle cx="688" cy="407" r="7" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>
  <text x="688" y="389" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1d4ed8">projected</text>
  <text x="688" y="424" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1d4ed8">(1/3,1/3,1/3)</text>

  <line x1="922" y1="514" x2="699" y2="411" stroke="#b91c1c" stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#simplex-red-arrow)"/>
  <circle cx="930" cy="520" r="7" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
  <text x="840" y="496" font-family="sans-serif" font-size="11" font-weight="700" fill="#991b1b">risky pure action</text>

  <text x="532" y="322" font-family="sans-serif" font-size="11" font-weight="700" fill="#166534">safe distributions</text>
  <text x="788" y="378" font-family="sans-serif" font-size="11" font-weight="700" fill="#991b1b">clipped away</text>
</svg>
""".strip()
