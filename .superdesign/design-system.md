# Design System: Organic Tech (Preset A)

This document defines the design tokens and specifications for the Amazon RecSys Interactive Dashboard UI.

## Palette
- **Primary / Primary Container:** `#2E4036` (Moss Green - representing clinical biotech / organic structure)
- **Accent / Highlight:** `#CC5833` (Clay Orange - representing activity, high-signal alerts, and primary buttons)
- **Background / Page Base:** `#F2F0E9` (Cream White - luxury aesthetic)
- **Text / Contrast Text:** `#1A1A1A` (Charcoal Black - high readability)
- **Surface / Cards:** `#EAE7DD` (Darker Cream - container surfaces)
- **Code / Monospace Background:** `#0E110F` (Dark Forest Black - for code and telemetry output)

## Typography
- **Headings (Main titles, Section labels):** `Plus Jakarta Sans`, sans-serif
- **Sub-Headings (Feature titles, metadata):** `Outfit`, sans-serif
- **Italic Drama (Emphasis, quotes, luxury statements):** `Cormorant Garamond`, serif (italicized)
- **Monospace Data (JSON output, telemetry feeds, numbers):** `IBM Plex Mono`, monospace

## Visual Styles & Elements
- **Borders & Corners:** Use large, premium rounded corners for all containers (`rounded-[2rem]`). Avoid sharp edges.
- **Glassmorphism:** Sub-components use backdrop-blur filters for depth layers (`bg-white/40 backdrop-blur-md`).
- **Global Texture:** Apply a fine noise background overlay to mitigate digital flatness:
  ```xml
  <svg class="pointer-events-none fixed inset-0 z-50 h-full w-full opacity-5 mix-blend-multiply" xmlns="http://www.w3.org/2000/svg">
    <filter id="noiseFilter">
      <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/>
    </filter>
    <rect width="100%" height="100%" filter="url(#noiseFilter)"/>
  </svg>
  ```

## Micro-Interactions
- **Magnetic Buttons:** Scale transitions on hover with smooth cubic-bezier timing:
  `scale(1.03)` with `transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)`.
- **Text Lifts:** Text links shift slightly upward (`translate-y-[-1px]`) on hover.
