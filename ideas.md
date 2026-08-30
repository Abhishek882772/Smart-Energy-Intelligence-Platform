# SmartEnergy Frontend — Design Direction

## Three possible approaches

### Theme Name: Solar Atelier
**Very Brief Intro:** A light, editorial energy-intelligence interface with warm paper tones, solar amber accents, and precise data visualization. It feels like a premium climate-tech field journal translated into a modern control surface.
**Probability:** 0.07

### Theme Name: Gridline Observatory
**Very Brief Intro:** A dark, observatory-inspired monitoring system with deep ink surfaces, electric mint signals, and restrained diagnostic highlights. It makes the data feel instrument-like without becoming a generic cyber dashboard.
**Probability:** 0.04

### Theme Name: Verdant Utility
**Very Brief Intro:** A calm, domestic smart-home workspace using mineral green, chalk, and terracotta to make energy decisions feel practical and humane. It emphasizes approachable sustainability over technical spectacle.
**Probability:** 0.08

## Chosen Direction: Solar Atelier

### Design Movement
Contemporary editorial dashboard design, borrowing from climate-tech reports, Swiss information design, and the tactile quality of a well-made field notebook. The interface should feel intelligent and presentation-ready without looking like an enterprise admin template.

### Core Principles
1. **Instrumented, not bureaucratic:** Surface the most important signal first, with clear labels and generous breathing room.
2. **Warm precision:** Pair exact analytical visuals with paper-like neutrals and solar accents so intelligence feels approachable.
3. **Asymmetric storytelling:** Use a strong left rail, offset hero copy, wide charts, and varied card proportions instead of a uniform card grid.
4. **Explain every signal:** Alerts and recommendations should state what changed, why it matters, and what the user can do next.

### Color Philosophy
The foundation is a warm mineral canvas rather than sterile white, evoking a printed energy report or sunlit home. Deep ink provides confident contrast for navigation and key headings. Signature solar amber marks moments of attention, while moss and teal communicate healthy, measured performance. Coral is reserved for anomalies so the visual language distinguishes urgency from ordinary activity. The palette should make the dashboard feel optimistic, grounded, and legible in a college project presentation.

### Layout Paradigm
A persistent charcoal left sidebar anchors the experience like the spine of a field notebook. The main workspace begins with an offset welcome block, a slim context ribbon, then a wide primary chart paired with a narrow insight rail. Supporting content moves between full-width analytical bands and compact modules. On mobile, the rail becomes a top drawer and the workspace collapses to a single narrative column without losing hierarchy.

### Signature Elements
- **Signal ribbon:** thin amber and teal indicator lines that label the health of the data pipeline from Meter to NILM to Insight.
- **Notched cards:** mostly square cards with one subtle clipped corner or bracket detail, echoing technical print annotations without excessive rounding.
- **Field-note microcopy:** compact uppercase labels, timestamp stamps, and short explanatory notes that make the dashboard feel observed rather than merely computed.

### Interaction Philosophy
Interactions should reward curiosity and preserve context. Chart ranges switch instantly but smoothly, table rows reveal detail without abandoning the current page, and action buttons confirm mock behavior with concise feedback. Hover states use a slight lift and a stronger edge accent; focus states are visible and high-contrast. No interaction should feel ornamental or delay an analytical task.

### Animation
Use only purposeful motion: sections fade and translate upward by a few pixels on first load, KPI values can ease in once, and chart controls use a 160–220ms ease-out. Sidebar collapse should slide with the content rather than bounce. Avoid looping decoration, parallax, or large transforms. Respect `prefers-reduced-motion` by removing entrance translations and keeping state changes instant.

### Typography System
Use **DM Sans** for body copy, controls, labels, and data values because it stays crisp at dashboard sizes. Use **Fraunces** for the occasional editorial display line in the welcome area and feature callouts; its soft serif contrast gives the product a crafted climate-report personality. Headings are sentence case with tight tracking; metadata is uppercase with generous letter spacing; numbers use tabular figures where possible.

### Brand Essence
**SmartEnergy turns noisy household meter data into clear, actionable energy intelligence for people who want a more efficient home without needing to be data scientists.**

Personality: **observant, grounded, encouraging**.

### Brand Voice
Headlines are confident but human. CTAs are specific and useful rather than promotional. Microcopy explains the significance of a signal in plain language and never claims certainty beyond the available data.

Example lines:
- “See what your home is telling you.”
- “A small shift tonight could trim next month’s bill.”

### Wordmark & Logo
The mark is a compact **sun-through-circuit** symbol: a half-sun arc intersected by three small circuit traces, contained in a slightly offset square. The symbol should be legible at favicon size and appear beside the custom wordmark “SmartEnergy” in a wide, confident sans treatment. The generated logo asset is symbol-only so the wordmark remains crisp and editable in the interface.

### Signature Brand Color
**Solar Amber — `#F5A524`**. It is bright enough to signal energy and attention, warmer and more ownable than generic yellow, and pairs naturally with the deep ink rail and mineral canvas.

### File-level style reminders
- `client/src/index.css`: Solar Atelier tokens, warm mineral canvas, deep ink rail, amber signal, restrained shadows, no purple gradients.
- `client/src/App.tsx`: Keep the experience as an immersive analytical workspace with the left rail and responsive shell intact.
- `client/src/pages/Home.tsx`: Lead with narrative hierarchy, asymmetric composition, and explanatory energy insights rather than a generic admin card wall.
- `client/src/components/*`: Components should feel like reusable field instruments—clear labels, strong states, and meaningful microcopy.
