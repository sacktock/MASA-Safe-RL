from pathlib import Path
p = Path('docs/index.md')
s = p.read_text()
s = s.replace('# Introduction\n\n## Welcome\n\n![title card](assets/images/logo_large.png)', '''---
hide:
  - toc
---

# MASA-Safe-RL

<div class="masa-hero" markdown="1">

![MASA-Safe-RL — safe learning across agents, constraints and worlds](assets/images/logo_large.svg){ .masa-wordmark }

<p class="masa-eyebrow">Single agent · Multi-agent · Constrained RL</p>

[Get started](Get%20Started/Quick%20Start.md){ .md-button .md-button--primary }
[Explore tutorials](Tutorials/Basics.md){ .md-button }

</div>

## Welcome''')
s = s.replace('## Organization', '''<div class="masa-cards" markdown="1">
<div markdown="1">

**[Build your first experiment](Get%20Started/Basic%20Usage.md)**

Install MASA, understand its core concepts, and run a safe-learning experiment.

</div>
<div markdown="1">

**[Explore the API](Common/Constraints.md)**

Work with constraints, wrappers, temporal logic, and metrics.

</div>
<div markdown="1">

**[Understand shielding](Shielding/Overview.md)**

Explore deterministic and probabilistic approaches to safer action selection.

</div>
</div>

## Organization''')
p.write_text(s)
