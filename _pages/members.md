---
layout: page
title: Members
permalink: /members/
description: Lab members, their GitHub profiles, and what they're currently working on.
nav: true
nav_order: 5
---

<div class="projects">
  <div class="row row-cols-1 row-cols-md-3">
  {% for member in site.data.members %}
    {% include member.liquid %}
  {% endfor %}
  </div>
</div>

---

**Are you a lab member?** Add yourself in two steps — no local setup needed:

1. Open [`_data/members.yml`](https://github.com/PaulusLab/PaulusLab.github.io/edit/main/_data/members.yml) — this opens GitHub's web editor directly on that file.
2. Copy the example entry, fill in your name, role, GitHub username, and a one-line note on what you're currently working on, then commit. If you don't have write access, GitHub will offer to open a pull request for you automatically.
