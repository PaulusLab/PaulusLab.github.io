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
