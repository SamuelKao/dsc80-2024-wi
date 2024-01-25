---
layout: home
title: 🏠 Home
nav_exclude: false
nav_order: 1
---

# {{ site.tagline }}

{: .mb-2 }
{{ site.description }}
{: .fs-6 .fw-300 }

{{ site.staffersnobio }}

[Jump to the current week](#week-3-messy-data-statistical-testing){: .btn } [Lab Solutions](https://edstem.org/us/courses/51951/discussion/4183397){: .btn .btn-green }

{: .red }
**Please complete the [Pre-Lecture Reading](resources/lectures/lec06/pre-lec06.html) for Thursday 1/25's lecture before class!**

Click the 🎥 button to view the recording of a lecture/discussion.<br>Click the 📝 button to view lecture notebooks after they've been filled in during lecture.

{% for module in site.modules %}
{{ module }}
{% endfor %}