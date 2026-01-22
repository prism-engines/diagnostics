Why Jekyll is a good starting framework

Simple: You write content in Markdown (like README files)
Free hosting: GitHub Pages runs Jekyll natively
No server needed: It generates static files, so it's fast and cheap to host
Big community: Lots of help available when you get stuck

What the Editorial Jekyll version includes
From the GitHub repo I found, it adds:

Blog post pagination (automatic "next/previous" links)
Tag pages (group content by topic)
Search functionality (using lunr.js)
Comments (via Disqus, optional)

What working with it looks like
Instead of editing HTML, you'd create a file like:
markdown---
title: Vector Layer Documentation
layout: post
tags: [architecture, metrics]
---

The Vector Layer computes 51 behavioral metrics from raw sensor data...
Save that as _posts/2024-01-20-vector-layer.md and Jekyll automatically generates the full HTML page with your sidebar, styling, navigation — everything.

Want me to set up a starter Jekyll project using the Editorial theme? I can create the basic structure with a few example pages so you can see how it works and start adding your Ørthon content.yes12:22 PM5 stepsCreate the Jekyll directory structure manually since git clone is blockedbashmkdir -p /home/claude/orthon-docs/{_layouts,_includes,_posts,_data,assets/{css,js,images},_sass}Outputexit code 0Architected Jekyll site structure with Editorial theme components.Good, now let me create the Jekyll structure from scratch. I'll build it based on the Editorial theme's design principles. I need to create:

_config.yml - Jekyll configuration
_layouts/default.html - Main template with sidebar
_layouts/post.html - Post/article template
_includes/sidebar.html - The sidebar navigation
_includes/header.html - Header partial
index.html - Home page
A few sample posts for Ørthon documentation
Basic CSS that mimics Editorial's style
Gemfile for dependencies

Let me start building this out.Let me build the Jekyll structure from scratch — I'll create it customized for Ørthon from the start.32 stepsPresented filePrepared to elucidate package contents and usage instructions.Great, now let me provide a summary of what's in the package and how to use it.Done! Here's your Jekyll starter project.
What's Inside
orthon-docs/
├── _config.yml          # Site settings (title, URL, etc.)
├── _layouts/
│   ├── default.html     # Main template with sidebar
│   └── doc.html         # Documentation page wrapper
├── _docs/               # Your documentation goes here
│   ├── overview.md      # System overview (example)
│   ├── vector-layer.md  # Vector layer docs (example)
│   └── quickstart.md    # Getting started guide (example)
├── assets/
│   ├── css/main.css     # Editorial-style theme
│   └── js/main.js       # Accordion menu behavior
├── index.html           # Home page
├── Gemfile              # Ruby dependencies
└── README.md            # Setup instructions
To Run Locally

Install Jekyll (one-time):

bash   gem install bundler jekyll

Install dependencies:

bash   cd orthon-docs
   bundle install

Start the server:

bash   bundle exec jekyll serve

Open http://localhost:4000 — you'll see the site with the sidebar, home page, and example docs.

How to Add Content
Create a new file in _docs/ like geometry-layer.md:
markdown---
title: Geometry Layer
description: Analyzes pairwise relationships between sensors
---

Your content here in Markdown...
Then add a link in _layouts/default.html under the appropriate menu section.
The README in the zip has more details on deployment to GitHub Pages when you're ready.
