# Demystifying Serving Frameworks

Memory, parallelism, and internals of vLLM, TensorRT-LLM, SGLang, and beyond.

## Live Site

Deployed via GitHub Pages. See the [website](https://joonhyung-lee.github.io/demystifying-serving-frameworks/).

## Structure

```
.
├── index.html                          # Homepage (blog index)
├── blog/
│   └── demystifying-serving-frameworks/
│       └── index.html                  # Blog post
├── css/
│   ├── style.css                       # Main styles
│   ├── blog.css                        # Blog post styles
│   └── animations.css                  # Animation utilities
├── js/
│   ├── main.js                         # UI components (tabs, accordions, etc.)
│   └── animations.js                   # Scroll animations, counters, effects
├── assets/
│   └── images/
├── .github/workflows/deploy.yml        # GitHub Pages deployment
└── .nojekyll
```

## Adding a New Post

1. Create `blog/<post-slug>/index.html`
2. Use the blog post template (copy from an existing post)
3. Add a `<div class="blog-item">` entry in `index.html`

## Animation Classes

```html
<!-- Scroll-triggered -->
<div class="animate-on-scroll">Fades in on scroll</div>
<div class="slide-left stagger-2">Slides from left with delay</div>
<div class="scale-up">Scales up on scroll</div>
<div class="blur-in">Blurs in on scroll</div>

<!-- Continuous -->
<div class="float">Gentle float</div>
<div class="glow">Glow effect</div>

<!-- Hover -->
<div class="hover-lift">Lifts on hover</div>
<div class="hover-scale">Scales on hover</div>

<!-- Interactive (mouse trail particles) -->
<div class="interactive-area">Canvas particle trail</div>
```

## Local Development

```bash
python -m http.server 8000
# http://localhost:8000
```

## Deployment

Push to `gh-pages-website`. GitHub Actions deploys automatically.

Setup: Repository Settings > Pages > Source: "GitHub Actions"
