/**
 * Main JavaScript
 * Tabs, accordions, and interactive components
 */

(function () {
  'use strict';

  // --- Tabs ---
  function initTabs() {
    document.querySelectorAll('.tabs').forEach((tabContainer) => {
      const buttons = tabContainer.querySelectorAll('.tabs__btn');
      const panels = tabContainer.querySelectorAll('.tabs__panel');

      buttons.forEach((btn) => {
        btn.addEventListener('click', () => {
          const target = btn.getAttribute('data-tab');

          buttons.forEach((b) => b.classList.remove('active'));
          panels.forEach((p) => p.classList.remove('active'));

          btn.classList.add('active');
          const targetPanel = tabContainer.querySelector(`#${target}`);
          if (targetPanel) targetPanel.classList.add('active');
        });
      });
    });
  }

  // --- Accordions ---
  function initAccordions() {
    document.querySelectorAll('.accordion__header').forEach((header) => {
      header.addEventListener('click', () => {
        const body = header.nextElementSibling;
        const isOpen = header.classList.contains('open');

        header.classList.toggle('open');
        body.classList.toggle('open');

        if (!isOpen) {
          body.style.maxHeight = body.scrollHeight + 'px';
        } else {
          body.style.maxHeight = '0';
        }
      });
    });
  }

  // --- Smooth Scroll for Anchor Links ---
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;

        const targetEl = document.querySelector(targetId);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          });
        }
      });
    });
  }

  // --- Copy Code Block ---
  function initCodeCopy() {
    document.querySelectorAll('pre code').forEach((block) => {
      const pre = block.parentElement;
      if (pre.querySelector('.copy-btn')) return;

      const btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        padding: 4px 10px;
        font-size: 12px;
        background: var(--color-accent);
        color: #fff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s ease;
        font-family: var(--font-sans);
      `;

      pre.style.position = 'relative';

      pre.addEventListener('mouseenter', () => {
        btn.style.opacity = '1';
      });
      pre.addEventListener('mouseleave', () => {
        btn.style.opacity = '0';
      });

      btn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(block.textContent);
          btn.textContent = 'Copied!';
          setTimeout(() => {
            btn.textContent = 'Copy';
          }, 1500);
        } catch {
          btn.textContent = 'Failed';
        }
      });

      pre.appendChild(btn);
    });
  }

  // --- Active TOC Highlighting ---
  function initTocHighlight() {
    const toc = document.querySelector('.toc');
    if (!toc) return;

    const tocLinks = toc.querySelectorAll('a');
    const headings = [];

    tocLinks.forEach((link) => {
      const id = link.getAttribute('href')?.replace('#', '');
      const heading = id && document.getElementById(id);
      if (heading) headings.push({ link, heading });
    });

    if (headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            tocLinks.forEach((l) => (l.style.color = ''));
            const active = headings.find((h) => h.heading === entry.target);
            if (active) {
              active.link.style.color = 'var(--color-accent)';
              active.link.style.fontWeight = '600';
            }
          }
        });
      },
      {
        rootMargin: '-80px 0px -70% 0px',
        threshold: 0,
      }
    );

    headings.forEach((h) => observer.observe(h.heading));
  }

  // --- Back to Top ---
  function initBackToTop() {
    const btn = document.createElement('button');
    btn.innerHTML = '<i class="fa-solid fa-arrow-up"></i>';
    btn.setAttribute('aria-label', 'Back to top');
    btn.style.cssText = `
      position: fixed;
      bottom: 30px;
      right: 30px;
      width: 42px;
      height: 42px;
      border-radius: 50%;
      background: var(--color-accent);
      color: #fff;
      border: none;
      cursor: pointer;
      font-size: 16px;
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.3s ease, visibility 0.3s ease, transform 0.3s ease;
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    `;

    btn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    btn.addEventListener('mouseenter', () => {
      btn.style.transform = 'scale(1.1)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.transform = 'scale(1)';
    });

    document.body.appendChild(btn);

    window.addEventListener(
      'scroll',
      () => {
        if (window.scrollY > 300) {
          btn.style.opacity = '1';
          btn.style.visibility = 'visible';
        } else {
          btn.style.opacity = '0';
          btn.style.visibility = 'hidden';
        }
      },
      { passive: true }
    );
  }

  // --- Image Lightbox ---
  function initLightbox() {
    // Create overlay once
    let overlay = document.querySelector('.lightbox-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.className = 'lightbox-overlay';
      overlay.innerHTML = '<img src="" alt="">';
      document.body.appendChild(overlay);

      overlay.addEventListener('click', () => {
        overlay.classList.remove('open');
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') overlay.classList.remove('open');
      });
    }

    const overlayImg = overlay.querySelector('img');

    // Attach to all blog images and fig-grid images
    document
      .querySelectorAll('.fig-grid figure img, .blog-content figure img, .blog-image')
      .forEach((img) => {
        if (img.dataset.lightbox) return;
        img.dataset.lightbox = '1';
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', (e) => {
          e.stopPropagation();
          overlayImg.src = img.src;
          overlayImg.alt = img.alt;
          overlay.classList.add('open');
        });
      });
  }

  // --- Initialize ---
  function init() {
    initTabs();
    initAccordions();
    initSmoothScroll();
    initCodeCopy();
    initTocHighlight();
    initBackToTop();
    initLightbox();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose for dynamic content
  window.MainUI = {
    initTabs,
    initAccordions,
    initCodeCopy,
    initTocHighlight,
    initLightbox,
    reinit: init,
  };
})();
