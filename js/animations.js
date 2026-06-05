/**
 * Animation Engine
 * Scroll-triggered animations, counters, and interactive effects
 */

(function () {
  'use strict';

  // --- Scroll-Triggered Animations via IntersectionObserver ---
  const observerOptions = {
    root: null,
    rootMargin: '0px 0px -50px 0px',
    threshold: 0.1,
  };

  const animationObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        // Once visible, stop observing (one-time animation)
        animationObserver.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Observe all elements with animation classes
  function initScrollAnimations() {
    const animatedElements = document.querySelectorAll(
      '.animate-on-scroll, .fade-in, .slide-left, .slide-right, .scale-up, .blur-in'
    );
    animatedElements.forEach((el) => animationObserver.observe(el));
  }

  // --- Count-Up Animation ---
  function animateCountUp(element) {
    const target = parseInt(element.getAttribute('data-target'), 10);
    const duration = parseInt(element.getAttribute('data-duration') || '1500', 10);
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(eased * target);

      element.textContent = current.toLocaleString();

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }

  // Observer for count-up elements
  const countUpObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCountUp(entry.target);
          countUpObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  function initCountUps() {
    document.querySelectorAll('.count-up').forEach((el) => {
      countUpObserver.observe(el);
    });
  }

  // --- Progress Bar Animation ---
  const barObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const bar = entry.target;
          const targetWidth = bar.getAttribute('data-width');
          // Delay slightly for visual impact
          setTimeout(() => {
            bar.style.width = targetWidth;
          }, 200);
          barObserver.unobserve(bar);
        }
      });
    },
    { threshold: 0.3 }
  );

  function initProgressBars() {
    document.querySelectorAll('.progress-bar__fill, .benchmark-bar').forEach((bar) => {
      const targetWidth = bar.style.width || bar.getAttribute('data-width');
      if (targetWidth) {
        bar.setAttribute('data-width', targetWidth);
        bar.style.width = '0%';
        barObserver.observe(bar);
      }
    });
  }

  // --- Typewriter Effect ---
  function initTypewriters() {
    document.querySelectorAll('.typewriter').forEach((el) => {
      const text = el.getAttribute('data-text') || el.textContent;
      el.textContent = '';
      el.style.borderRight = '2px solid var(--color-accent)';

      let i = 0;
      const speed = parseInt(el.getAttribute('data-speed') || '50', 10);

      function type() {
        if (i < text.length) {
          el.textContent += text.charAt(i);
          i++;
          setTimeout(type, speed);
        } else {
          // Blink cursor for a while then stop
          setTimeout(() => {
            el.style.borderRight = 'none';
          }, 2000);
        }
      }

      // Start when visible
      const tw = new IntersectionObserver(
        (entries) => {
          if (entries[0].isIntersecting) {
            type();
            tw.unobserve(el);
          }
        },
        { threshold: 0.5 }
      );
      tw.observe(el);
    });
  }

  // --- Parallax Scroll Effect ---
  function initParallax() {
    const parallaxElements = document.querySelectorAll('.parallax');
    if (parallaxElements.length === 0) return;

    window.addEventListener(
      'scroll',
      () => {
        const scrollY = window.pageYOffset;
        parallaxElements.forEach((el) => {
          const speed = parseFloat(el.getAttribute('data-speed') || '0.3');
          const rect = el.getBoundingClientRect();
          const offset = (rect.top + scrollY) * speed;
          el.style.transform = `translateY(${scrollY * speed - offset}px)`;
        });
      },
      { passive: true }
    );
  }

  // --- Mouse Trail Effect (optional, for interactive sections) ---
  function initMouseTrail(container) {
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '1';
    container.style.position = 'relative';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const particles = [];
    const maxParticles = 50;

    function resize() {
      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    container.addEventListener('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      for (let i = 0; i < 2; i++) {
        particles.push({
          x,
          y,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          life: 1,
          size: Math.random() * 3 + 1,
        });
      }

      // Limit particles
      while (particles.length > maxParticles) {
        particles.shift();
      }
    });

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.02;

        if (p.life <= 0) {
          particles.splice(i, 1);
          continue;
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(89, 139, 231, ${p.life * 0.5})`;
        ctx.fill();
      }

      requestAnimationFrame(animate);
    }
    animate();
  }

  // --- Initialize Everything ---
  function init() {
    initScrollAnimations();
    initCountUps();
    initProgressBars();
    initTypewriters();
    initParallax();

    // Init mouse trail on interactive areas
    document.querySelectorAll('.interactive-area').forEach(initMouseTrail);
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose for dynamic content
  window.AnimationEngine = {
    initScrollAnimations,
    initCountUps,
    initProgressBars,
    initTypewriters,
    initParallax,
    initMouseTrail,
    animateCountUp,
    reinit: init,
  };
})();
