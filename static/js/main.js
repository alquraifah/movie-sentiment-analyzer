/* main.js – Movie Sentiment Analyzer */

// ── Chart.js global dark-theme defaults ───────────────────────────────────────
if (typeof Chart !== "undefined") {
  Chart.defaults.color          = "#8b8fa8";
  Chart.defaults.borderColor    = "rgba(255,255,255,0.07)";
  Chart.defaults.font.family    = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size      = 12;

  Chart.defaults.plugins.tooltip.backgroundColor = "rgba(18,18,42,0.95)";
  Chart.defaults.plugins.tooltip.borderColor      = "rgba(124,58,237,0.4)";
  Chart.defaults.plugins.tooltip.borderWidth      = 1;
  Chart.defaults.plugins.tooltip.padding          = 10;
  Chart.defaults.plugins.tooltip.titleColor       = "#f0f0f8";
  Chart.defaults.plugins.tooltip.bodyColor        = "#a0a0b8";
  Chart.defaults.plugins.tooltip.cornerRadius     = 8;
}

// ── Animate elements in on scroll ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const cards = document.querySelectorAll(".glass-card, .feature-card, .insight-card");

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.style.opacity   = "1";
            e.target.style.transform = "translateY(0)";
            observer.unobserve(e.target);
          }
        });
      },
      { threshold: 0.08 }
    );

    cards.forEach((el) => {
      el.style.opacity   = "0";
      el.style.transform = "translateY(20px)";
      el.style.transition = "opacity 0.45s ease, transform 0.45s ease";
      observer.observe(el);
    });
  }

  // Highlight the current nav link
  const path  = window.location.pathname;
  const links = document.querySelectorAll(".nav-link");
  links.forEach((a) => {
    if (a.getAttribute("href") === path) {
      a.classList.add("active");
    }
  });
});
