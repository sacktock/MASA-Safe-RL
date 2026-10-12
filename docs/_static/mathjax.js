/* Render arithmatex on initial load and after instant navigation. */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      let pending = MathJax.startup.promise;
      document$.subscribe(() => {
        pending = pending.then(() => {
          MathJax.typesetClear();
          MathJax.texReset();
          return MathJax.typesetPromise();
        }).catch((error) => console.error("MathJax rendering failed:", error));
      });
    },
  },
};
