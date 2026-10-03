document.addEventListener("DOMContentLoaded", async () => {
  const titles = {
    collect: ["Collect", "— gather images from booru sites or the web"],
    edit: ["Edit", "— erase signatures and watermarks before captioning"],
    caption: ["Caption", "— tag your dataset before training"],
    train: ["Train", "— configure and launch an Anima LoRA run"],
    monitor: ["Monitor", "— watch samples, checkpoints, and loss as it trains"],
  };

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
      const [title, sub] = titles[view];
      document.getElementById("topbar-title").innerHTML = `${title}<span class="dim">${sub}</span>`;
    });
  });

  Collect.init();
  Edit.init();
  Caption.init();
  Theme.init();
  try {
    await Train.init();
  } catch (err) {
    toast(`Couldn't load training settings: ${err.message}`, "error");
  }
  Monitor.init();

  // Best-effort initial backend reachability check, silent on failure.
  try {
    const url = document.getElementById("backend-url").value.trim();
    const result = await Api.backendPing(url);
    const dot = document.getElementById("backend-dot");
    dot.classList.add(result.reachable ? "ok" : "bad");
    document.getElementById("backend-status-text").textContent = result.reachable ? "Backend online" : "Backend offline";
  } catch {
    /* backend not running yet - fine, the user hasn't necessarily started it */
  }
});
