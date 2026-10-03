// Small DOM + UX helpers shared across tabs.

function h(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  // .flat(Infinity) absorbs any accidentally-nested arrays (e.g. a bare
  // .map() result dropped into a children array without being spread).
  const flatChildren = [].concat(children).flat(Infinity);
  for (const child of flatChildren) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function toast(message, type = "info") {
  const stack = document.getElementById("toast-stack");
  const el = h("div", { class: `toast ${type}` }, message);
  stack.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

function ratingBadge(rating) {
  const r = rating || "unknown";
  return h("span", { class: `badge rating-${r}` }, [h("span", { class: `rating-dot ${r}` }), r]);
}

function debounce(fn, wait = 350) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

// ------------------------- Subset tab strip (Caption + Edit tabs) -------------------------
// A rail of little tabs for quickly switching between datasets: every
// folder under the datasets root, plus any folder anywhere on disk the
// user has explicitly added via "+ Add subset" - and, separately, every
// immediate subfolder of whatever folder is currently loaded (the
// common "10_concept" style layout, where the real image subsets live
// one level below the folder you'd naturally point this at) so those
// show up as their own tabs too instead of silently looking empty.
const SubsetTabs = (() => {
  // Per-container "which folder's children are we showing as the 'In
  // this folder' group" - kept stable across clicking between siblings,
  // so picking one subset doesn't make its neighbors disappear (they'd
  // otherwise be recomputed as children of whatever's now loaded, and a
  // leaf subset - the normal case, just full of images - has none).
  const _browseRoots = new Map(); // containerId -> folder path
  const _browseChildren = new Map(); // containerId -> that folder's children (cached)

  function samePath(a, b) {
    if (!a || !b) return false;
    const norm = (p) => p.replace(/[\\/]+$/, "").toLowerCase();
    return norm(a) === norm(b);
  }

  function isRootOrKnownChild(path, root, children) {
    return samePath(path, root) || children.some((c) => samePath(c.path, path));
  }

  function tabEl(entry, activePath, onSelect, containerId, render) {
    const isActive = samePath(entry.path, activePath);
    const tab = h("button", {
      type: "button",
      class: `subset-tab${isActive ? " active" : ""}`,
      title: entry.path,
      onclick: () => onSelect(entry.path),
    }, [
      h("span", { class: "subset-tab-name" }, entry.name),
      h("span", { class: "subset-tab-count" }, String(entry.image_count)),
    ]);
    if (entry.external) {
      tab.appendChild(h("span", {
        class: "subset-tab-remove",
        title: "Remove this tab (your files are untouched)",
        onclick: async (e) => {
          e.stopPropagation();
          try {
            await Api.removeSubset(entry.path);
            render(containerId, activePath, onSelect);
          } catch (err) {
            toast(err.message, "error");
          }
        },
      }, "×"));
    }
    return tab;
  }

  async function render(containerId, activePath, onSelect) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let subsets = [];
    try {
      subsets = (await Api.listSubsets()).subsets || [];
    } catch { /* tabs are a convenience layer - the manual path field above still works */ }

    // Subfolders of a stable "root" folder - this is what makes a
    // "<dataset>/10_concept/20_other" layout usable: loading the outer
    // folder shows nothing directly, but its subsets appear here as
    // one-click tabs. The root only changes when the newly active
    // folder isn't part of the current sibling group at all (a genuinely
    // different dataset was loaded) - selecting a sibling within the
    // group keeps the whole group visible instead of collapsing it.
    let root = _browseRoots.get(containerId) || null;
    let children = _browseChildren.get(containerId) || [];

    if (!activePath) {
      root = null;
      children = [];
      _browseRoots.delete(containerId);
      _browseChildren.delete(containerId);
    } else if (!root || !isRootOrKnownChild(activePath, root, children)) {
      root = activePath;
      try {
        const data = await Api.browse(root);
        children = (data.dirs || []).filter((d) => !subsets.some((s) => samePath(s.path, d.path)));
      } catch {
        children = []; // root might not be browsable (e.g. not created yet) - fine, just skip
      }
      _browseRoots.set(containerId, root);
      _browseChildren.set(containerId, children);
    }
    // else: activePath is the current root or one of its already-known
    // children/siblings - keep showing the same group, just re-render
    // below with the new tab marked active.

    container.innerHTML = "";

    if (subsets.length) {
      if (children.length) container.appendChild(h("div", { class: "subset-sidebar-label" }, "Datasets"));
      subsets.forEach((s) => container.appendChild(tabEl(s, activePath, onSelect, containerId, render)));
    }

    if (children.length) {
      container.appendChild(h("div", { class: "subset-sidebar-label" }, "In this folder"));
      children.forEach((d) => container.appendChild(
        tabEl({ name: d.name, path: d.path, image_count: d.image_count }, activePath, onSelect, containerId, render)
      ));
    }

    container.appendChild(h("button", {
      type: "button",
      class: "subset-tab subset-tab-add",
      title: "Add a folder from anywhere on your PC as a subset tab",
      onclick: () => {
        FolderBrowser.open(null, async (path) => {
          try {
            await Api.addSubset(path);
            onSelect(path);
          } catch (err) {
            toast(err.message, "error");
          }
        });
      },
    }, "+ Add subset"));
  }

  return { render };
})();

// ------------------------- Folder browser modal -------------------------
const FolderBrowser = (() => {
  let currentPath = null;
  let onChoose = null;

  const backdrop = () => document.getElementById("folder-modal");
  const pathEl = () => document.getElementById("folder-modal-path");
  const listEl = () => document.getElementById("folder-modal-list");
  const rootsEl = () => document.getElementById("folder-modal-roots");

  function renderRoots(roots) {
    const el = rootsEl();
    el.innerHTML = "";
    for (const r of roots || []) {
      const isHere = currentPath && r.path.replace(/[\\/]+$/, "").toLowerCase() === currentPath.replace(/[\\/]+$/, "").toLowerCase();
      el.appendChild(h("button", {
        type: "button",
        class: `modal-root-chip${isHere ? " active" : ""}`,
        onclick: () => refresh(r.path),
      }, r.label));
    }
  }

  async function refresh(path) {
    try {
      const data = await Api.browse(path);
      currentPath = data.path;
      pathEl().textContent = data.path;
      renderRoots(data.roots);
      const list = listEl();
      list.innerHTML = "";
      if (data.parent) {
        list.appendChild(rowEl("..", data.parent, null, () => refresh(data.parent)));
      }
      for (const d of data.dirs) {
        list.appendChild(rowEl(d.name, d.path, d.image_count, () => refresh(d.path)));
      }
      if (data.dirs.length === 0 && !data.parent) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No subfolders here.")]));
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function rowEl(name, path, imageCount, onClick) {
    const row = h("div", { class: "modal-dir-row", onclick: onClick }, [
      h("span", {}, "📁"),
      h("span", {}, name),
    ]);
    if (imageCount !== null && imageCount !== undefined) {
      row.appendChild(h("span", { class: "img-count" }, imageCount ? `${imageCount} images` : ""));
    }
    return row;
  }

  function open(startPath, callback) {
    onChoose = callback;
    backdrop().classList.add("open");
    refresh(startPath || null);
  }

  function close() {
    backdrop().classList.remove("open");
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("folder-modal-cancel").addEventListener("click", close);
    document.getElementById("folder-modal-choose").addEventListener("click", () => {
      if (onChoose && currentPath) onChoose(currentPath);
      close();
    });
    document.getElementById("folder-modal-new").addEventListener("click", async () => {
      const name = prompt("New folder name:");
      if (!name) return;
      const newPath = currentPath.replace(/[\\/]+$/, "") + "/" + name;
      try {
        await Api.mkdir(newPath);
        refresh(newPath);
      } catch (err) {
        toast(err.message, "error");
      }
    });
    backdrop().addEventListener("click", (e) => {
      if (e.target === backdrop()) close();
    });
  });

  return { open, close };
})();

// ------------------------- File browser modal (pick a single file) -------------------------
const FileBrowser = (() => {
  let currentPath = null;
  let extFilter = null;
  let onChoose = null;

  const backdrop = () => document.getElementById("file-modal");
  const pathEl = () => document.getElementById("file-modal-path");
  const listEl = () => document.getElementById("file-modal-list");

  async function refresh(path) {
    try {
      const data = await Api.browseFiles(path, extFilter);
      currentPath = data.path;
      pathEl().textContent = data.path;
      const list = listEl();
      list.innerHTML = "";
      if (data.parent) {
        list.appendChild(dirRow("..", data.parent, () => refresh(data.parent)));
      }
      for (const d of data.dirs) {
        list.appendChild(dirRow(d.name, d.path, () => refresh(d.path)));
      }
      for (const f of (data.files || [])) {
        list.appendChild(fileRow(f.name, f.path));
      }
      if (data.dirs.length === 0 && (data.files || []).length === 0 && !data.parent) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "Nothing here.")]));
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function dirRow(name, path, onClick) {
    return h("div", { class: "modal-dir-row", onclick: onClick }, [h("span", {}, "📁"), h("span", {}, name)]);
  }

  function fileRow(name, path) {
    return h("div", { class: "modal-dir-row", onclick: () => { if (onChoose) onChoose(path); close(); } }, [
      h("span", {}, "📄"), h("span", {}, name),
    ]);
  }

  function open(startPath, ext, callback) {
    onChoose = callback;
    extFilter = ext || null;
    backdrop().classList.add("open");
    refresh(startPath || null);
  }

  function close() {
    backdrop().classList.remove("open");
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("file-modal-cancel").addEventListener("click", close);
    backdrop().addEventListener("click", (e) => {
      if (e.target === backdrop()) close();
    });
  });

  return { open, close };
})();

// ------------------------- Active dataset (shared across tabs) -------------------------
// Collect, Edit, and Caption all work on "a folder"; this remembers the
// most recent one so moving between tabs - and adding it to Train -
// doesn't mean re-browsing the filesystem every time. Persisted in
// localStorage so it survives a page reload too.
const ActiveDataset = (() => {
  let folder = localStorage.getItem("animaStudio.activeDataset") || "";
  const listeners = [];

  function set(path) {
    if (!path || path === folder) return;
    folder = path;
    localStorage.setItem("animaStudio.activeDataset", path);
    listeners.forEach((fn) => fn(folder));
  }

  function get() {
    return folder;
  }

  function onChange(fn) {
    listeners.push(fn);
  }

  return { set, get, onChange };
})();


function openLightbox(src, captionText, action) {
  document.getElementById("lightbox-img").src = src;
  const capEl = document.getElementById("lightbox-caption");
  if (captionText) {
    capEl.textContent = captionText;
    capEl.style.display = "block";
  } else {
    capEl.style.display = "none";
  }
  const actionBtn = document.getElementById("lightbox-action");
  if (action) {
    actionBtn.textContent = action.label;
    actionBtn.className = `btn primary${action.active ? "" : " ghost"}`;
    actionBtn.style.display = "inline-flex";
    actionBtn.onclick = (e) => {
      e.stopPropagation();
      action.onClick();
    };
  } else {
    actionBtn.style.display = "none";
    actionBtn.onclick = null;
  }
  document.getElementById("lightbox").classList.add("open");
}
document.addEventListener("DOMContentLoaded", () => {
  const lb = document.getElementById("lightbox");
  lb.addEventListener("click", (e) => {
    if (e.target.id !== "lightbox-img" && e.target.id !== "lightbox-action") lb.classList.remove("open");
  });
});
