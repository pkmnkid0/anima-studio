const Collect = (() => {
  const selectedSources = new Set(["danbooru"]);
  let currentMode = "booru"; // booru | webpage | urls | local
  let results = [];
  const selected = new Set();
  let currentPage = 1;
  let isBooruSearch = false; // whether Prev/Next should re-run a booru search vs. do nothing

  function init() {
    const destInput = document.getElementById("collect-dest-dir");
    if (!destInput.value && ActiveDataset.get()) destInput.value = ActiveDataset.get();
    const localDestInput = document.getElementById("collect-local-dest-dir");
    if (!localDestInput.value && ActiveDataset.get()) localDestInput.value = ActiveDataset.get();

    document.querySelectorAll(".source-tab").forEach((tab) => {
      tab.addEventListener("click", () => selectMode(tab.dataset.source));
    });
    document.querySelectorAll(".source-pill").forEach((pill) => {
      pill.addEventListener("click", () => toggleSourcePill(pill.dataset.source));
    });
    document.getElementById("collect-search-btn").addEventListener("click", doSearch);
    document.getElementById("collect-scan-btn").addEventListener("click", doScanPage);
    document.getElementById("collect-urls-btn").addEventListener("click", doFromUrls);
    document.getElementById("collect-select-all").addEventListener("click", () => setAllSelected(true));
    document.getElementById("collect-select-none").addEventListener("click", () => setAllSelected(false));
    document.getElementById("collect-browse-dest").addEventListener("click", () => {
      FolderBrowser.open(destInput.value || null, (path) => {
        destInput.value = path;
      });
    });
    document.getElementById("collect-local-browse-dest").addEventListener("click", () => {
      FolderBrowser.open(localDestInput.value || null, (path) => {
        localDestInput.value = path;
      });
    });
    document.getElementById("collect-local-add-btn").addEventListener("click", doLocalUpload);
    document.getElementById("collect-download-btn").addEventListener("click", () => doDownload());
    document.getElementById("collect-download-all-btn").addEventListener("click", doDownloadAll);
    document.getElementById("collect-prev-btn").addEventListener("click", () => changePage(-1));
    document.getElementById("collect-next-btn").addEventListener("click", () => changePage(1));

    document.getElementById("collect-tags").addEventListener("keydown", onTagsKeydown);
    document.getElementById("collect-tags").addEventListener("input", debounce(onTagsInput, 200));
    document.getElementById("collect-tags").addEventListener("input", updateDanbooruTagHint);
    document.getElementById("collect-rating").addEventListener("change", updateDanbooruTagHint);
    updateDanbooruTagHint();
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".autocomplete-wrap")) closeAutocomplete();
    });

    document.getElementById("collect-api-keys-btn").addEventListener("click", openApiKeysModal);
    document.getElementById("apikeys-modal-close").addEventListener("click", closeApiKeysModal);
    document.getElementById("apikeys-modal-save").addEventListener("click", saveApiKeys);
    refreshKeyDots();
  }

  // ------------------------------ booru API keys ------------------------------

  const SOURCE_LABELS = { gelbooru: "Gelbooru", rule34: "Rule34" };

  async function refreshKeyDots() {
    try {
      const data = await Api.collectCredentials();
      Object.entries(data.sources || {}).forEach(([source, info]) => {
        const dot = document.getElementById(`key-dot-${source}`);
        if (dot) dot.style.display = info.configured ? "none" : "inline";
      });
    } catch {
      // non-critical - the pills just won't show the "needs a key" hint
    }
  }

  async function openApiKeysModal() {
    const wrap = document.getElementById("apikeys-fields");
    wrap.innerHTML = "<p style='font-size:12.5px;color:var(--text-dim)'>Loading…</p>";
    document.getElementById("apikeys-modal").classList.add("open");
    let data;
    try {
      data = await Api.collectCredentials();
    } catch (err) {
      wrap.innerHTML = "";
      return toast(err.message, "error");
    }
    wrap.innerHTML = "";
    Object.entries(data.sources || {}).forEach(([source, info]) => {
      const label = SOURCE_LABELS[source] || source;
      wrap.appendChild(h("div", { class: "field span-full", style: "margin-bottom:10px" }, [
        h("label", {}, `${label} ${info.configured ? "✓ configured" : "(not configured)"}`),
        h("div", { class: "row", style: "gap:8px" }, [
          h("input", { type: "text", id: `apikey-${source}-key`, placeholder: "api_key", value: info.api_key || "", style: "flex:1" }),
          h("input", { type: "text", id: `apikey-${source}-user`, placeholder: "user_id", value: info.user_id || "", style: "flex:1" }),
        ]),
        h("a", { href: info.signup_url, target: "_blank", rel: "noopener", style: "font-size:11.5px;color:var(--accent)" },
          "Get your key from your account settings ↗"),
      ]));
    });
  }

  function closeApiKeysModal() {
    document.getElementById("apikeys-modal").classList.remove("open");
  }

  async function saveApiKeys() {
    const btn = document.getElementById("apikeys-modal-save");
    btn.disabled = true;
    btn.textContent = "Saving…";
    try {
      for (const source of Object.keys(SOURCE_LABELS)) {
        const keyInput = document.getElementById(`apikey-${source}-key`);
        const userInput = document.getElementById(`apikey-${source}-user`);
        if (!keyInput) continue;
        await Api.collectSaveCredentials(source, keyInput.value.trim(), userInput.value.trim());
      }
      toast("Saved.", "success");
      await refreshKeyDots();
      closeApiKeysModal();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Save";
    }
  }

  // ------------------------------ tag autocomplete ------------------------------

  let acItems = [];
  let acIndex = -1;

  function currentToken(input) {
    const value = input.value;
    const upTo = value.slice(0, input.selectionStart ?? value.length);
    const lastSpace = upTo.lastIndexOf(" ");
    const token = upTo.slice(lastSpace + 1);
    return { token: token.replace(/^-/, ""), negated: token.startsWith("-"), start: lastSpace + 1 };
  }

  async function onTagsInput() {
    if (currentMode !== "booru" || selectedSources.size === 0) return closeAutocomplete();
    const input = document.getElementById("collect-tags");
    const { token } = currentToken(input);
    if (token.length < 2) return closeAutocomplete();
    try {
      // Autocomplete against the first selected source - ambiguous to
      // merge suggestions across sites, and this covers the common case
      // of exactly one source anyway.
      const source = [...selectedSources][0];
      const data = await Api.collectAutocomplete(source, token, 8);
      renderAutocomplete(data.results || []);
    } catch {
      closeAutocomplete();
    }
  }

  function formatCount(n) {
    if (n === null || n === undefined) return "";
    if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
    return String(n);
  }

  function renderAutocomplete(items) {
    acItems = items;
    acIndex = -1;
    const list = document.getElementById("collect-autocomplete");
    list.innerHTML = "";
    if (!items.length) return closeAutocomplete();
    items.forEach((item, idx) => {
      list.appendChild(h("div", {
        class: "autocomplete-item",
        onmousedown: (e) => { e.preventDefault(); applySuggestion(idx); },
      }, [
        h("span", { class: `cat-dot ${item.category || "general"}` }),
        h("span", { class: "name" }, item.name.replace(/_/g, " ")),
        h("span", { class: "count" }, formatCount(item.post_count)),
      ]));
    });
    list.classList.add("open");
  }

  function closeAutocomplete() {
    acItems = [];
    acIndex = -1;
    const list = document.getElementById("collect-autocomplete");
    if (list) { list.classList.remove("open"); list.innerHTML = ""; }
  }

  function applySuggestion(idx) {
    const item = acItems[idx];
    if (!item) return;
    const input = document.getElementById("collect-tags");
    const { negated, start } = currentToken(input);
    const before = input.value.slice(0, start);
    const after = input.value.slice(input.selectionStart ?? input.value.length);
    const inserted = (negated ? "-" : "") + item.name;
    input.value = `${before}${inserted} ${after}`.replace(/\s+/g, " ").replace(/^\s+/, "");
    const caret = (before + inserted + " ").length;
    input.focus();
    input.setSelectionRange(caret, caret);
    closeAutocomplete();
  }

  function highlightAutocomplete() {
    const list = document.getElementById("collect-autocomplete");
    [...list.children].forEach((el, i) => el.classList.toggle("active", i === acIndex));
  }

  function onTagsKeydown(e) {
    const list = document.getElementById("collect-autocomplete");
    const isOpen = list.classList.contains("open");
    if (isOpen && e.key === "ArrowDown") {
      e.preventDefault();
      acIndex = Math.min(acIndex + 1, acItems.length - 1);
      highlightAutocomplete();
      return;
    }
    if (isOpen && e.key === "ArrowUp") {
      e.preventDefault();
      acIndex = Math.max(acIndex - 1, 0);
      highlightAutocomplete();
      return;
    }
    if (isOpen && e.key === "Escape") {
      closeAutocomplete();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (isOpen && acIndex >= 0) applySuggestion(acIndex);
      else doSearch();
    }
  }

  function selectMode(mode) {
    currentMode = mode;
    closeAutocomplete();
    document.querySelectorAll(".source-tab").forEach((t) => t.classList.toggle("active", t.dataset.source === mode));
    document.getElementById("collect-booru-fields").style.display = mode === "booru" ? "block" : "none";
    document.getElementById("collect-webpage-fields").style.display = mode === "webpage" ? "grid" : "none";
    document.getElementById("collect-urls-fields").style.display = mode === "urls" ? "grid" : "none";
    document.getElementById("collect-local-fields").style.display = mode === "local" ? "grid" : "none";
  }

  function toggleSourcePill(source) {
    if (selectedSources.has(source)) {
      if (selectedSources.size === 1) return; // keep at least one selected
      selectedSources.delete(source);
    } else {
      selectedSources.add(source);
    }
    document.querySelectorAll(".source-pill").forEach((p) => p.classList.toggle("active", selectedSources.has(p.dataset.source)));
    updateSearchButtonLabel();
    updateDanbooruTagHint();
  }

  function updateDanbooruTagHint() {
    const hint = document.getElementById("danbooru-tag-limit-hint");
    if (!selectedSources.has("danbooru")) { hint.style.display = "none"; return; }

    const userTags = document.getElementById("collect-tags").value.trim().split(/\s+/).filter(Boolean).length;
    const ratingAddsTag = document.getElementById("collect-rating").value !== "any" ? 1 : 0;
    const total = userTags + ratingAddsTag;

    if (total > 2) {
      hint.textContent = `Danbooru only lets anonymous search send 2 tags at once - your other `
        + `${total - 2} tag${total - 2 === 1 ? "" : "s"} still apply, just as a filter on each result `
        + `afterward instead of the search itself, so this may take a little longer and return fewer `
        + `results per page than usual.`;
      hint.style.display = "block";
    } else {
      hint.style.display = "none";
    }
  }

  function updateSearchButtonLabel() {
    const btn = document.getElementById("collect-search-btn");
    if (btn.disabled) return; // mid-search - don't clobber the "Searching…" label
    btn.textContent = selectedSources.size > 1 ? `Search & merge (${selectedSources.size})` : "Search";
  }

  async function doSearch() {
    currentPage = 1;
    isBooruSearch = selectedSources.size === 1; // paging a merged multi-source result set isn't well-defined
    await runSearch();
  }

  async function changePage(delta) {
    if (!isBooruSearch) return;
    const nextPage = currentPage + delta;
    if (nextPage < 1) return;
    const previousPage = currentPage;
    currentPage = nextPage;
    const gotResults = await runSearch({ silentEmpty: delta > 0 });
    if (delta > 0 && !gotResults) {
      currentPage = previousPage; // bounced off the end - stay put
      toast("No more results.", "info");
    }
  }

  async function runSearch({ silentEmpty = false } = {}) {
    const tags = document.getElementById("collect-tags").value.trim();
    const rating = document.getElementById("collect-rating").value;
    const limit = parseInt(document.getElementById("collect-limit").value, 10) || 40;
    const sources = [...selectedSources];
    const btn = document.getElementById("collect-search-btn");
    btn.disabled = true;
    btn.textContent = sources.length > 1 ? `Searching ${sources.length} sources…` : "Searching…";

    const outcomes = await Promise.allSettled(
      sources.map((source) => Api.collectSearch({ source, tags, rating, limit, page: currentPage })),
    );

    const merged = [];
    const failedMessages = [];
    let credentialsNeeded = null;
    outcomes.forEach((outcome, i) => {
      if (outcome.status === "fulfilled") merged.push(...outcome.value.results);
      else {
        const reason = outcome.reason;
        if (reason && reason.needsCredentials) credentialsNeeded = reason.needsCredentials;
        const detail = reason && reason.message ? reason.message : "unknown error";
        failedMessages.push(sources.length > 1 ? `${sources[i]}: ${detail}` : detail);
      }
    });

    setResults(merged);
    const gotResults = merged.length > 0;
    if (credentialsNeeded) {
      toast(`${credentialsNeeded} needs an API key - click "API keys…" above the sources to add one.`, "error");
    } else if (failedMessages.length) {
      toast(failedMessages.join(" | "), "error");
    }
    if (!gotResults && !failedMessages.length && !silentEmpty) toast("No results for that search.", "info");

    btn.disabled = false;
    updateSearchButtonLabel();
    return gotResults;
  }

  function updatePaginationBar() {
    const bar = document.getElementById("collect-pagination");
    bar.style.display = isBooruSearch ? "flex" : "none";
    document.getElementById("collect-page-indicator").textContent = `Page ${currentPage}`;
    document.getElementById("collect-prev-btn").disabled = currentPage <= 1;
  }

  async function doScanPage() {
    const url = document.getElementById("collect-page-url").value.trim();
    if (!url) return toast("Enter a page URL first.", "error");
    isBooruSearch = false;
    const btn = document.getElementById("collect-scan-btn");
    btn.disabled = true;
    btn.textContent = "Scanning…";
    try {
      const data = await Api.collectScanPage({ url, same_domain_only: true });
      setResults(data.results);
      if (data.results.length === 0) toast("No images found on that page.", "info");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Scan page";
    }
  }

  async function doFromUrls() {
    const raw = document.getElementById("collect-urls").value;
    const urls = raw.split("\n").map((u) => u.trim()).filter(Boolean);
    if (urls.length === 0) return toast("Paste at least one URL.", "error");
    isBooruSearch = false;
    const data = await Api.collectFromUrls({ urls });
    setResults(data.results);
  }

  async function doLocalUpload() {
    const destDir = document.getElementById("collect-local-dest-dir").value.trim();
    const fileInput = document.getElementById("collect-local-files");
    const files = fileInput.files;
    if (!destDir) return toast("Choose a destination folder first.", "error");
    if (!files || files.length === 0) return toast("Choose at least one image first.", "error");

    const btn = document.getElementById("collect-local-add-btn");
    const spinner = document.getElementById("collect-local-spinner");
    const statusEl = document.getElementById("collect-local-status");
    btn.disabled = true;
    spinner.style.display = "inline-block";
    statusEl.textContent = `Adding ${files.length} image${files.length === 1 ? "" : "s"}…`;

    const formData = new FormData();
    formData.append("dest_dir", destDir);
    for (const file of files) formData.append("files", file);

    try {
      const resp = await fetch("/api/collect/upload_local", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      const s = data.summary;
      statusEl.textContent = `Done: ${s.ok} added, ${s.errored} failed.`;
      toast(`Added ${s.ok} images to ${data.dest_dir}`, "success");
      ActiveDataset.set(data.dest_dir);
      fileInput.value = "";
    } catch (err) {
      toast(err.message, "error");
      statusEl.textContent = "Add failed.";
    } finally {
      btn.disabled = false;
      spinner.style.display = "none";
    }
  }

  function setResults(list) {
    results = list;
    selected.clear();
    renderResults();
    updatePaginationBar();
  }

  function showPreview(idx) {
    const item = results[idx];
    const tagsText = flattenTagsForDisplay(item.tags);
    const isSelected = selected.has(idx);
    openLightbox(item.file_url || item.preview_url, tagsText, {
      label: isSelected ? "✓ Selected — click to remove" : "Select for download",
      active: isSelected,
      onClick: () => { toggleSelect(idx); showPreview(idx); },
    });
  }

  function renderResults() {
    const card = document.getElementById("collect-results-card");
    const grid = document.getElementById("collect-results-grid");
    grid.innerHTML = "";
    document.getElementById("collect-results-count").textContent = results.length ? `${results.length} found` : "";
    card.style.display = results.length ? "block" : "none";

    results.forEach((item, idx) => {
      const isSelected = selected.has(idx);
      const el = h("div", {
        class: `result-card${isSelected ? " selected" : ""}`,
        onclick: () => toggleSelect(idx),
      }, [
        h("img", { src: item.preview_url, loading: "lazy", alt: "" }),
        h("div", { class: "check" }, [
          h("svg", { width: "11", height: "9", viewBox: "0 0 11 9", html: "<path d='M1 4.5L4 7.5L10 1' stroke='white' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/>" }),
        ]),
        h("button", {
          class: "preview-btn", type: "button", title: "View & select individually",
          onclick: (e) => { e.stopPropagation(); showPreview(idx); },
        }, [
          h("svg", { width: "12", height: "12", viewBox: "0 0 16 16", fill: "none", html: "<path d='M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5Z' stroke='white' stroke-width='1.3'/><circle cx='8' cy='8' r='2' stroke='white' stroke-width='1.3'/>" }),
        ]),
        h("div", { class: "meta" }, [
          ratingBadge(item.rating),
          item.score !== undefined && item.score !== null
            ? h("span", { class: "score" }, `★${item.score}`)
            : null,
        ]),
      ]);
      grid.appendChild(el);
    });

    updateDownloadCard();
  }

  function flattenTagsForDisplay(tags) {
    if (!tags) return "";
    const order = ["artist", "copyright", "character", "general", "meta"];
    const flat = [];
    for (const cat of order) (tags[cat] || []).forEach((t) => flat.push(t.replace(/_/g, " ")));
    return flat.join(", ");
  }

  function toggleSelect(idx) {
    if (selected.has(idx)) selected.delete(idx);
    else selected.add(idx);
    renderResults();
  }

  function setAllSelected(state) {
    selected.clear();
    if (state) results.forEach((_, idx) => selected.add(idx));
    renderResults();
  }

  function updateDownloadCard() {
    const card = document.getElementById("collect-download-card");
    card.style.display = results.length > 0 ? "block" : "none";
    document.getElementById("collect-selected-count").textContent = selected.size ? `${selected.size} selected` : "";
    document.getElementById("collect-download-btn").disabled = selected.size === 0;
  }

  async function doDownload(items) {
    const destDir = document.getElementById("collect-dest-dir").value.trim();
    if (!destDir) return toast("Choose a destination folder first.", "error");
    items = items || [...selected].map((idx) => results[idx]);
    if (!items.length) return toast("Nothing to download - select at least one image, or use Download all.", "error");
    const btn = document.getElementById("collect-download-btn");
    const allBtn = document.getElementById("collect-download-all-btn");
    const spinner = document.getElementById("collect-download-spinner");
    const statusEl = document.getElementById("collect-download-status");
    btn.disabled = true;
    allBtn.disabled = true;
    spinner.style.display = "inline-block";
    statusEl.textContent = `Downloading ${items.length} images…`;
    try {
      const data = await Api.collectDownload({
        items,
        dest_dir: destDir,
        write_caption: document.getElementById("collect-write-caption").checked,
        trigger_word: document.getElementById("collect-trigger").value.trim(),
      });
      const s = data.summary;
      statusEl.textContent = `Done: ${s.ok} saved, ${s.skipped} skipped, ${s.errored} failed.`;
      toast(`Saved ${s.ok} images to ${data.dest_dir}`, "success");
      ActiveDataset.set(data.dest_dir);
    } catch (err) {
      toast(err.message, "error");
      statusEl.textContent = "Download failed.";
    } finally {
      btn.disabled = selected.size === 0;
      allBtn.disabled = false;
      spinner.style.display = "none";
    }
  }

  function doDownloadAll() {
    return doDownload(results.slice());
  }

  return { init };
})();
