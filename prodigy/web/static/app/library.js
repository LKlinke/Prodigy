// The example library: a sheet listing pgfexamples, grouped by category, with search and
// keyboard navigation (↑ ↓ to move, Return to open, Esc to close).
import { h, icon } from "./components/dom.js";

const COMMAND_LABELS = {
  check_equality: "invariant",
  invariant_synthesis: "synthesis",
  independent_vars: "independence",
  main: "",
  skip: "",
};

export function createLibrary(dialog, { api, onLoad }) {
  const list = dialog.querySelector("#library-list");
  const search = dialog.querySelector("#library-search");
  let examples = null;
  let rows = [];
  let active = -1;

  function setActive(index) {
    active = Math.max(0, Math.min(rows.length - 1, index));
    rows.forEach((row, i) => row.setAttribute("aria-selected", String(i === active)));
    rows[active]?.scrollIntoView({ block: "nearest" });
    if (rows[active]) list.setAttribute("aria-activedescendant", rows[active].id);
  }

  function render() {
    const query = search.value.trim().toLowerCase();
    const matches = (examples || []).filter((e) => !query ||
      `${e.title} ${e.category_title} ${e.id}`.toLowerCase().includes(query));
    list.replaceChildren();
    rows = [];
    if (!examples) {
      list.append(h("p", { class: "library-empty" }, "Loading examples…"));
      return;
    }
    if (!matches.length) {
      list.append(h("p", { class: "library-empty" }, examples.length ? "No example matches your search." : "No examples found on the server."));
      return;
    }
    let group = null;
    for (const example of matches) {
      if (example.category_title !== group) {
        group = example.category_title;
        list.append(h("div", { class: "library-group", role: "presentation" }, group || "Other"));
      }
      const label = COMMAND_LABELS[example.command] ?? example.command;
      const row = h("button", {
        type: "button", class: "library-row", role: "option", id: `example-${rows.length}`, "aria-selected": "false",
        title: example.id,
      },
      h("span", { class: "row-title" }, example.title),
      h("span", { style: "display:inline-flex;gap:6px" },
        example.known_issue ? h("span", { class: "chip", title: "Skipped in the test suite (known issue)" }, icon("warning"), "known issue") : null,
        label ? h("span", { class: "chip" }, label) : null));
      const index = rows.length;
      row.addEventListener("click", () => choose(example));
      row.addEventListener("mousemove", () => { if (active !== index) setActive(index); });
      rows.push(row);
      list.append(row);
    }
    setActive(0);
  }

  async function choose(example) {
    dialog.close("load");
    await onLoad(example);
  }

  search.addEventListener("input", render);
  search.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") { event.preventDefault(); setActive(active + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); setActive(active - 1); }
    else if (event.key === "Enter") {
      event.preventDefault();
      const query = search.value.trim().toLowerCase();
      const matches = (examples || []).filter((e) => !query || `${e.title} ${e.category_title} ${e.id}`.toLowerCase().includes(query));
      if (matches[active]) choose(matches[active]);
    }
  });

  return {
    async open() {
      search.value = "";
      render();
      dialog.showModal();
      search.focus();
      if (!examples) {
        try {
          examples = (await api.examples()).examples;
        } catch (error) {
          examples = [];
          list.replaceChildren(h("p", { class: "library-empty" }, error.message));
          return;
        }
        render();
      }
    },
  };
}
