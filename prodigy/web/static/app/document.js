// The program as a document: New, Open… (⌘O), Save (⌘S) and Save As… (⇧⌘S).
//
// Where the browser has the File System Access API (Chrome, Edge), Open and Save work on a file
// on disk, and the file stays linked across reloads (its handle is kept in IndexedDB). Elsewhere
// (Safari, Firefox) Open reads a chosen file and Save downloads the program. The text itself is
// always kept in the browser as well, so a reload never loses work.
//
// A program is pGCL or .ppy; the file extension decides when opening, and Save writes the
// extension of the program's syntax (a .pgcl file converted to .ppy is saved as a new file).
import { h } from "./components/dom.js";
import { load, save } from "./store.js";

const EXTENSIONS = { pgcl: ".pgcl", ppy: ".ppy" };
const OPEN_TYPES = [{ description: "Prodigy program (.ppy, .pgcl)", accept: { "text/plain": [".ppy", ".pgcl"] } }];
const SAVE_TYPES = {
  pgcl: [{ description: "pGCL program", accept: { "text/plain": [".pgcl"] } }],
  ppy: [{ description: ".ppy program", accept: { "text/plain": [".ppy"] } }],
};

export const canWriteFiles = typeof window.showOpenFilePicker === "function" &&
  typeof window.showSaveFilePicker === "function";

/** The syntax of a file: .ppy, or pGCL for everything else (.pgcl, .txt). */
export function syntaxOf(fileName) {
  return /\.ppy$/i.test(String(fileName || "")) ? "ppy" : "pgcl";
}

function stem(fileName) {
  const base = String(fileName || "").split(/[\\/]/).pop();
  return base.replace(/\.(?:pgcl|ppy|txt)$/i, "");
}

// ---- file handles in IndexedDB (localStorage cannot hold them) -------------------------------

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("prodigy-gui", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("handles");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore(mode, action) {
  const db = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction("handles", mode);
      const request = action(transaction.objectStore("handles"));
      transaction.oncomplete = () => resolve(request?.result ?? null);
      transaction.onerror = () => reject(transaction.error);
    });
  } finally {
    db.close();
  }
}

async function storeHandle(handle) {
  try {
    await withStore("readwrite", (store) => (handle ? store.put(handle, "current") : store.delete("current")));
  } catch {
    // a convenience only: without it, Save asks for the location again after a reload
  }
}

async function loadHandle() {
  try {
    return await withStore("readonly", (store) => store.get("current"));
  } catch {
    return null;
  }
}

async function writeFile(handle, text) {
  const options = { mode: "readwrite" };
  if (handle.queryPermission && (await handle.queryPermission(options)) !== "granted" &&
      (await handle.requestPermission(options)) !== "granted") {
    throw new Error(`Prodigy may not write to ${handle.name}.`);
  }
  const writable = await handle.createWritable();
  await writable.write(text);
  await writable.close();
}

function download(fileName, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const link = h("a", { href: url, download: fileName, hidden: true });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---- dialogs -----------------------------------------------------------------------------------

/**
 * Shows an alert and resolves to the value of the button that closed it ("" for Esc). The default
 * button is the dialog's only submit button, so Return chooses it even from a text field; the
 * other buttons close the dialog with their value.
 */
function ask(dialog) {
  return new Promise((resolve) => {
    const buttons = [...dialog.querySelectorAll("button[type=button][value]")];
    const choose = (event) => dialog.close(event.currentTarget.value);
    buttons.forEach((button) => button.addEventListener("click", choose));
    dialog.addEventListener("close", () => {
      buttons.forEach((button) => button.removeEventListener("click", choose));
      resolve(dialog.returnValue);
    }, { once: true });
    dialog.returnValue = "";
    dialog.showModal();
  });
}

// ---- the document ----------------------------------------------------------------------------------

/**
 * meta = { name, origin, category, saved, file }
 *   name     file name without extension, or the example's title (null: Untitled)
 *   origin   "untitled" | "file" | "example"
 *   category the example's category ("" otherwise)
 *   saved    the text as last opened or saved (null: never saved)
 *   file     the full name of the linked file (origin "file")
 * The syntax is the editor's (editor.getSyntax()).
 */
export function createDocument({ editor, initialText, onReplace, notify, onChange }) {
  let handle = null;
  let meta = load("document", null);
  if (!meta || typeof meta !== "object") {
    // first start, or a program kept by an older version (title {title, subtitle})
    const old = load("title", null);
    const edited = Boolean(old?.subtitle?.endsWith("edited"));
    meta = old?.title
      ? { name: old.title, origin: "example", category: (old.subtitle || "").replace(/(?: · )?edited$/, ""),
          saved: edited ? null : initialText }
      : { name: null, origin: "untitled", category: "", saved: initialText };
    persist();
  }

  function persist() {
    save("document", meta);
  }

  function displayName() {
    return meta.name || "Untitled";
  }

  function isEdited(text = editor.getSource()) {
    if (meta.saved === null) return text.trim() !== "";
    return text !== meta.saved;
  }

  const extension = () => EXTENSIONS[editor.getSyntax()];

  function suggestedFileName() {
    return `${meta.fileStem || meta.name || "Untitled"}${extension()}`;
  }

  /** The linked file, if the program is still in its syntax (a converted program is not). */
  function linkedFile() {
    if (meta.origin !== "file") return null;
    const file = meta.file || `${meta.name}.pgcl`;
    return syntaxOf(file) === editor.getSyntax() ? file : null;
  }

  function changed() {
    onChange?.({ name: displayName(), category: meta.category, edited: isEdited(), origin: meta.origin,
      fileName: meta.origin === "file" ? (meta.file || `${meta.name}.pgcl`) : null, syntax: editor.getSyntax() });
  }

  function become(next, newHandle = null) {
    meta = next;
    handle = newHandle;
    persist();
    storeHandle(newHandle);
    changed();
  }

  /** Asks whether to save unsaved changes. Resolves to false if the person cancels. */
  async function confirmClose() {
    if (!isEdited()) return true;
    const dialog = document.getElementById("close-alert");
    dialog.querySelector("#close-alert-title").textContent = `Do you want to save the changes to “${displayName()}”?`;
    const choice = await ask(dialog);
    if (choice === "discard") return true;
    if (choice !== "save") return false;
    return saveDocument();
  }

  async function askName(suggested) {
    const dialog = document.getElementById("name-alert");
    const field = dialog.querySelector("#name-alert-field");
    dialog.querySelector("#name-alert-extension").textContent = extension();
    field.value = suggested;
    const answer = ask(dialog);
    field.select();
    const choice = await answer;
    const name = stem(field.value.trim());
    return choice === "save" && name ? name : null;
  }

  /** Asks for the syntax of a new program; resolves to "ppy", "pgcl" or null (cancelled). */
  async function askSyntax() {
    const choice = await ask(document.getElementById("new-alert"));
    return choice === "ppy" || choice === "pgcl" ? choice : null;
  }

  async function newDocument() {
    if (!(await confirmClose())) return;
    const syntax = await askSyntax();
    if (!syntax) return;
    editor.setSource("", syntax);
    become({ name: null, origin: "untitled", category: "", saved: "" });
    onReplace({ kind: "new" });
    editor.focus();
  }

  async function pickFile() {
    if (canWriteFiles) {
      try {
        const [picked] = await window.showOpenFilePicker({ types: OPEN_TYPES, multiple: false });
        const file = await picked.getFile();
        return { handle: picked, name: file.name, text: await file.text() };
      } catch (error) {
        if (error.name === "AbortError") return null;
        throw error;
      }
    }
    return new Promise((resolve) => {
      const input = h("input", { type: "file", accept: ".ppy,.pgcl,.txt,text/plain", hidden: true });
      input.addEventListener("change", async () => {
        const file = input.files?.[0];
        input.remove();
        resolve(file ? { handle: null, name: file.name, text: await file.text() } : null);
      });
      input.addEventListener("cancel", () => { input.remove(); resolve(null); });
      document.body.append(input);
      input.click();
    });
  }

  async function openDocument() {
    if (!(await confirmClose())) return;
    let picked;
    try {
      picked = await pickFile();
    } catch (error) {
      notify(`Could not open the file: ${error.message}`, "error");
      return;
    }
    if (!picked) return;
    // a fresh editor state: undo must not bring another program into this file
    editor.load(picked.text, syntaxOf(picked.name));
    become({ name: stem(picked.name), origin: "file", category: "", saved: picked.text, file: picked.name },
      picked.handle);
    onReplace({ kind: "open" });
    notify(`Opened ${picked.name}.`, "info");
    editor.focus();
  }

  async function saveDocument({ as = false } = {}) {
    const text = editor.getSource();
    if (canWriteFiles) {
      let target = !as && linkedFile() ? handle : null;
      if (!target) {
        try {
          target = await window.showSaveFilePicker({ suggestedName: suggestedFileName(),
            types: SAVE_TYPES[editor.getSyntax()] });
        } catch (error) {
          if (error.name !== "AbortError") notify(`Could not save: ${error.message}`, "error");
          return false;
        }
      }
      try {
        await writeFile(target, text);
      } catch (error) {
        notify(`Could not save ${target.name}: ${error.message}`, "error");
        return false;
      }
      if (syntaxOf(target.name) !== editor.getSyntax()) {
        notify(`${target.name} does not end in ${extension()}, so it would be opened as the wrong syntax. ` +
          `Choose a name ending in ${extension()}.`, "error");
        return false;
      }
      become({ name: stem(target.name), origin: "file", category: "", saved: text, file: target.name }, target);
      notify(`Saved ${target.name}.`, "info");
      return true;
    }
    // no file system access: download, asking for a name unless the program came from a file
    const name = !as && linkedFile() && meta.name ? meta.name : await askName(suggestedFileName());
    if (!name) return false;
    const fileName = `${name}${extension()}`;
    download(fileName, text);
    become({ name, origin: "file", category: "", saved: text, file: fileName });
    notify(`Downloaded ${fileName}.`, "info");
    return true;
  }

  /** An example replaced the program (as an undoable change). */
  function exampleLoaded(example, text) {
    become({ name: example.title, origin: "example", category: example.category_title || "", saved: text,
      fileStem: stem(example.id) });
  }

  async function restoreHandle() {
    if (meta.origin !== "file" || !canWriteFiles) return;
    const stored = await loadHandle();
    if (stored && stored.name === (meta.file || `${meta.name}.pgcl`)) handle = stored;
  }

  restoreHandle();

  return {
    confirmClose,
    new: newDocument,
    open: openDocument,
    save: saveDocument,
    exampleLoaded,
    changed,
    isEdited,
    displayName,
  };
}
