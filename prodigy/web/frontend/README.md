# Frontend libraries of the web GUI

The GUI itself (`../static/app/`) is plain JavaScript modules and needs no build step.
Only the third-party libraries are bundled once and checked in under `../static/vendor/`:

* `codemirror.js` – CodeMirror 6 with the pGCL highlighting mode (`codemirror-entry.js`)
* `katex/` – KaTeX for the closed forms (woff2 fonts only)

To update them (Node 18 or newer):

```sh
cd prodigy/web/frontend
npm install
npm run build
```
