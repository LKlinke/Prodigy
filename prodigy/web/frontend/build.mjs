// Builds static/vendor/codemirror.js and copies KaTeX (woff2 fonts only) to static/vendor/katex.
import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const vendor = join(here, "..", "static", "vendor");
mkdirSync(vendor, { recursive: true });

await build({
  entryPoints: [join(here, "codemirror-entry.js")],
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2020",
  outfile: join(vendor, "codemirror.js"),
  legalComments: "eof",
});

const katexSrc = join(here, "node_modules", "katex", "dist");
const katexOut = join(vendor, "katex");
rmSync(katexOut, { recursive: true, force: true });
mkdirSync(join(katexOut, "fonts"), { recursive: true });
cpSync(join(katexSrc, "katex.min.js"), join(katexOut, "katex.min.js"));
for (const font of readdirSync(join(katexSrc, "fonts"))) {
  if (font.endsWith(".woff2")) cpSync(join(katexSrc, "fonts", font), join(katexOut, "fonts", font));
}
// keep only the woff2 sources so that browsers do not request missing files
const css = readFileSync(join(katexSrc, "katex.min.css"), "utf8")
  .replace(/,url\(fonts\/[^)]+\.woff\) format\("woff"\)/g, "")
  .replace(/,url\(fonts\/[^)]+\.ttf\) format\("truetype"\)/g, "");
writeFileSync(join(katexOut, "katex.min.css"), css);
cpSync(join(here, "node_modules", "katex", "LICENSE"), join(katexOut, "LICENSE"));
console.log("vendor bundle written to", vendor);
