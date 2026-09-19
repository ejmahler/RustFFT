// Minimal WASI harness: runs planner_tuning.wasm under node with argv/stdio passthrough.
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
const wasi = new WASI({
  version: 'preview1',
  args: ['planner_tuning', ...process.argv.slice(2)],
  env: {},
  preopens: { '.': '.' },
  returnOnExit: true,
});
const bytes = await readFile(new URL('./target/wasm32-wasip1/release/planner_tuning.wasm', import.meta.url));
const wasm = await WebAssembly.compile(bytes);
const instance = await WebAssembly.instantiate(wasm, wasi.getImportObject());
process.exitCode = wasi.start(instance);
