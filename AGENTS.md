# Agent context: `eloquage/onnx`

This directory is the standalone `eloquage/onnx` package. The Laravel project
above it is only a local test bench; do not add Laravel, Illuminate, or a
service provider to this package.

## Source boundaries

- `src/` is the framework-agnostic PHP API and the source of truth.
- `native/` is reserved for optional TypePHP sources. It is empty unless a
  separately approved, measured native implementation is added.
- `tests/` is the pure-PHP Pest suite. Tests must run without FFI, ONNX
  Runtime, a system shared library, or a TypePHP extension.
- `TYPEPHP.md` and `project.yml.example` describe the optional extension
  build contract; they do not make native acceleration an install
  prerequisite.

Keep `Onnx::name(): string` returning `onnx`, keep the public API additive,
and preserve the pure-PHP path. The optional `OrtSession` boundary may detect
the `phpmlkit/onnxruntime` binding dynamically, but Composer must not require
that binding, `swoole/typephp`, or any Illuminate package.

## Required checks

From this directory:

```bash
composer test
composer test-coverage
composer format
```

`composer test-coverage` is the required `src/` gate and enforces 90% minimum
coverage. Keep runtime-boundary tests injected and deterministic; do not make
the suite depend on a downloaded model or external native library.

## TypePHP boundary

TypePHP is extension mode only (`mode: ext`) and Docker-only. The documented
builder image is `ghcr.io/eloquage/typephp-builder:latest`; the project
configuration writes normal generated artifacts under the ignored `build/`
directory.

For `onnx-runtime-session-helpers`, native compilation is explicitly skipped:
there is no measured typed FFI/ONNX Runtime wrapper under `native/`, and a
successful generic `.so` build would not prove that the external runtime loads
or performs inference. Do not run or report
`docker/typephp/build-package.sh onnx` as this change's ORT acceptance gate.
Treat existing generated `build/` output as user work and do not modify it.

## Harness boundary

The root welcome page may use `FakeSession` for an observable demo. It must
not construct `Onnx::session()` or require a production model/runtime. Update
the harness Feature test when changing that demo.

Use the README for end-user installation and API examples. Keep this file
focused on agent constraints, boundaries, and verification commands.
