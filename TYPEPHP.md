# TypePHP readiness

`eloquage/onnx` is framework-agnostic PHP first.

- **Source of truth:** the public PHP API under `src/`
- **Optional accel:** TypePHP AOT sources can live in `native/`
- **Build sketch:** copy `project.yml.example` to `project.yml` and run `tpc` when ready

Do not require `swoole/typephp` from Composer for consumers. Native builds are a maintainer/CI concern; ship a pure-PHP fallback always.
