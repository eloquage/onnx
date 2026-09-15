# eloquage/onnx

Thin ONNX Runtime helpers for PHP inference, with a deterministic test
session and an optional runtime adapter.

## ONNX, ONNX Runtime, and this package

- **ONNX** is a portable model format and interchange standard.
- **ONNX Runtime** is a native engine that loads ONNX models and executes
  them.
- **`eloquage/onnx`** provides a small PHP contract for named tensor inputs
  and outputs. Its pure-PHP `FakeSession` is always available; `OrtSession`
  is an optional boundary around the `phpmlkit/onnxruntime` FFI binding.

The package does not require ONNX Runtime, PHP FFI, a shared library, or a
Python process when it is installed.

## Installation

```bash
composer require eloquage/onnx
```

The package requires PHP `^8.3` and has no required native or framework
dependency.

## Portable fake session

Use `FakeSession` for application tests, examples, and local development. It
uses the same named-input and named-output contract as the production adapter
without loading a model file or native runtime:

```php
use Eloquage\Onnx\FakeSession;
use Eloquage\Onnx\Tensor;
use Eloquage\Onnx\TensorSpec;

$session = new FakeSession(
    'demo-model',
    [new TensorSpec('input', 'float32', [1, 2])],
    [new TensorSpec('output', 'float32', [1])],
    [
        'output' => static fn (array $inputs): Tensor => new Tensor(
            'float32',
            [1],
            [array_sum($inputs['input']->values())],
        ),
    ],
);

$outputs = $session->run([
    'input' => new Tensor('float32', [1, 2], [1.0, 2.0]),
]);

echo $outputs['output']->values()[0]; // 3
```

`Tensor` values are flat, row-major PHP arrays. Shapes may contain fixed
non-negative dimensions; `TensorSpec` also accepts `null`, a negative integer,
or a symbolic string as a dynamic dimension.

## Optional production runtime

The production factory is deliberately opt-in:

```php
use Eloquage\Onnx\Onnx;
use Eloquage\Onnx\Tensor;

$session = Onnx::session('/absolute/path/to/model.onnx');
$outputs = $session->run([
    'input' => new Tensor('float32', [1, 2], [1.0, 2.0]),
]);
```

To use it in an application, install and configure the optional binding and
its runtime separately:

```bash
composer require phpmlkit/onnxruntime
```

Then enable PHP's FFI extension and provision the ONNX Runtime shared library
required by that binding for the target platform. Follow the binding's
installation and native-library loader instructions for those two steps.
`Onnx::session()` uses the binding's default runtime behavior; provider menus
are outside this package's v1 API. A missing binding, disabled FFI extension,
unreadable model, or unavailable native library raises a package exception—no
fake fallback is selected implicitly.

## Public API

- `Onnx::name()` returns the stable package identity, `onnx`.
- `Onnx::session(string $modelPath): Session` creates an optional production
  session.
- `Session` exposes `modelPath()`, `inputs()`, `outputs()`, `inputNames()`,
  `outputNames()`, and `run(array $inputs): array`.
- `Tensor` carries a canonical dtype, shape, and flat values. Supported dtypes
  are `bool`, `float32`, `float64`, signed and unsigned 8/16/32/64-bit
  integers, and `string`.
- `TensorSpec` describes a named input or output and validates rank, dtype,
  fixed dimensions, and dynamic dimensions.
- `ModelPathException`, `InputValidationException`,
  `UnsupportedDtypeException`, `TensorShapeException`, `InferenceException`,
  and `RuntimeUnavailableException` are the package-level error types.

## Testing

Run the pure-PHP suite from this package directory:

```bash
composer test
composer test-coverage
composer format
```

The coverage command enforces the package's `src/` floor of 90%. It does not
require FFI, ONNX Runtime, a system shared library, or a TypePHP extension.

TypePHP remains an optional extension build path for maintainers. The current
change does not contain a measured native ONNX Runtime wrapper; see
[TYPEPHP.md](TYPEPHP.md) for the Docker build contract and its explicit skip.

## Security

Model paths are local filesystem paths. Keep model files and any surrounding
application credentials under the application's normal access controls; do
not place secrets in model paths or runtime diagnostics.

## Development notes

The Laravel application at the monorepo root is a local harness, not a
runtime dependency of this package. See [AGENTS.md](AGENTS.md) for package
agent instructions.
