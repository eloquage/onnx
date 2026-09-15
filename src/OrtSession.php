<?php

namespace Eloquage\Onnx;

use Closure;
use Eloquage\Onnx\Concerns\ValidatesSessionInputs;
use Eloquage\Onnx\Exceptions\InferenceException;
use Eloquage\Onnx\Exceptions\ModelPathException;
use Eloquage\Onnx\Exceptions\OnnxException;
use Eloquage\Onnx\Exceptions\RuntimeUnavailableException;
use ReflectionMethod;
use Throwable;
use Traversable;

/**
 * Optional adapter for the PhpMlKit ONNX Runtime FFI binding.
 *
 * The optional constructor arguments are an internal seam for testing the
 * boundary without installing FFI or ONNX Runtime. Normal consumers pass only
 * a local model path and the adapter discovers the binding itself.
 */
final class OrtSession implements Session
{
    use ValidatesSessionInputs;

    private const BINDING_SESSION = 'PhpMlKit\\ONNXRuntime\\InferenceSession';

    private const BINDING_VALUE = 'PhpMlKit\\ONNXRuntime\\OrtValue';

    private const BINDING_DTYPE = 'PhpMlKit\\ONNXRuntime\\Enums\\DataType';

    /** @var array<string, TensorSpec> */
    private array $inputs = [];

    /** @var array<string, TensorSpec> */
    private array $outputs = [];

    private ?object $runtimeSession = null;

    private ?Closure $valueFactory = null;

    private bool $disposed = false;

    public function __construct(
        private string $modelPath,
        ?object $runtimeSession = null,
        ?Closure $valueFactory = null,
    ) {
        self::assertModelPath($modelPath);

        $this->runtimeSession = $runtimeSession ?? $this->openRuntime($modelPath);
        $this->valueFactory = $valueFactory ?? function (Tensor $tensor): object {
            return $this->createOrtValue($tensor);
        };

        try {
            $this->inputs = self::normalizeSpecifications($this->runtimeCall('inputs'), 'input');
            $this->outputs = self::normalizeSpecifications($this->runtimeCall('outputs'), 'output');
        } catch (OnnxException $exception) {
            $this->dispose();

            throw $exception;
        } catch (Throwable $exception) {
            $this->dispose();

            throw new RuntimeUnavailableException(
                'ONNX Runtime returned unusable model metadata.',
                0,
                $exception,
            );
        }
    }

    public function modelPath(): string
    {
        return $this->modelPath;
    }

    /** @return array<string, TensorSpec> */
    public function inputs(): array
    {
        return $this->inputs;
    }

    /** @return array<string, TensorSpec> */
    public function outputs(): array
    {
        return $this->outputs;
    }

    /** @return list<string> */
    public function inputNames(): array
    {
        return array_keys($this->inputs);
    }

    /** @return list<string> */
    public function outputNames(): array
    {
        return array_keys($this->outputs);
    }

    /** @param array<string, Tensor> $inputs @return array<string, Tensor> */
    public function run(array $inputs): array
    {
        $this->validateSessionInputs($inputs, $this->inputs);
        $runtimeInputs = [];

        try {
            foreach ($inputs as $name => $tensor) {
                $runtimeInputs[$name] = ($this->valueFactory)($tensor);
            }

            $runtimeOutputs = $this->runtimeCall('run', [$runtimeInputs]);
            $results = $this->normalizeOutputs($runtimeOutputs);
        } catch (OnnxException $exception) {
            throw $exception;
        } catch (Throwable $exception) {
            throw new InferenceException(
                'ONNX Runtime inference failed: '.$exception->getMessage(),
                0,
                $exception,
            );
        } finally {
            if (isset($runtimeOutputs) && is_array($runtimeOutputs)) {
                self::disposeValues($runtimeOutputs);
            }
        }

        return $results;
    }

    public function dispose(): void
    {
        if ($this->disposed || $this->runtimeSession === null) {
            return;
        }

        $this->disposed = true;

        if (method_exists($this->runtimeSession, 'dispose')) {
            try {
                $this->runtimeSession->dispose();
            } catch (Throwable) {
                // Destructors and explicit cleanup must not mask inference errors.
            }
        }

        $this->runtimeSession = null;
    }

    public function __destruct()
    {
        $this->dispose();
    }

    private function openRuntime(string $modelPath): object
    {
        if (! extension_loaded('FFI') || ! class_exists(self::BINDING_SESSION)) {
            throw new RuntimeUnavailableException(
                'ONNX Runtime is unavailable: enable PHP FFI, install phpmlkit/onnxruntime, and provision its native runtime library.',
            );
        }

        try {
            return self::BINDING_SESSION::fromFile($modelPath);
        } catch (Throwable $exception) {
            throw new RuntimeUnavailableException(
                'ONNX Runtime could not load the model or native runtime library.',
                0,
                $exception,
            );
        }
    }

    private function createOrtValue(Tensor $tensor): object
    {
        $dtypeName = [
            'bool' => 'BOOL',
            'float32' => 'FLOAT',
            'float64' => 'DOUBLE',
            'int8' => 'INT8',
            'uint8' => 'UINT8',
            'int16' => 'INT16',
            'uint16' => 'UINT16',
            'int32' => 'INT32',
            'uint32' => 'UINT32',
            'int64' => 'INT64',
            'uint64' => 'UINT64',
            'string' => 'STRING',
        ][$tensor->dtype()];

        $dataType = constant(self::BINDING_DTYPE.'::'.$dtypeName);
        $method = new ReflectionMethod(self::BINDING_VALUE, 'fromArray');
        $arguments = [$tensor->values(), $dataType];

        if ($method->getNumberOfParameters() >= 3) {
            $arguments[] = $tensor->shape();
        }

        /** @var object */
        return $method->invokeArgs(null, $arguments);
    }

    private function runtimeCall(string $method, array $arguments = []): mixed
    {
        if ($this->runtimeSession === null || ! method_exists($this->runtimeSession, $method)) {
            throw new RuntimeUnavailableException(sprintf('The ONNX Runtime session does not expose %s().', $method));
        }

        return $this->runtimeSession->{$method}(...$arguments);
    }

    /** @return array<string, TensorSpec> */
    private static function normalizeSpecifications(mixed $metadata, string $kind): array
    {
        if (! is_array($metadata) && ! $metadata instanceof Traversable) {
            throw new RuntimeUnavailableException(sprintf('ONNX Runtime returned invalid %s metadata.', $kind));
        }

        $specifications = [];

        foreach ($metadata as $key => $entry) {
            $name = is_string($key) && trim($key) !== ''
                ? $key
                : self::metadataString($entry, ['name', 'getName'], $kind.' name');
            $dtype = self::metadataString($entry, ['dtype', 'dataType', 'getDtype', 'getDataType'], $name.' dtype');
            $shape = self::metadataArray($entry, ['shape', 'getShape', 'getSymbolicShape'], $name.' shape');

            $specifications[$name] = new TensorSpec($name, $dtype, $shape);
        }

        if ($specifications === []) {
            throw new RuntimeUnavailableException(sprintf('ONNX Runtime returned no %s metadata.', $kind));
        }

        return $specifications;
    }

    /** @return array<string, Tensor> */
    private function normalizeOutputs(mixed $runtimeOutputs): array
    {
        if (! is_array($runtimeOutputs)) {
            throw new InferenceException('ONNX Runtime returned outputs in an invalid format.');
        }

        if (array_is_list($runtimeOutputs)) {
            if (count($runtimeOutputs) !== count($this->outputNames())) {
                throw new InferenceException('ONNX Runtime returned an incomplete output list.');
            }

            $namedOutputs = array_combine($this->outputNames(), $runtimeOutputs);
        } else {
            $namedOutputs = $runtimeOutputs;
        }

        if ($namedOutputs === false) {
            throw new InferenceException('ONNX Runtime returned an incomplete output list.');
        }

        $results = [];

        foreach ($this->outputs as $name => $specification) {
            if (! array_key_exists($name, $namedOutputs)) {
                throw new InferenceException(sprintf('ONNX Runtime did not return required output "%s".', $name));
            }

            $runtimeValue = $namedOutputs[$name];
            $values = self::runtimeArray($runtimeValue, $name);
            $shape = self::runtimeShape($runtimeValue) ?? self::fallbackShape($specification, count(self::flatten($values)));
            $tensor = new Tensor($specification->dtype(), $shape, self::flatten($values));

            try {
                $specification->validate($tensor);
            } catch (OnnxException $exception) {
                throw new InferenceException(sprintf(
                    'ONNX Runtime output "%s" does not satisfy its declared specification: %s',
                    $name,
                    $exception->getMessage(),
                ), 0, $exception);
            }

            $results[$name] = $tensor;
        }

        return $results;
    }

    /** @return list<mixed> */
    private static function runtimeArray(mixed $value, string $name): array
    {
        if ($value instanceof Tensor) {
            return $value->values();
        }

        if (is_array($value)) {
            return $value;
        }

        if (is_object($value) && method_exists($value, 'toArray')) {
            $values = $value->toArray();

            if (is_array($values)) {
                return $values;
            }
        }

        throw new InferenceException(sprintf('ONNX Runtime output "%s" is not array-like.', $name));
    }

    /** @return list<int>|null */
    private static function runtimeShape(mixed $value): ?array
    {
        if (! is_object($value)) {
            return null;
        }

        foreach (['shape', 'getShape'] as $method) {
            if (method_exists($value, $method)) {
                $shape = $value->{$method}();

                if (is_array($shape) && array_is_list($shape)) {
                    return array_map(static fn (mixed $dimension): int => (int) $dimension, $shape);
                }
            }
        }

        return null;
    }

    /** @return list<int> */
    private static function fallbackShape(TensorSpec $specification, int $valueCount): array
    {
        $shape = $specification->shape();

        if (! in_array(null, $shape, true)) {
            /** @var list<int> */
            return $shape;
        }

        if ($specification->rank() !== 1) {
            throw new InferenceException(sprintf(
                'ONNX Runtime did not report the concrete shape for dynamic output "%s".',
                $specification->name(),
            ));
        }

        return [$valueCount];
    }

    /** @param mixed $values @return list<mixed> */
    private static function flatten(mixed $values): array
    {
        if (! is_array($values)) {
            return [$values];
        }

        $flattened = [];

        foreach ($values as $value) {
            array_push($flattened, ...self::flatten($value));
        }

        return $flattened;
    }

    /** @param array<int|string, mixed> $values */
    private static function disposeValues(array $values): void
    {
        foreach ($values as $value) {
            if (is_object($value) && method_exists($value, 'dispose')) {
                try {
                    $value->dispose();
                } catch (Throwable) {
                    // A cleanup failure must not replace the successful result.
                }
            }
        }
    }

    /** @param list<string> $methods @return string */
    private static function metadataString(mixed $metadata, array $methods, string $label): string
    {
        $value = self::metadataValue($metadata, $methods);

        if (is_object($value)) {
            if (isset($value->name)) {
                $value = $value->name;
            } elseif (method_exists($value, 'name')) {
                $value = $value->name();
            } elseif (isset($value->value)) {
                $value = $value->value;
            } elseif (method_exists($value, 'value')) {
                $value = $value->value();
            }
        }

        if (! is_string($value) || trim($value) === '') {
            throw new RuntimeUnavailableException(sprintf('ONNX Runtime returned invalid %s.', $label));
        }

        return $value;
    }

    /** @param list<string> $methods @return array<int|string, mixed> */
    private static function metadataArray(mixed $metadata, array $methods, string $label): array
    {
        $value = self::metadataValue($metadata, $methods);

        if ($value instanceof Traversable) {
            $value = iterator_to_array($value, false);
        }

        if (! is_array($value)) {
            throw new RuntimeUnavailableException(sprintf('ONNX Runtime returned invalid %s.', $label));
        }

        return $value;
    }

    /** @param list<string> $methods @return mixed */
    private static function metadataValue(mixed $metadata, array $methods): mixed
    {
        foreach ($methods as $method) {
            if (is_array($metadata) && array_key_exists($method, $metadata)) {
                return $metadata[$method];
            }

            if (is_object($metadata) && method_exists($metadata, $method)) {
                return $metadata->{$method}();
            }
        }

        throw new RuntimeUnavailableException('ONNX Runtime metadata is missing a required property.');
    }

    private static function assertModelPath(string $modelPath): void
    {
        if ($modelPath === '' || preg_match('/^[a-zA-Z][a-zA-Z0-9+.-]*:/', $modelPath) === 1) {
            throw new ModelPathException('ONNX models must be loaded from a local file path.');
        }

        if (! is_file($modelPath) || ! is_readable($modelPath)) {
            throw new ModelPathException('ONNX model path is not a readable file.');
        }
    }
}
