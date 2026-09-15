<?php

namespace Eloquage\Onnx;

use Eloquage\Onnx\Concerns\ValidatesSessionInputs;
use Eloquage\Onnx\Exceptions\InferenceException;
use Eloquage\Onnx\Exceptions\InputValidationException;
use Eloquage\Onnx\Exceptions\ModelPathException;

final class FakeSession implements Session
{
    use ValidatesSessionInputs;

    /** @var array<string, TensorSpec> */
    private array $inputs;

    /** @var array<string, TensorSpec> */
    private array $outputs;

    /** @var array<string, Tensor|callable> */
    private array $outputValues;

    /**
     * @param  array<int|string, TensorSpec>  $inputs
     * @param  array<int|string, TensorSpec>  $outputs
     * @param  array<string, Tensor|callable>  $outputValues
     */
    public function __construct(
        private string $modelPath,
        array $inputs,
        array $outputs,
        array $outputValues = [],
    ) {
        if (trim($modelPath) === '') {
            throw new ModelPathException('Fake session model identifiers must not be empty.');
        }

        $this->inputs = TensorSpec::index($inputs);
        $this->outputs = TensorSpec::index($outputs);

        foreach ($outputValues as $name => $value) {
            if (! is_string($name) || ! array_key_exists($name, $this->outputs)) {
                throw new InputValidationException(sprintf('Unknown fake-session output "%s".', (string) $name));
            }

            if (! $value instanceof Tensor && ! is_callable($value)) {
                throw new InputValidationException(sprintf(
                    'Fake-session output "%s" must be a Tensor or callable fixture.',
                    $name,
                ));
            }
        }

        $this->outputValues = $outputValues;
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
        $results = [];

        foreach ($this->outputs as $name => $specification) {
            if (! array_key_exists($name, $this->outputValues)) {
                throw new InferenceException(sprintf('Fake session has no fixture for output "%s".', $name));
            }

            $value = $this->outputValues[$name];
            $tensor = is_callable($value) ? $value($inputs) : $value;

            if (! $tensor instanceof Tensor) {
                throw new InferenceException(sprintf('Fake fixture for output "%s" did not return a Tensor.', $name));
            }

            try {
                $specification->validate($tensor);
            } catch (InputValidationException $exception) {
                throw new InferenceException(sprintf(
                    'Fake output "%s" does not satisfy its declared specification: %s',
                    $name,
                    $exception->getMessage(),
                ), 0, $exception);
            }

            $results[$name] = $tensor;
        }

        return $results;
    }
}
