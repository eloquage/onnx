<?php

namespace Eloquage\Onnx\Concerns;

use Eloquage\Onnx\Exceptions\InputValidationException;
use Eloquage\Onnx\Tensor;
use Eloquage\Onnx\TensorSpec;

trait ValidatesSessionInputs
{
    /** @param array<string, mixed> $inputs @param array<string, TensorSpec> $specifications */
    private function validateSessionInputs(array $inputs, array $specifications): void
    {
        foreach ($inputs as $name => $tensor) {
            if (! is_string($name) || ! array_key_exists($name, $specifications)) {
                throw new InputValidationException(sprintf(
                    'Unknown session input "%s". Expected one of: %s.',
                    (string) $name,
                    implode(', ', array_keys($specifications)),
                ));
            }

            if (! $tensor instanceof Tensor) {
                throw new InputValidationException(sprintf(
                    'Session input "%s" must be a Tensor instance.',
                    $name,
                ));
            }

            $specifications[$name]->validate($tensor);
        }

        foreach ($specifications as $name => $specification) {
            if (! array_key_exists($name, $inputs)) {
                throw new InputValidationException(sprintf(
                    'Missing required session input "%s" with dtype "%s" and shape %s.',
                    $name,
                    $specification->dtype(),
                    '['.implode(', ', array_map(static fn (?int $dimension): string => $dimension === null ? '?' : (string) $dimension, $specification->shape())).']',
                ));
            }
        }
    }
}
