<?php

namespace Eloquage\Onnx;

use Eloquage\Onnx\Exceptions\InputValidationException;
use Eloquage\Onnx\Exceptions\TensorShapeException;

final class TensorSpec
{
    /** @var list<int|null> */
    private array $shape;

    /**
     * @param  list<int|string|null>  $shape
     */
    public function __construct(
        private string $name,
        private string $dtype,
        array $shape,
    ) {
        if (trim($name) === '') {
            throw new InputValidationException('Tensor specification names must not be empty.');
        }

        $this->dtype = Tensor::canonicalDtype($dtype);
        $this->shape = self::normalizeShape($shape);
    }

    public function name(): string
    {
        return $this->name;
    }

    public function dtype(): string
    {
        return $this->dtype;
    }

    /** @return list<int|null> */
    public function shape(): array
    {
        return $this->shape;
    }

    public function rank(): int
    {
        return count($this->shape);
    }

    public function isDynamicDimension(int $index): bool
    {
        if (! array_key_exists($index, $this->shape)) {
            throw new TensorShapeException(sprintf('Tensor specification has no dimension %d.', $index));
        }

        return $this->shape[$index] === null;
    }

    public function validate(Tensor $tensor): void
    {
        if ($tensor->dtype() !== $this->dtype) {
            throw new InputValidationException(sprintf(
                'Tensor "%s" has dtype "%s"; expected "%s".',
                $this->name,
                $tensor->dtype(),
                $this->dtype,
            ));
        }

        if ($tensor->shape() === [] && $this->shape !== []) {
            throw new TensorShapeException(sprintf(
                'Tensor "%s" has rank 0; expected rank %d with shape %s.',
                $this->name,
                $this->rank(),
                self::formatShape($this->shape),
            ));
        }

        if (count($tensor->shape()) !== $this->rank()) {
            throw new TensorShapeException(sprintf(
                'Tensor "%s" has rank %d and shape %s; expected rank %d with shape %s.',
                $this->name,
                count($tensor->shape()),
                self::formatShape($tensor->shape()),
                $this->rank(),
                self::formatShape($this->shape),
            ));
        }

        foreach ($this->shape as $index => $dimension) {
            if ($dimension !== null && $tensor->shape()[$index] !== $dimension) {
                throw new TensorShapeException(sprintf(
                    'Tensor "%s" dimension %d is %d; expected %d in shape %s.',
                    $this->name,
                    $index,
                    $tensor->shape()[$index],
                    $dimension,
                    self::formatShape($this->shape),
                ));
            }
        }
    }

    /** @param array<int|string, TensorSpec> $specifications @return array<string, TensorSpec> */
    public static function index(array $specifications): array
    {
        $indexed = [];

        foreach ($specifications as $specification) {
            if (! $specification instanceof self) {
                throw new InputValidationException('Tensor specifications must contain only TensorSpec instances.');
            }

            if (array_key_exists($specification->name, $indexed)) {
                throw new InputValidationException(sprintf(
                    'Tensor specification name "%s" is declared more than once.',
                    $specification->name,
                ));
            }

            $indexed[$specification->name] = $specification;
        }

        return $indexed;
    }

    /** @param list<int|string|null> $shape @return list<int|null> */
    private static function normalizeShape(array $shape): array
    {
        if (! array_is_list($shape)) {
            throw new TensorShapeException('Tensor specification shape must be a zero-based list.');
        }

        return array_map(static function (mixed $dimension): ?int {
            if ($dimension === null || (is_int($dimension) && $dimension < 0)) {
                return null;
            }

            if (is_string($dimension) && trim($dimension) !== '') {
                return null;
            }

            if (is_int($dimension) && $dimension >= 0) {
                return $dimension;
            }

            throw new TensorShapeException('Tensor specification dimensions must be non-negative integers or dynamic markers.');
        }, $shape);
    }

    /** @param list<int|null> $shape */
    private static function formatShape(array $shape): string
    {
        return '['.implode(', ', array_map(static fn (?int $dimension): string => $dimension === null ? '?' : (string) $dimension, $shape)).']';
    }
}
