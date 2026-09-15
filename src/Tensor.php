<?php

namespace Eloquage\Onnx;

use Eloquage\Onnx\Exceptions\TensorShapeException;
use Eloquage\Onnx\Exceptions\UnsupportedDtypeException;

final class Tensor
{
    /** @var list<string> */
    private const SUPPORTED_DTYPES = [
        'bool',
        'float32',
        'float64',
        'int8',
        'uint8',
        'int16',
        'uint16',
        'int32',
        'uint32',
        'int64',
        'uint64',
        'string',
    ];

    /** @var list<int> */
    private array $shape;

    /** @var list<mixed> */
    private array $values;

    /**
     * @param  list<int>  $shape
     * @param  list<mixed>  $values
     */
    public function __construct(
        private string $dtype,
        array $shape,
        array $values,
    ) {
        $this->dtype = self::canonicalDtype($dtype);
        $this->shape = self::normalizeShape($shape);

        if (! array_is_list($values) || self::elementCount($this->shape) !== count($values)) {
            throw new TensorShapeException(sprintf(
                'Tensor value count does not match shape %s: expected %d, received %d.',
                self::formatShape($this->shape),
                self::elementCount($this->shape),
                count($values),
            ));
        }

        $this->values = $values;
    }

    /**
     * @param  list<mixed>  $values
     * @param  list<int>  $shape
     */
    public static function fromArray(array $values, string $dtype, array $shape): self
    {
        return new self($dtype, $shape, $values);
    }

    public static function canonicalDtype(string $dtype): string
    {
        $normalized = strtolower(trim($dtype));
        $aliases = [
            'boolean' => 'bool',
            'double' => 'float64',
            'float' => 'float32',
        ];
        $normalized = $aliases[$normalized] ?? $normalized;

        if (! in_array($normalized, self::SUPPORTED_DTYPES, true)) {
            throw new UnsupportedDtypeException(sprintf(
                'Unsupported tensor dtype "%s". Supported dtypes: %s.',
                $dtype,
                implode(', ', self::SUPPORTED_DTYPES),
            ));
        }

        return $normalized;
    }

    public function dtype(): string
    {
        return $this->dtype;
    }

    /** @return list<int> */
    public function shape(): array
    {
        return $this->shape;
    }

    /** @return list<mixed> */
    public function values(): array
    {
        return $this->values;
    }

    /** @return list<mixed> */
    public function toArray(): array
    {
        return $this->values;
    }

    /** @param list<int> $shape */
    private static function normalizeShape(array $shape): array
    {
        if (! array_is_list($shape)) {
            throw new TensorShapeException('Tensor shape must be a zero-based list of non-negative dimensions.');
        }

        foreach ($shape as $dimension) {
            if (! is_int($dimension) || $dimension < 0) {
                throw new TensorShapeException('Tensor shape dimensions must be non-negative integers.');
            }
        }

        return $shape;
    }

    /** @param list<int> $shape */
    private static function elementCount(array $shape): int
    {
        $count = 1;

        foreach ($shape as $dimension) {
            $count *= $dimension;
        }

        return $count;
    }

    /** @param list<int> $shape */
    private static function formatShape(array $shape): string
    {
        return '['.implode(', ', $shape).']';
    }
}
