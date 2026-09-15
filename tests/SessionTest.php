<?php

use Eloquage\Onnx\Exceptions\InferenceException;
use Eloquage\Onnx\Exceptions\InputValidationException;
use Eloquage\Onnx\Exceptions\ModelPathException;
use Eloquage\Onnx\Exceptions\RuntimeUnavailableException;
use Eloquage\Onnx\Exceptions\TensorShapeException;
use Eloquage\Onnx\Exceptions\UnsupportedDtypeException;
use Eloquage\Onnx\FakeSession;
use Eloquage\Onnx\Onnx;
use Eloquage\Onnx\OrtSession;
use Eloquage\Onnx\Tensor;
use Eloquage\Onnx\TensorSpec;

final class FakeOrtValue
{
    public bool $disposed = false;

    public function __construct(
        private array $values,
        private array $shape,
    ) {}

    public function toArray(): array
    {
        return $this->values;
    }

    public function shape(): array
    {
        return $this->shape;
    }

    public function dispose(): void
    {
        $this->disposed = true;
    }
}

final class FakeOrtRuntime
{
    public bool $disposed = false;

    /** @var list<array<string, object>> */
    public array $receivedInputs = [];

    public function __construct(
        private array $inputMetadata,
        private array $outputMetadata,
        private array $runtimeOutputs,
    ) {}

    public function inputs(): array
    {
        return $this->inputMetadata;
    }

    public function outputs(): array
    {
        return $this->outputMetadata;
    }

    public function run(array $inputs): array
    {
        $this->receivedInputs[] = $inputs;

        return $this->runtimeOutputs;
    }

    public function dispose(): void
    {
        $this->disposed = true;
    }
}

function onnx_tensor(string $dtype, array $shape, array $values): Tensor
{
    return new Tensor($dtype, $shape, $values);
}

function onnx_model_path(): string
{
    $path = tempnam(sys_get_temp_dir(), 'eloquage-onnx-');

    if ($path === false || file_put_contents($path, 'model') === false) {
        throw new RuntimeException('Could not create an ONNX model test file.');
    }

    return $path;
}

it('preserves the identity entrypoint and exposes a factory for production sessions', function () {
    expect((new Onnx)->name())->toBe('onnx')
        ->and(Onnx::class)->not->toHaveMethod('run');

    $path = onnx_model_path();

    try {
        expect(fn () => Onnx::session($path))->toThrow(RuntimeUnavailableException::class);
    } finally {
        unlink($path);
    }
});

it('runs a named fake session and exposes metadata', function () {
    $session = new FakeSession(
        'demo-model',
        [new TensorSpec('input_ids', 'float32', [null, 2])],
        [new TensorSpec('scores', 'float32', [1])],
        ['scores' => onnx_tensor('float32', [1], [0.75])],
    );

    $outputs = $session->run([
        'input_ids' => onnx_tensor('float32', [3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]);

    expect($session->modelPath())->toBe('demo-model')
        ->and($session->inputNames())->toBe(['input_ids'])
        ->and($session->outputNames())->toBe(['scores'])
        ->and($session->inputs()['input_ids']->shape())->toBe([null, 2])
        ->and($session->inputs()['input_ids']->isDynamicDimension(0))->toBeTrue()
        ->and($session->outputs()['scores']->dtype())->toBe('float32')
        ->and($outputs['scores']->values())->toBe([0.75]);
});

it('supports callable fake output fixtures and rejects invalid output fixtures', function () {
    $session = new FakeSession(
        'demo-model',
        [new TensorSpec('input', 'int64', [1])],
        [new TensorSpec('output', 'int64', [1])],
        ['output' => static fn (array $inputs): Tensor => new Tensor('int64', [1], [$inputs['input']->values()[0] + 1])],
    );

    expect($session->run(['input' => onnx_tensor('int64', [1], [4])])['output']->values())->toBe([5]);

    $invalid = new FakeSession(
        'demo-model',
        [new TensorSpec('input', 'int64', [1])],
        [new TensorSpec('output', 'int64', [1])],
        ['output' => static fn (): string => 'not a tensor'],
    );

    expect(fn () => $invalid->run(['input' => onnx_tensor('int64', [1], [4])]))
        ->toThrow(InferenceException::class, 'did not return a Tensor');
});

it('validates fake model identifiers, names, required inputs, dtypes, ranks, and fixed dimensions', function () {
    expect(fn () => new FakeSession('', [], []))->toThrow(ModelPathException::class)
        ->and(fn () => new FakeSession('demo', [new TensorSpec('input', 'float32', [1])], [], ['unknown' => onnx_tensor('float32', [1], [1.0])]))
        ->toThrow(InputValidationException::class, 'Unknown fake-session output');

    $session = new FakeSession(
        'demo-model',
        [new TensorSpec('input', 'float32', [2, 3])],
        [new TensorSpec('output', 'float32', [1])],
        ['output' => onnx_tensor('float32', [1], [1.0])],
    );

    expect(fn () => $session->run(['unknown' => onnx_tensor('float32', [2, 3], array_fill(0, 6, 1.0))]))
        ->toThrow(InputValidationException::class, 'Expected one of: input')
        ->and(fn () => $session->run([]))
        ->toThrow(InputValidationException::class, 'Missing required session input "input"')
        ->and(fn () => $session->run(['input' => onnx_tensor('float64', [2, 3], array_fill(0, 6, 1.0))]))
        ->toThrow(InputValidationException::class, 'expected "float32"')
        ->and(fn () => $session->run(['input' => onnx_tensor('float32', [3], array_fill(0, 3, 1.0))]))
        ->toThrow(TensorShapeException::class, 'expected rank 2')
        ->and(fn () => $session->run(['input' => onnx_tensor('float32', [2, 4], array_fill(0, 8, 1.0))]))
        ->toThrow(TensorShapeException::class, 'expected 3');
});

it('rejects unsupported and malformed tensor dtypes and shapes', function () {
    expect(fn () => new Tensor('complex64', [1], [1]))
        ->toThrow(UnsupportedDtypeException::class, 'Supported dtypes')
        ->and(fn () => new Tensor('float32', [-1], [1.0]))
        ->toThrow(TensorShapeException::class, 'non-negative')
        ->and(fn () => new Tensor('float32', [2], [1.0]))
        ->toThrow(TensorShapeException::class, 'expected 2')
        ->and(fn () => new TensorSpec('input', 'float32', ['']))
        ->toThrow(TensorShapeException::class, 'dynamic markers')
        ->and(fn () => new TensorSpec('input', 'float32', [1 => 2]))
        ->toThrow(TensorShapeException::class, 'zero-based list')
        ->and(fn () => new TensorSpec('', 'float32', [1]))
        ->toThrow(InputValidationException::class, 'must not be empty');
});

it('translates an optional runtime session and disposes runtime values', function () {
    $path = onnx_model_path();
    $output = new FakeOrtValue([0.25, 0.75], [1, 2]);
    $runtime = new FakeOrtRuntime(
        ['input' => ['dtype' => 'FLOAT', 'shape' => [-1, 2]]],
        ['probabilities' => ['dtype' => 'FLOAT', 'shape' => [1, 2]]],
        ['probabilities' => $output],
    );

    try {
        $session = new OrtSession(
            $path,
            $runtime,
            static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape()),
        );
        $result = $session->run(['input' => onnx_tensor('float32', [4, 2], array_fill(0, 8, 0.5))]);

        expect($session->inputs()['input']->shape())->toBe([null, 2])
            ->and($session->outputs()['probabilities']->shape())->toBe([1, 2])
            ->and($runtime->receivedInputs)->toHaveCount(1)
            ->and($result['probabilities']->values())->toBe([0.25, 0.75])
            ->and($output->disposed)->toBeTrue();

        $session->dispose();
        $session->dispose();

        expect($runtime->disposed)->toBeTrue();
    } finally {
        unlink($path);
    }
});

it('normalizes positional runtime outputs and maps runtime failures to package errors', function () {
    $path = onnx_model_path();
    $runtime = new class([['input' => ['dtype' => 'INT64', 'shape' => [1]]], ['first' => ['dtype' => 'INT64', 'shape' => [1]], 'second' => ['dtype' => 'INT64', 'shape' => [1]]]])
    {
        public bool $disposed = false;

        public function __construct(private array $metadata) {}

        public function inputs(): array
        {
            return $this->metadata[0];
        }

        public function outputs(): array
        {
            return $this->metadata[1];
        }

        public function run(array $inputs): array
        {
            return [new FakeOrtValue([2], [1]), new FakeOrtValue([3], [1])];
        }

        public function dispose(): void
        {
            $this->disposed = true;
        }
    };

    try {
        $session = new OrtSession(
            $path,
            $runtime,
            static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape()),
        );
        $result = $session->run(['input' => onnx_tensor('int64', [1], [1])]);

        expect($result['first']->values())->toBe([2])
            ->and($result['second']->values())->toBe([3]);
    } finally {
        unlink($path);
    }

    $failingRuntime = new class
    {
        public function inputs(): array
        {
            return ['input' => ['dtype' => 'FLOAT', 'shape' => [1]]];
        }

        public function outputs(): array
        {
            return ['output' => ['dtype' => 'FLOAT', 'shape' => [1]]];
        }

        public function run(array $inputs): array
        {
            throw new RuntimeException('native failure');
        }
    };
    $path = onnx_model_path();

    try {
        $session = new OrtSession(
            $path,
            $failingRuntime,
            static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape()),
        );

        expect(fn () => $session->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'native failure');
    } finally {
        unlink($path);
    }
});

it('rejects invalid production model paths before checking optional runtime', function () {
    expect(fn () => new OrtSession(''))
        ->toThrow(ModelPathException::class, 'local file path')
        ->and(fn () => new OrtSession('/definitely/missing/model.onnx'))
        ->toThrow(ModelPathException::class, 'readable file')
        ->and(fn () => new OrtSession('https://example.test/model.onnx'))
        ->toThrow(ModelPathException::class, 'local file path');
});

it('covers tensor aliases, scalar tensors, dynamic metadata, and session indexing guards', function () {
    $tensor = Tensor::fromArray([1], 'double', [1]);
    $boolean = new Tensor('boolean', [], [true]);
    $dynamic = new TensorSpec('dynamic', 'float', [-1, 'width', null]);

    expect($tensor->dtype())->toBe('float64')
        ->and($tensor->toArray())->toBe([1])
        ->and($boolean->dtype())->toBe('bool')
        ->and($dynamic->dtype())->toBe('float32')
        ->and($dynamic->rank())->toBe(3)
        ->and($dynamic->isDynamicDimension(1))->toBeTrue()
        ->and($dynamic->isDynamicDimension(0))->toBeTrue()
        ->and($dynamic->isDynamicDimension(3))->toBeFalse();
})->throws(TensorShapeException::class, 'no dimension 3');

it('rejects non-tensor inputs, duplicate specifications, and invalid fake fixtures', function () {
    $specification = new TensorSpec('input', 'float32', [1]);

    expect(fn () => TensorSpec::index([$specification, $specification]))
        ->toThrow(InputValidationException::class, 'declared more than once')
        ->and(fn () => TensorSpec::index(['not a specification']))
        ->toThrow(InputValidationException::class, 'only TensorSpec instances')
        ->and(fn () => new FakeSession('demo', [$specification], [$specification], ['input' => new stdClass]))
        ->toThrow(InputValidationException::class, 'Tensor or callable')
        ->and(fn () => (new FakeSession('demo', [$specification], [$specification]))->run(['input' => 'not a tensor']))
        ->toThrow(InputValidationException::class, 'must be a Tensor instance');

    $missingFixture = new FakeSession('demo', [$specification], [$specification]);
    expect(fn () => $missingFixture->run(['input' => onnx_tensor('float32', [1], [1.0])]))
        ->toThrow(InferenceException::class, 'has no fixture');

    $wrongFixture = new FakeSession('demo', [$specification], [$specification], [
        'input' => onnx_tensor('float32', [2], [1.0, 2.0]),
    ]);
    expect(fn () => $wrongFixture->run(['input' => onnx_tensor('float32', [1], [1.0])]))
        ->toThrow(InferenceException::class, 'does not satisfy');
});

it('reads object and traversable runtime metadata and normalizes nested outputs', function () {
    $path = onnx_model_path();
    $inputMetadata = new class
    {
        public function getName(): string
        {
            return 'input';
        }

        public function getDataType(): object
        {
            return new class
            {
                public string $name = 'FLOAT';
            };
        }

        public function getSymbolicShape(): Traversable
        {
            return new ArrayIterator([-1, 2]);
        }
    };
    $firstOutputMetadata = new class
    {
        public function getName(): string
        {
            return 'first';
        }

        public function getDataType(): object
        {
            return new class
            {
                public function name(): string
                {
                    return 'FLOAT';
                }
            };
        }

        public function getShape(): array
        {
            return [1, 2];
        }
    };
    $secondOutputMetadata = new class
    {
        public function getName(): string
        {
            return 'second';
        }

        public function getDataType(): object
        {
            return new class
            {
                public function value(): string
                {
                    return 'FLOAT';
                }
            };
        }

        public function getShape(): array
        {
            return [1];
        }
    };
    $runtime = new class($inputMetadata, $firstOutputMetadata, $secondOutputMetadata)
    {
        public function __construct(private object $input, private object $first, private object $second) {}

        public function inputs(): Traversable
        {
            return new ArrayIterator([$this->input]);
        }

        public function outputs(): Traversable
        {
            return new ArrayIterator([$this->first, $this->second]);
        }

        public function run(array $inputs): array
        {
            return [
                new FakeOrtValue([[1.0, 2.0]], [1, 2]),
                new FakeOrtValue([3.0], [1]),
            ];
        }
    };

    try {
        $session = new OrtSession(
            $path,
            $runtime,
            static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape()),
        );
        $outputs = $session->run(['input' => onnx_tensor('float32', [2, 2], [1.0, 2.0, 3.0, 4.0])]);

        expect($session->modelPath())->toBe($path)
            ->and($session->inputNames())->toBe(['input'])
            ->and($session->outputNames())->toBe(['first', 'second'])
            ->and($outputs['first']->values())->toBe([1.0, 2.0])
            ->and($outputs['second']->values())->toBe([3.0]);
    } finally {
        unlink($path);
    }
});

it('covers optional runtime metadata and output error boundaries', function () {
    $path = onnx_model_path();

    $cases = [
        [
            'runtime' => new class
            {
                public function inputs(): string
                {
                    return 'invalid';
                }

                public function dispose(): void
                {
                    throw new RuntimeException('cleanup failure');
                }
            },
            'exception' => RuntimeUnavailableException::class,
            'message' => 'invalid input metadata',
        ],
        [
            'runtime' => new class
            {
                public function inputs(): array
                {
                    return [];
                }
            },
            'exception' => RuntimeUnavailableException::class,
            'message' => 'no input metadata',
        ],
        [
            'runtime' => new class
            {
                public function inputs(): array
                {
                    throw new RuntimeException('metadata failure');
                }
            },
            'exception' => RuntimeUnavailableException::class,
            'message' => 'unusable model metadata',
        ],
        [
            'runtime' => new class
            {
                public function inputs(): array
                {
                    return [['dtype' => 'FLOAT', 'shape' => [1]]];
                }

                public function outputs(): array
                {
                    return [['dtype' => 'FLOAT', 'shape' => [1]]];
                }
            },
            'exception' => RuntimeUnavailableException::class,
            'message' => 'missing a required property',
        ],
    ];

    try {
        foreach ($cases as $case) {
            expect(fn () => new OrtSession($path, $case['runtime'], static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape())))
                ->toThrow($case['exception'], $case['message']);
        }
    } finally {
        unlink($path);
    }
});

it('handles output fallback shapes and rejects malformed runtime results', function () {
    $path = onnx_model_path();
    $makeSession = static function (object $runtime) use ($path): OrtSession {
        return new OrtSession(
            $path,
            $runtime,
            static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape()),
        );
    };

    $baseRuntime = static function (array $outputMetadata, array $runtimeOutputs): object {
        return new class($outputMetadata, $runtimeOutputs)
        {
            public function __construct(private array $outputMetadata, private array $runtimeOutputs) {}

            public function inputs(): array
            {
                return ['input' => ['dtype' => 'FLOAT', 'shape' => [1]]];
            }

            public function outputs(): array
            {
                return $this->outputMetadata;
            }

            public function run(array $inputs): array
            {
                return $this->runtimeOutputs;
            }
        };
    };

    try {
        $dynamicSession = $makeSession(
            $baseRuntime(['output' => ['dtype' => 'FLOAT', 'shape' => [-1]]], ['output' => [1.0, 2.0]]),
        );
        expect($dynamicSession->run(['input' => onnx_tensor('float32', [1], [1.0])])['output']->shape())->toBe([2]);

        $dynamicRankTwo = $makeSession(
            $baseRuntime(['output' => ['dtype' => 'FLOAT', 'shape' => [-1, 2]]], ['output' => [1.0, 2.0]]),
        );
        expect(fn () => $dynamicRankTwo->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'concrete shape');

        $invalidOutput = $makeSession(
            $baseRuntime(['output' => ['dtype' => 'FLOAT', 'shape' => [1]]], ['output' => new stdClass]),
        );
        expect(fn () => $invalidOutput->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'not array-like');

        $invalidList = $makeSession(
            $baseRuntime(['output' => ['dtype' => 'FLOAT', 'shape' => [1]]], []),
        );
        expect(fn () => $invalidList->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'incomplete output list');

        $missingOutput = $makeSession(
            $baseRuntime(['output' => ['dtype' => 'FLOAT', 'shape' => [1]]], ['other' => [1.0]]),
        );
        expect(fn () => $missingOutput->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'required output');

        $wrongShape = $makeSession(
            $baseRuntime(['output' => ['dtype' => 'FLOAT', 'shape' => [2]]], ['output' => new FakeOrtValue([1.0], [1])]),
        );
        expect(fn () => $wrongShape->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'does not satisfy');
    } finally {
        unlink($path);
    }
});

it('uses the default runtime value factory when the optional binding classes are available', function () {
    if (! class_exists('PhpMlKit\\ONNXRuntime\\Enums\\DataType')) {
        eval('namespace PhpMlKit\\ONNXRuntime\\Enums; final class DataType { public const FLOAT = "FLOAT"; }');
    }

    if (! class_exists('PhpMlKit\\ONNXRuntime\\OrtValue')) {
        eval('namespace PhpMlKit\\ONNXRuntime; final class OrtValue { public static function fromArray(array $values, mixed $dataType, ?array $shape = null): object { return new self; } }');
    }

    $path = onnx_model_path();
    $runtime = new class
    {
        public array $receivedInputs = [];

        public function inputs(): array
        {
            return ['input' => ['dtype' => 'FLOAT', 'shape' => [1]]];
        }

        public function outputs(): array
        {
            return ['output' => ['dtype' => 'FLOAT', 'shape' => [1]]];
        }

        public function run(array $inputs): array
        {
            $this->receivedInputs[] = $inputs;

            return ['output' => [1.0]];
        }
    };

    try {
        $session = new OrtSession($path, $runtime);
        $session->run(['input' => onnx_tensor('float32', [1], [1.0])]);

        expect($runtime->receivedInputs[0]['input'])->toBeObject();
    } finally {
        unlink($path);
    }
});

it('maps missing runtime methods, invalid runtime output formats, and cleanup failures', function () {
    $path = onnx_model_path();
    $factory = static fn (Tensor $tensor): FakeOrtValue => new FakeOrtValue($tensor->values(), $tensor->shape());

    try {
        expect(fn () => new OrtSession($path, new stdClass, $factory))
            ->toThrow(RuntimeUnavailableException::class, 'does not expose inputs');

        $runtime = new class
        {
            public function inputs(): array
            {
                return ['input' => ['dtype' => 'FLOAT', 'shape' => [1]]];
            }

            public function outputs(): array
            {
                return ['output' => ['dtype' => 'FLOAT', 'shape' => [1]]];
            }

            public function run(array $inputs): string
            {
                return 'invalid';
            }
        };
        $session = new OrtSession($path, $runtime, $factory);

        expect(fn () => $session->run(['input' => onnx_tensor('float32', [1], [1.0])]))
            ->toThrow(InferenceException::class, 'invalid format');

        $cleanupRuntime = new class
        {
            public bool $disposed = false;

            public function inputs(): array
            {
                return ['input' => ['dtype' => 'FLOAT', 'shape' => [1]]];
            }

            public function outputs(): array
            {
                return ['output' => ['dtype' => 'FLOAT', 'shape' => [1]]];
            }

            public function run(array $inputs): array
            {
                return ['output' => new class
                {
                    public function toArray(): array
                    {
                        return [1.0];
                    }

                    public function shape(): array
                    {
                        return [1];
                    }

                    public function dispose(): void
                    {
                        throw new RuntimeException('value cleanup failure');
                    }
                }];
            }

            public function dispose(): void
            {
                $this->disposed = true;
            }
        };
        $session = new OrtSession($path, $cleanupRuntime, $factory);
        expect($session->run(['input' => onnx_tensor('float32', [1], [1.0])])['output']->values())->toBe([1.0]);
        $session->dispose();
        expect($cleanupRuntime->disposed)->toBeTrue();
    } finally {
        unlink($path);
    }
});
