<?php

namespace Eloquage\Onnx;

interface Session
{
    public function modelPath(): string;

    /** @return array<string, TensorSpec> */
    public function inputs(): array;

    /** @return array<string, TensorSpec> */
    public function outputs(): array;

    /** @return list<string> */
    public function inputNames(): array;

    /** @return list<string> */
    public function outputNames(): array;

    /** @param array<string, Tensor> $inputs @return array<string, Tensor> */
    public function run(array $inputs): array;
}
