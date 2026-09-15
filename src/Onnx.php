<?php

namespace Eloquage\Onnx;

/**
 * Primary entrypoint for eloquage/onnx.
 *
 * Pure-PHP implementation lives here. Optional TypePHP/native acceleration
 * can be added under native/ later without changing this public API.
 */
final class Onnx
{
    public function name(): string
    {
        return 'onnx';
    }
}
