<?php

use Eloquage\Onnx\Onnx;

it('bootstraps the package entrypoint', function () {
    $instance = new Onnx;

    expect($instance->name())->toBe('onnx');
});
