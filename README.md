# removebg-onnx-web-example

This repository contains a web-based application that uses ONNX Runtime to remove backgrounds from images. It's an example of how to integrate ONNX Runtime in a web environment. The project utilizes only *onnxruntime-web* for model inference, without relying on any other JavaScript libraries.

Try here: https://pstwh.github.io/removebg-onnx-web-example/


### Example

<table>
    <thead>
        <tr>
            <td>input</td>
            <td>output</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://raw.githubusercontent.com/pstwh/removebg-onnx-web-example/main/examples/input.jpg" width="512" /></td>
            <td><img src="https://raw.githubusercontent.com/pstwh/removebg-onnx-web-example/main/examples/output.png" width="512" /></td>
        </tr>
    </tbody>
</table>
* The removebg output image example is compressed in this repo to shrink size.

### Model

As an example, the model is really small. It's an u2netp from [source](https://github.com/xuebinqin/U-2-Net).