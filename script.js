const IMAGE_WIDTH = 320;
const IMAGE_HEIGHT = 320;
const IMAGE_CHANNELS = 3;
const ONNX_MODEL_PATH = "./assets/u2netp.onnx";
const ONNX_PROCESSOR_PATH = "./assets/output_processor.onnx";
const INPUT_TENSOR_NAME = "input.1";
const OUTPUT_TENSOR_NAME = "1959";
const OUTPUT_RESIZED_TENSOR_NAME = "output";
const MASK_TENSOR_NAME = "mask";
const ORIGINAL_SHAPE_TENSOR_NAME = "original_shape";

const CANVAS_ID = "imageCanvas";
const DROP_AREA_ID = "dropArea";
const INPUT_ELEMENT_ID = "imageInput";
const LOADING_SPINNER_ID = "loadingSpinner";
const REMOVE_BUTTON_ID = "removeButton";

const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

async function initializeSession(modelPath, options) {
    return await ort.InferenceSession.create(modelPath, options);
}

function setupCanvas(canvasId, width, height) {
    const canvas = document.getElementById(canvasId);
    canvas.width = width;
    canvas.height = height;
    return canvas.getContext("2d");
}

function preprocessImage(image, width, height, mean, std) {
    const offscreenCanvas = document.createElement("canvas");
    const offscreenCtx = offscreenCanvas.getContext("2d");
    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
    offscreenCtx.drawImage(image, 0, 0, width, height);

    const imageData = offscreenCtx.getImageData(0, 0, width, height);
    const data = imageData.data;

    const pixels = new Float32Array(1 * IMAGE_CHANNELS * width * height);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const r = (data[index] / 255 - mean[0]) / std[0];
            const g = (data[index + 1] / 255 - mean[1]) / std[1];
            const b = (data[index + 2] / 255 - mean[2]) / std[2];

            const newIndex = y * width + x;
            pixels[newIndex] = r;
            pixels[newIndex + width * height] = g;
            pixels[newIndex + 2 * width * height] = b;
        }
    }

    return new ort.Tensor("float32", pixels, [1, IMAGE_CHANNELS, width, height]);
}

function updateCanvasWithMask(ctx, imageData, resizedMask, imageWidth, imageHeight) {
    for (let y = 0; y < imageHeight; y++) {
        for (let x = 0; x < imageWidth; x++) {
            const index = (y * imageWidth + x) * 4;
            imageData.data[index + 3] = 255 * resizedMask[y * imageWidth + x];
        }
    }
    ctx.putImageData(imageData, 0, 0);
}

function toggleLoadingSpinner(show) {
    document.getElementById(LOADING_SPINNER_ID).style.display = show ? 'block' : 'none';
}

function toggleDropArea(show) {
    document.getElementById(DROP_AREA_ID).style.display = show ? 'flex' : 'none';
}

function toggleRemoveButton(show) {
    document.getElementById(REMOVE_BUTTON_ID).style.display = show ? 'block' : 'none';
}

async function handleImageUpload(file) {
    const modelSessionOptions = {
        executionProviders: ["webgpu", "wasm"]
    };

    const processorSessionOptions = {
        executionProviders: ["wasm"]
    };

    const sessionModel = await initializeSession(ONNX_MODEL_PATH, modelSessionOptions);
    const sessionProcessor = await initializeSession(ONNX_PROCESSOR_PATH, processorSessionOptions);

    if (file) {
        toggleLoadingSpinner(true);
        toggleDropArea(false);

        const reader = new FileReader();

        reader.onload = function (event) {
            const image = new Image();
            image.onload = async function () {
                const ctx = setupCanvas(CANVAS_ID, image.width, image.height);
                ctx.drawImage(image, 0, 0);
                const imageDataSource = ctx.getImageData(0, 0, image.width, image.height);

                const pixelsTensor = preprocessImage(image, IMAGE_WIDTH, IMAGE_HEIGHT, mean, std);
                const inputDictModel = { [INPUT_TENSOR_NAME]: pixelsTensor };

                try {
                    const outputModel = await sessionModel.run(inputDictModel);
                    const mask = outputModel[OUTPUT_TENSOR_NAME].data;

                    const maskTensor = new ort.Tensor("float32", mask, [1, IMAGE_WIDTH, IMAGE_HEIGHT]);
                    const shapeTensor = new ort.Tensor("int64", [image.height, image.width], [2]);
                    const inputDictProcessor = { [MASK_TENSOR_NAME]: maskTensor, [ORIGINAL_SHAPE_TENSOR_NAME]: shapeTensor };

                    const outputProcessor = await sessionProcessor.run(inputDictProcessor);
                    const resizedMask = outputProcessor[OUTPUT_RESIZED_TENSOR_NAME].data;

                    updateCanvasWithMask(ctx, imageDataSource, resizedMask, image.width, image.height);
                } catch (error) {
                    console.error("Error during inference: ", error);
                } finally {
                    toggleLoadingSpinner(false);
                    toggleRemoveButton(true);
                }
            };
            image.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById(DROP_AREA_ID).classList.add('hover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById(DROP_AREA_ID).classList.remove('hover');
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById(DROP_AREA_ID).classList.remove('hover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
}

function handleClick() {
    document.getElementById(INPUT_ELEMENT_ID).click();
}

function resetFileInput() {
    const fileInput = document.getElementById(INPUT_ELEMENT_ID);
    fileInput.value = "";
}

function handleRemoveImage() {
    const canvas = document.getElementById(CANVAS_ID);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    resetFileInput();
    toggleRemoveButton(false);
    toggleDropArea(true);
}

document.getElementById(INPUT_ELEMENT_ID).addEventListener("change", function (event) {
    handleImageUpload(event.target.files[0]);
});

const dropArea = document.getElementById(DROP_AREA_ID);
dropArea.addEventListener("dragover", handleDragOver);
dropArea.addEventListener("dragleave", handleDragLeave);
dropArea.addEventListener("drop", handleDrop);
dropArea.addEventListener("click", handleClick);

document.getElementById(REMOVE_BUTTON_ID).addEventListener("click", handleRemoveImage);
