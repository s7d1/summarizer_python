import os
import pynvml

os.environ["PYNVML_DLL_PATH"] = "C:/Windows/System32/nvml.dll"


def load_or_initialize_processor(identifier, model_dir, processor_class):
    if os.path.exists(os.path.join(model_dir, identifier)):
        return processor_class.from_pretrained(
            os.path.join(model_dir, identifier))
    else:
        return processor_class.from_pretrained(identifier, cache_dir=model_dir)


def load_or_initialize_model(identifier, model_dir, model_class):
    if os.path.exists(os.path.join(model_dir, identifier)):
        return model_class.from_pretrained(os.path.join(model_dir, identifier))
    else:
        return model_class.from_pretrained(identifier, cache_dir=model_dir)


def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_train_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
