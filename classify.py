import argparse
import sys
import time
import platform
import cv2
import numpy as np
import dataclasses
from typing import List

# 嘗試載入 tflite_runtime，若無則使用 tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate


@dataclasses.dataclass
class ImageClassifierOptions:
    """設定影像分類器的參數"""
    enable_edgetpu: bool = False
    max_results: int = 3
    num_threads: int = 1
    score_threshold: float = 0.0
    label_path: str = "labels.txt"  # 需要提供標籤檔


@dataclasses.dataclass
class Category:
    """儲存分類結果"""
    label: str
    score: float


def edgetpu_lib_name():
    """取得 EdgeTPU 的動態連結庫名稱"""
    return {
        'Darwin': 'libedgetpu.1.dylib',
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
    }.get(platform.system(), None)


class ImageClassifier:
    """TensorFlow Lite 影像分類器"""

    def __init__(self, model_path: str, options: ImageClassifierOptions):
        """初始化模型"""
        self._options = options
        self._load_labels(options.label_path)

        # 載入模型
        if options.enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError("當前 OS 不支援 Coral EdgeTPU")
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate(edgetpu_lib_name())],
                num_threads=options.num_threads
            )
        else:
            interpreter = Interpreter(model_path=model_path, num_threads=options.num_threads)
        
        interpreter.allocate_tensors()

        # 取得輸入與輸出資訊
        self._input_details = interpreter.get_input_details()
        self._output_details = interpreter.get_output_details()

        self._input_height = self._input_details[0]['shape'][1]
        self._input_width = self._input_details[0]['shape'][2]

        self._is_quantized_input = self._input_details[0]['dtype'] == np.uint8
        self._is_quantized_output = self._output_details[0]['dtype'] == np.uint8

        self._interpreter = interpreter

    def _load_labels(self, label_path: str):
        """讀取標籤檔"""
        try:
            with open(label_path, "r") as f:
                self._label_list = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"警告：無法找到標籤檔 {label_path}，請確保標籤檔存在")
            self._label_list = []

    def _set_input_tensor(self, image: np.ndarray):
        """設定模型的輸入張量"""
        tensor_index = self._input_details[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """影像前處理"""
        input_tensor = cv2.resize(image, (self._input_width, self._input_height))
        if not self._is_quantized_input:
            input_tensor = np.float32(input_tensor) / 255.0
        return input_tensor

    def classify(self, image: np.ndarray) -> List[Category]:
        """執行影像分類"""
        image = self._preprocess(image)
        self._set_input_tensor(image)
        self._interpreter.invoke()
        output_tensor = np.squeeze(self._interpreter.get_tensor(self._output_details[0]['index']))

        return self._postprocess(output_tensor)

    def _postprocess(self, output_tensor: np.ndarray) -> List[Category]:
        """處理模型輸出"""
        if self._is_quantized_output:
            scale, zero_point = self._output_details[0]['quantization']
            output_tensor = scale * (output_tensor - zero_point)

        # 取得排序後的結果
        sorted_indices = np.argsort(output_tensor)[::-1]
        categories = [
            Category(label=self._label_list[idx] if idx < len(self._label_list) else "Unknown", 
                     score=output_tensor[idx])
            for idx in sorted_indices[:self._options.max_results]
        ]

        return [c for c in categories if c.score >= self._options.score_threshold]


# 影像視覺化參數
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


def run(model: str, label_path: str, max_results: int, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
    """持續從攝影機取得影像並進行分類"""
    
    options = ImageClassifierOptions(
        num_threads=num_threads,
        max_results=max_results,
        enable_edgetpu=enable_edgetpu,
        label_path=label_path
    )
    classifier = ImageClassifier(model, options)

    counter, fps = 0, 0
    start_time = time.time()
    url = "http://<IP>:4747/video"
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: 無法讀取攝影機影像，請檢查攝影機設定。')

        counter += 1
        image = cv2.flip(image, 1)

        categories = classifier.classify(image)

        for idx, category in enumerate(categories):
            result_text = f"{category.label} ({round(category.score, 2)})"
            text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
            end_time = time.time()
            fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        fps_text = f"FPS = {int(fps)}"
        text_location = (_LEFT_MARGIN, _ROW_SIZE)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('image_classification', image)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='TFLite 影像分類模型檔案名稱。',
        required=False,
        default='efficientnet_lite0.tflite')
    parser.add_argument(
        '--labels',
        help='標籤檔名稱。',
        required=False,
        default='labels.txt')
    parser.add_argument(
        '--maxResults',
        help='回傳的最大分類數量。',
        required=False,
        default=3)
    parser.add_argument(
        '--numThreads',
        help='運行模型的 CPU 執行緒數量。',
        required=False,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='是否啟用 EdgeTPU 加速。',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--cameraId', help='攝影機 ID。', required=False, default=0)
    parser.add_argument(
        '--frameWidth', help='攝影機影像寬度。', required=False, default=640)
    parser.add_argument(
        '--frameHeight', help='攝影機影像高度。', required=False, default=480)
    args = parser.parse_args()

    run(args.model, args.labels, int(args.maxResults), int(args.numThreads),
        bool(args.enableEdgeTPU), int(args.cameraId), int(args.frameWidth),
        int(args.frameHeight))


if __name__ == '__main__':
    main()
