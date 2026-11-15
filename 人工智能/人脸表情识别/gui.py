"""PyQt5 图片上传与摄像头表情识别界面。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_TASKS_IMPORT_ERROR = None
except Exception as exc:
    mp = None
    mp_tasks_python = None
    mp_vision = None
    MEDIAPIPE_TASKS_IMPORT_ERROR = exc

try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("缺少 PyQt5，请先运行：pip install PyQt5")
    raise

from utils.common import CLASS_NAMES
from utils.predict_utils import CLASS_NAMES_CN, load_emotion_model, predict_image


BASIC_LABELS_CN = CLASS_NAMES_CN
BASIC_LABELS_EN = CLASS_NAMES

COMPOUND_LABELS_CN = [
    "开心-惊讶",
    "开心-厌恶",
    "悲伤-恐惧",
    "悲伤-愤怒",
    "悲伤-惊讶",
    "悲伤-厌恶",
    "恐惧-愤怒",
    "恐惧-惊讶",
    "愤怒-惊讶",
    "愤怒-厌恶",
    "厌恶-惊讶",
]

COMPOUND_LABELS_EN = [
    "Happily Surprised",
    "Happily Disgusted",
    "Sadly Fearful",
    "Sadly Angry",
    "Sadly Surprised",
    "Sadly Disgusted",
    "Fearfully Angry",
    "Fearfully Surprised",
    "Angrily Surprised",
    "Angrily Disgusted",
    "Disgustedly Surprised",
]

RAFML_LABELS_CN = ["惊讶", "恐惧", "厌恶", "开心", "悲伤", "愤怒"]
RAFML_LABELS_EN = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger"]


TASK_OPTIONS = [
    {
        "title": "基本表情识别",
        "task_type": "basic_7",
        "output_mode": "七类 softmax 单标签分类",
        "labels_text": "愤怒、厌恶、恐惧、开心、悲伤、惊讶、中性",
        "description": "用于识别单一基本表情，输出七类基本表情概率，适合课程主实验、真实图片和摄像头演示。",
    },
    {
        "title": "复合表情识别",
        "task_type": "compound_11",
        "output_mode": "十一类 softmax 单标签分类",
        "labels_text": "开心-惊讶、开心-厌恶、悲伤-恐惧、悲伤-愤怒、悲伤-惊讶、悲伤-厌恶、恐惧-愤怒、恐惧-惊讶、愤怒-惊讶、愤怒-厌恶、厌恶-惊讶",
        "description": "用于识别固定类别的复合表情，例如开心-惊讶、悲伤-恐惧、愤怒-厌恶等，主要作为扩展实验展示。",
    },
    {
        "title": "混合表情识别",
        "task_type": "rafml_6",
        "output_mode": "六维 sigmoid 多标签识别",
        "labels_text": "惊讶、恐惧、厌恶、开心、悲伤、愤怒",
        "description": "用于判断多个基础情绪成分是否同时存在，不包含中性 Neutral，适合展示多标签混合表情识别。",
    },
]


MODEL_OPTIONS = [
    {
        "title": "AffectNet RGB Full MiniResNet",
        "task_type": "basic_7",
        "model_name": "rgb_mini_resnet",
        "num_classes": 7,
        "output_type": "softmax",
        "weights": "checkpoints/best_affectnet_rgb_full_40epoch.pth",
        "labels_cn": BASIC_LABELS_CN,
        "labels_en": BASIC_LABELS_EN,
        "probability_title": "七类基本表情概率",
        "recommendation": "推荐用于摄像头实时识别和真实场景图片识别。",
        "description": """模型说明：
该模型基于 AffectNet YOLO Format 数据集中除 Contempt 以外的有效样本训练，输入为 96×96 RGB 图像。AffectNet 图像来源更复杂，包含更多真实网络场景；根据当前电脑摄像头实时识别测试，该模型表现更稳定，推荐用于摄像头实时识别。

适用场景：
- AffectNet 风格的真实网络人脸图像；
- 表情变化较复杂的彩色图片；
- 摄像头实时表情识别；
- 用于和 RAF-DB RGB 模型进行对比分析。

模型特点：
- 使用更多 AffectNet 数据训练；
- 数据来源更复杂；
- 推荐用于当前 GUI 摄像头实时识别；
- 对部分真实场景图片可能有较好的泛化能力；
- 是当前 GUI 默认推荐模型。

注意事项：
- AffectNet YOLO 数据可能存在标注噪声；
- 在 RAF-DB 数据集正式测试中，仍建议同时参考 RAF-DB RGB MiniResNet 的结果。""",
    },
    {
        "title": "RAF-DB RGB MiniResNet",
        "task_type": "basic_7",
        "model_name": "rgb_mini_resnet",
        "num_classes": 7,
        "output_type": "softmax",
        "weights": "checkpoints/best_rafdb_rgb_mini_resnet.pth",
        "labels_cn": BASIC_LABELS_CN,
        "labels_en": BASIC_LABELS_EN,
        "probability_title": "七类基本表情概率",
        "recommendation": "适合真实彩色图片和 RAF-DB 数据分布。",
        "description": """模型说明：
该模型基于 RAF-DB 彩色真实场景人脸表情数据集从头训练，输入为 96×96 RGB 图像。它能够利用彩色图像信息，推荐用于 RAF-DB 数据集测试和真实彩色图片识别。

适用场景：
- 彩色人脸照片；
- RAF-DB 数据集测试；
- 真实彩色图片表情识别演示。

模型特点：
- 使用 RGB 彩色图像训练；
- 适合 RAF-DB 风格和真实彩色图片；
- 在 RAF-DB test set 上取得较好结果；
- 可作为真实彩色图片识别和数据集实验对比模型。""",
    },
    {
        "title": "AffectNet 预训练 + RAF-DB 微调 MiniResNet",
        "task_type": "basic_7",
        "model_name": "rgb_mini_resnet",
        "num_classes": 7,
        "output_type": "softmax",
        "weights": "checkpoints/best_affectnet_pretrained_rafdb_rgb_mini_resnet.pth",
        "labels_cn": BASIC_LABELS_CN,
        "labels_en": BASIC_LABELS_EN,
        "probability_title": "七类基本表情概率",
        "recommendation": "迁移学习实验模型，可与 RAF-DB RGB 从头训练模型和 AffectNet RGB Full 模型进行对比。",
        "missing_weight_message": "未找到 AffectNet 预训练 + RAF-DB 微调模型权重，请检查 checkpoints/best_affectnet_pretrained_rafdb_rgb_mini_resnet.pth 是否存在。",
        "description": """模型说明：
该模型先在 AffectNet RGB 数据集上进行预训练，再在 RAF-DB RGB 数据集上微调，属于迁移学习实验模型。它用于基本七类表情识别，可与 RAF-DB RGB 从头训练模型、AffectNet RGB Full 模型进行对比。

适用场景：
- 展示 AffectNet 预训练 + RAF-DB 微调的迁移学习流程；
- 与 RAF-DB RGB 从头训练模型进行对比；
- 与 AffectNet RGB Full 模型进行真实场景和摄像头效果对比。

注意事项：
- 该模型仍然输出基本七类表情；
- 输入为 96×96 RGB 图像；
- 它是扩展对比实验模型，不一定在所有场景下优于默认推荐模型。""",
    },
    {
        "title": "FER2013 灰度 MiniResNet",
        "task_type": "basic_7",
        "model_name": "mini_resnet",
        "num_classes": 7,
        "output_type": "softmax",
        "weights": "checkpoints/best_mini_resnet.pth",
        "labels_cn": BASIC_LABELS_CN,
        "labels_en": BASIC_LABELS_EN,
        "probability_title": "七类基本表情概率",
        "recommendation": "基于 FER2013 灰度图训练，适合展示基础实验。",
        "description": """模型说明：
该模型基于 FER2013 灰度人脸表情数据集训练，输入为 48×48 灰度图像。它适合展示 FER2013 主实验结果，也适合对灰度或低分辨率人脸图像进行测试。

适用场景：
- FER2013 数据集风格的灰度图像；
- 课程报告中的主实验模型对比；
- 低分辨率人脸表情识别演示。

注意事项：
- 该模型会将输入图片转为灰度；
- 对真实彩色照片和摄像头画面不一定是最优选择；
- 更适合作为 FER2013 主实验模型展示。""",
    },
    {
        "title": "RAF-DB Compound 复合表情 MiniResNet",
        "task_type": "compound_11",
        "model_name": "rgb_mini_resnet",
        "num_classes": 11,
        "output_type": "softmax",
        "weights": "checkpoints/best_compound_rgb_affectnet_pretrained.pth",
        "labels_cn": COMPOUND_LABELS_CN,
        "labels_en": COMPOUND_LABELS_EN,
        "probability_title": "十一类复合表情概率",
        "recommendation": "用于 11 类复合表情识别，例如开心-惊讶、愤怒-厌恶等。",
        "description": """模型说明：
该模型基于 RAF-DB Compound Emotion 数据集训练，输入为 96×96 RGB 图像，输出 11 类复合表情。它可以识别“开心-惊讶”“愤怒-厌恶”等细粒度复合表情类别。

适用场景：
- 展示 RAF-DB Compound 复合表情扩展实验；
- 分析单标签复合表情分类结果；
- 与基本七类表情模型进行课程项目对比。

注意事项：
- 复合表情类别语义相近、样本数量较少；
- 该模型主要作为扩展实验展示，不替代基本七类表情识别模型；
- 输出使用 softmax，因此每次预测会选择一个最可能的复合表情类别。""",
    },
    {
        "title": "RAF-ML 多标签混合表情 MiniResNet",
        "task_type": "rafml_6",
        "model_name": "rgb_mini_resnet",
        "num_classes": 6,
        "output_type": "sigmoid",
        "weights": "checkpoints/best_rafml_affectnet_pretrained.pth",
        "labels_cn": RAFML_LABELS_CN,
        "labels_en": RAFML_LABELS_EN,
        "probability_title": "六维混合表情概率",
        "threshold": 0.5,
        "recommendation": "用于 6 维多标签混合表情识别，允许多个情绪成分同时存在。",
        "description": """模型说明：
该模型基于 RAF-ML 多标签混合表情数据集训练，输入为 96×96 RGB 图像，输出 6 个基础情绪成分的 sigmoid 概率。它不是从多个类别中只选一个，而是允许一张人脸同时具有多个表情成分，例如“开心”和“惊讶”同时较高。

适用场景：
- 展示多标签混合表情识别扩展实验；
- 观察同一张图片中多个表情成分的概率；
- 分析真实场景中更细腻的混合情绪表达。

注意事项：
- RAF-ML 不包含 Neutral 维度；
- 默认阈值为 0.50，超过阈值的表情成分会同时列出；
- 输出使用 sigmoid，每个维度概率相互独立，不应按 softmax 七类理解。""",
    },
]


class EmotionRecognitionWindow(QMainWindow):
    """图片上传表情识别主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("人脸表情识别系统")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 720)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image_path: str | None = None
        self.current_image_source: str | None = None
        self.current_pixmap: QPixmap | None = None
        self.current_camera_pixmap: QPixmap | None = None
        self.preview_mode = "image"
        self.recognition_mode = "image"
        self.cached_model = None
        self.cached_model_key: tuple[str, str, int] | None = None
        self.camera_capture = None
        self.camera_running = False
        self.camera_frame_count = 0
        self.camera_predict_interval = 5
        self.face_location_mode = "yunet"
        self.yunet_model_path = "models/detectors/face_detection_yunet_2023mar.onnx"
        self.mediapipe_model_path = "models/detectors/blaze_face_short_range.tflite"
        self.yunet_detector = None
        self.yunet_input_size: tuple[int, int] | None = None
        self.yunet_warning_messages: set[str] = set()
        self.mediapipe_face_detector = None
        self.mediapipe_warning_messages: set[str] = set()
        self.mediapipe_detect_interval = 8
        self.last_mediapipe_face_box: tuple[int, int, int, int] | None = None
        self.last_mediapipe_detect_frame = -1
        self.camera_timer = QTimer(self)
        self.camera_timer.setInterval(30)
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.last_camera_prediction: dict | None = None
        self.last_camera_status: str | None = None
        self.last_camera_frame_rgb = None

        self._build_ui()
        self._apply_style()
        self.task_combo.setCurrentIndex(0)
        self._on_task_changed(self.task_combo.currentIndex())
        self._apply_mode_state()

    def _build_ui(self) -> None:
        """创建界面布局。"""
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(24, 16, 24, 24)
        root_layout.setSpacing(14)

        header = QFrame()
        header.setObjectName("HeaderPanel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 2)
        header_layout.setSpacing(2)

        title = QLabel("人脸表情识别系统")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("支持图片上传识别、摄像头实时识别、基本表情识别、复合表情识别与混合表情分析")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setObjectName("ContentSplitter")
        content_splitter.setChildrenCollapsible(False)
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([760, 460])
        root_layout.addWidget(content_splitter, 1)

        self.setCentralWidget(central_widget)

    def _create_left_panel(self) -> QWidget:
        """创建左侧图片/摄像头预览和操作区域。"""
        self.preview_group = QGroupBox("图片预览")
        self.preview_group.setObjectName("ImageCard")
        layout = QVBoxLayout(self.preview_group)
        layout.setContentsMargins(20, 22, 20, 20)
        layout.setSpacing(14)

        self.image_label = QLabel("请加载图片或开启摄像头")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(440, 330)
        self.image_label.setMaximumSize(720, 540)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1, Qt.AlignCenter)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)
        mode_label = QLabel("识别模式：")
        mode_label.setObjectName("FormLabel")
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("ModeComboBox")
        self.mode_combo.setMinimumHeight(40)
        self.mode_combo.addItem("图片上传识别模式", "image")
        self.mode_combo.addItem("摄像头实时识别模式", "camera")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.mode_status_label = QLabel("当前模式：图片上传识别")
        self.mode_status_label.setObjectName("ModeStatusLabel")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo, 1)
        mode_layout.addWidget(self.mode_status_label)
        layout.addLayout(mode_layout)

        self.image_button_widget = QWidget()
        button_layout = QHBoxLayout(self.image_button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)
        self.load_image_button = QPushButton("加载图片")
        self.predict_button = QPushButton("预测表情")
        self.load_image_button.setFixedHeight(44)
        self.predict_button.setFixedHeight(44)
        self.load_image_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.predict_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.load_image_button.clicked.connect(self.load_image)
        self.predict_button.clicked.connect(self.predict_current_image)
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.predict_button)
        layout.addWidget(self.image_button_widget)

        self.camera_button_widget = QWidget()
        camera_button_layout = QHBoxLayout(self.camera_button_widget)
        camera_button_layout.setContentsMargins(0, 0, 0, 0)
        camera_button_layout.setSpacing(12)
        self.start_camera_button = QPushButton("开始摄像头")
        self.stop_camera_button = QPushButton("停止摄像头")
        self.stop_camera_button.setObjectName("StopCameraButton")
        self.start_camera_button.setFixedHeight(44)
        self.stop_camera_button.setFixedHeight(44)
        self.start_camera_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_camera_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_camera_button.clicked.connect(lambda: self.start_camera())
        self.stop_camera_button.clicked.connect(lambda: self.stop_camera())
        self.stop_camera_button.setEnabled(False)
        camera_button_layout.addWidget(self.start_camera_button)
        camera_button_layout.addWidget(self.stop_camera_button)
        layout.addWidget(self.camera_button_widget)

        return self.preview_group

    def _create_right_panel(self) -> QWidget:
        """创建右侧模型设置和结果区域。"""
        scroll_area = QScrollArea()
        scroll_area.setObjectName("RightScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        panel = QWidget()
        panel.setObjectName("RightContent")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 8, 2)
        layout.setSpacing(10)

        setting_group = QGroupBox("模型与任务设置")
        setting_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        setting_layout = QGridLayout(setting_group)
        setting_layout.setHorizontalSpacing(10)
        setting_layout.setVerticalSpacing(10)

        self.task_combo = QComboBox()
        self.task_combo.setMinimumHeight(40)
        for task in TASK_OPTIONS:
            self.task_combo.addItem(task["title"], task)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(40)

        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("请选择模型权重 .pth 文件")
        self.weights_edit.setMinimumHeight(40)
        self.browse_weights_button = QPushButton("浏览权重")
        self.browse_weights_button.setObjectName("BrowseWeightsButton")
        self.browse_weights_button.setFixedHeight(40)
        self.browse_weights_button.clicked.connect(self.browse_weights)

        self.device_label = QLabel(self.device.type)
        self.device_label.setObjectName("DeviceCudaLabel" if self.device.type == "cuda" else "DeviceCpuLabel")
        self.device_label.setAlignment(Qt.AlignCenter)
        self.device_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.face_location_combo = QComboBox()
        self.face_location_combo.setObjectName("FaceLocationComboBox")
        self.face_location_combo.setMinimumHeight(40)
        self.face_location_combo.addItem("YuNet 人脸检测（推荐）", "yunet")
        self.face_location_combo.addItem("MediaPipe 人脸检测", "mediapipe")
        self.face_location_combo.addItem("中心区域识别", "center")

        task_label = QLabel("识别任务类型：")
        task_label.setObjectName("FormLabel")
        task_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_label = QLabel("具体模型：")
        model_label.setObjectName("FormLabel")
        model_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        weights_label = QLabel("权重路径：")
        weights_label.setObjectName("FormLabel")
        weights_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.face_location_label = QLabel("人脸定位方式：")
        self.face_location_label.setObjectName("FormLabel")
        self.face_location_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        device_title_label = QLabel("当前设备：")
        device_title_label.setObjectName("FormLabel")
        device_title_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        setting_layout.addWidget(task_label, 0, 0)
        setting_layout.addWidget(self.task_combo, 0, 1, 1, 2)
        setting_layout.addWidget(model_label, 1, 0)
        setting_layout.addWidget(self.model_combo, 1, 1, 1, 2)
        setting_layout.addWidget(weights_label, 2, 0)
        setting_layout.addWidget(self.weights_edit, 2, 1)
        setting_layout.addWidget(self.browse_weights_button, 2, 2)
        setting_layout.addWidget(self.face_location_label, 3, 0)
        setting_layout.addWidget(self.face_location_combo, 3, 1, 1, 2)
        setting_layout.addWidget(device_title_label, 4, 0)
        setting_layout.addWidget(self.device_label, 4, 1, 1, 2, Qt.AlignLeft)
        setting_layout.setColumnStretch(1, 1)
        layout.addWidget(setting_group)

        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.face_location_combo.currentIndexChanged.connect(self._on_face_location_changed)

        info_group = QGroupBox("模型说明")
        info_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_layout = QVBoxLayout(info_group)
        self.model_info_text = QTextEdit()
        self.model_info_text.setObjectName("ModelInfoText")
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.model_info_text.setFixedHeight(125)
        info_layout.addWidget(self.model_info_text)
        layout.addWidget(info_group)

        result_group = QGroupBox("预测结果")
        result_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(6)
        self.result_label = QLabel("预测结果：等待识别")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setWordWrap(True)
        self.confidence_label = QLabel("置信度：--")
        self.confidence_label.setObjectName("ConfidenceLabel")
        self.confidence_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        layout.addWidget(result_group)

        self.prob_group = QGroupBox("七类基本表情概率")
        prob_group_layout = QVBoxLayout(self.prob_group)
        prob_group_layout.setSpacing(8)
        self.probability_rows: list[QWidget] = []
        self.probability_name_labels: list[QLabel] = []
        self.probability_bars: list[QProgressBar] = []
        self.probability_value_labels: list[QLabel] = []
        self.probability_scroll_area = QScrollArea()
        self.probability_scroll_area.setObjectName("ProbabilityScrollArea")
        self.probability_scroll_area.setWidgetResizable(True)
        self.probability_scroll_area.setFrameShape(QFrame.NoFrame)
        self.probability_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.probability_scroll_area.setMinimumHeight(220)
        self.probability_scroll_area.setMaximumHeight(300)
        self.probability_content = QWidget()
        self.probability_content.setObjectName("ProbabilityContent")
        self.probability_layout = QVBoxLayout(self.probability_content)
        self.probability_layout.setContentsMargins(0, 0, 0, 0)
        self.probability_layout.setSpacing(4)
        self.probability_scroll_area.setWidget(self.probability_content)
        prob_group_layout.addWidget(self.probability_scroll_area)
        layout.addWidget(self.prob_group)
        layout.addStretch(1)

        scroll_area.setWidget(panel)
        return scroll_area

    def _apply_style(self) -> None:
        """设置界面 QSS 样式。"""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #F8FAFC;
                color: #111827;
                font-family: "Microsoft YaHei", "Segoe UI", Arial;
                font-size: 14px;
            }
            #TitleLabel {
                background: transparent;
                font-size: 22px;
                font-weight: 700;
                color: #0B2E5F;
                padding: 0;
            }
            #SubtitleLabel {
                background: transparent;
                font-size: 12px;
                font-weight: 400;
                color: #6B7280;
                padding: 0;
            }
            QSplitter::handle {
                background: #E5E7EB;
                margin: 6px 4px;
                border-radius: 3px;
            }
            #RightScrollArea {
                background: transparent;
                border: none;
            }
            #ImageLabel {
                background: #F8FBFF;
                border: 2px dashed #B9CBE2;
                border-radius: 10px;
                color: #7A8797;
                font-size: 16px;
                font-weight: 600;
            }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 700;
                color: #111827;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1F2937;
                background: #FFFFFF;
            }
            QPushButton {
                background: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 9px 16px;
                font-weight: 700;
                min-height: 22px;
            }
            QPushButton:hover {
                background: #1D4ED8;
            }
            QPushButton:pressed {
                background: #1E40AF;
            }
            QPushButton:disabled {
                background: #D1D5DB;
                color: #F9FAFB;
            }
            #BrowseWeightsButton {
                background: #E0ECFF;
                color: #1D4ED8;
                padding-left: 14px;
                padding-right: 14px;
            }
            #BrowseWeightsButton:hover {
                background: #C7DBFF;
            }
            #BrowseWeightsButton:pressed {
                background: #B4CCFF;
            }
            #StopCameraButton {
                background: #F97316;
            }
            #StopCameraButton:hover {
                background: #EA580C;
            }
            #StopCameraButton:pressed {
                background: #C2410C;
            }
            #StopCameraButton:disabled {
                background: #D1D5DB;
            }
            QLineEdit, QComboBox {
                background: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                padding: 7px 10px;
                color: #111827;
                selection-background-color: #BFDBFE;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #2563EB;
                background: #FFFFFF;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            #FormLabel {
                background: transparent;
                color: #4B5563;
                font-weight: 700;
                font-size: 13px;
            }
            #ModeStatusLabel {
                background: #EFF6FF;
                color: #1D4ED8;
                border: 1px solid #BFDBFE;
                border-radius: 8px;
                padding: 5px 10px;
                font-weight: 700;
                font-size: 12px;
            }
            #ModelInfoText {
                background: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                padding: 8px;
                color: #6B7280;
                font-size: 12px;
                font-weight: 400;
                selection-background-color: #BFDBFE;
            }
            #DeviceCudaLabel {
                background: #ECFDF5;
                color: #047857;
                border: 1px solid #A7F3D0;
                border-radius: 10px;
                padding: 4px 10px;
                font-weight: 700;
                font-size: 12px;
            }
            #DeviceCpuLabel {
                background: #F3F4F6;
                color: #4B5563;
                border: 1px solid #D1D5DB;
                border-radius: 10px;
                padding: 4px 10px;
                font-weight: 700;
                font-size: 12px;
            }
            #ResultLabel {
                background: transparent;
                font-size: 20px;
                font-weight: 800;
                color: #2563EB;
                padding: 2px 0 1px 0;
            }
            #ConfidenceLabel {
                background: transparent;
                font-size: 14px;
                font-weight: 700;
                color: #4B5563;
                padding-bottom: 0;
            }
            #ProbabilityScrollArea, #ProbabilityContent {
                background: transparent;
                border: none;
            }
            #ProbabilityLabel {
                background: transparent;
                font-weight: 700;
                color: #374151;
                font-size: 12px;
            }
            #ProbabilityValueLabel {
                background: transparent;
                font-weight: 800;
                color: #059669;
                font-size: 12px;
            }
            #ProbabilityRow {
                background: transparent;
                border: none;
                border-bottom: 1px solid #EEF2F7;
                border-radius: 0;
            }
            #ProbabilityRow[active="true"] {
                background: #EFF6FF;
                border: none;
                border-bottom: 1px solid #BFDBFE;
            }
            #ProbabilityLabel[active="true"],
            #ProbabilityValueLabel[active="true"] {
                color: #1D4ED8;
                font-weight: 800;
            }
            QProgressBar {
                min-height: 15px;
                max-height: 15px;
                border: 1px solid #D1D5DB;
                border-radius: 5px;
                background: #F3F4F6;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #60A5FA;
            }
            QProgressBar[active="true"]::chunk {
                background: #2563EB;
            }
            QScrollBar:vertical {
                background: #F3F4F6;
                width: 10px;
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #D1D5DB;
                min-height: 36px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9CA3AF;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

    def _on_task_changed(self, index: int) -> None:
        """切换识别任务类型后，刷新该任务下可用的具体模型。"""
        task = self.task_combo.itemData(index)
        if not task:
            QMessageBox.warning(self, "任务类型错误", "当前任务类型未支持，请重新选择。")
            return
        self._populate_model_combo(task["task_type"])

    def _populate_model_combo(self, task_type: str) -> None:
        """根据任务类型重建具体模型下拉框。"""
        matched_options = [option for option in MODEL_OPTIONS if option.get("task_type") == task_type]
        if not matched_options:
            QMessageBox.warning(self, "任务类型错误", "当前任务类型没有可用模型，请检查 GUI 配置。")
            return

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for option in matched_options:
            self.model_combo.addItem(option["title"], option)
        self.model_combo.blockSignals(False)
        self.model_combo.setCurrentIndex(0)
        self._on_model_changed(0)

    def _on_model_changed(self, index: int) -> None:
        """具体模型变化时自动切换权重路径、说明文字和概率区域。"""
        option = self.model_combo.itemData(index)
        if not option:
            return
        try:
            self._validate_model_option(option)
        except ValueError as exc:
            QMessageBox.warning(self, "模型配置错误", str(exc))
            return

        self.weights_edit.setText(option["weights"])
        self.weights_edit.setToolTip(option["weights"])
        self.model_info_text.setPlainText(self._build_model_info_text(option))
        self._refresh_probability_panel(option)
        self.result_label.setText(self._default_result_text(option))
        self.confidence_label.setText("置信度：--")
        self.cached_model = None
        self.cached_model_key = None
        self.last_camera_prediction = None
        self.last_camera_status = None

    def _current_model_option(self) -> dict:
        """返回当前模型配置，避免下拉框为空时出现异常。"""
        current = self.model_combo.currentData()
        if current:
            return current
        task = self._current_task_option()
        for option in MODEL_OPTIONS:
            if option.get("task_type") == task.get("task_type"):
                return option
        return MODEL_OPTIONS[0]

    def _current_task_option(self) -> dict:
        """返回当前任务配置，避免任务下拉框为空时出现异常。"""
        return self.task_combo.currentData() or TASK_OPTIONS[0]

    def _validate_model_option(self, option: dict) -> None:
        """检查任务类型、类别数和输出方式是否匹配，避免误用权重。"""
        task_type = option.get("task_type")
        output_type = option.get("output_type")
        num_classes = int(option.get("num_classes", 0))
        label_count = len(option.get("labels_cn", []))

        if task_type not in {"basic_7", "compound_11", "rafml_6"}:
            raise ValueError("当前任务类型未支持，请检查 GUI 模型配置。")
        if task_type == "basic_7" and (output_type != "softmax" or num_classes != 7 or label_count != 7):
            raise ValueError("基本表情识别必须使用 7 类 softmax 标签。")
        if task_type == "compound_11" and (output_type != "softmax" or num_classes != 11 or label_count != 11):
            raise ValueError("复合表情模型必须使用 11 类 softmax 标签，不能按七类模型加载。")
        if task_type == "rafml_6" and (output_type != "sigmoid" or num_classes != 6 or label_count != 6):
            raise ValueError("RAF-ML 多标签模型必须使用 6 维 sigmoid 输出，不能误用 softmax。")

    def _build_model_info_text(self, option: dict) -> str:
        """合并当前任务说明和当前模型说明，帮助用户理解模型用途。"""
        task = self._current_task_option()
        return (
            f"当前任务：{task['title']}\n"
            f"输出方式：{task['output_mode']}\n"
            f"输出类别：{task['labels_text']}\n"
            f"任务说明：{task['description']}\n\n"
            f"当前模型：{option['title']}\n"
            f"模型类型：{option['model_name']}\n"
            f"类别数量：{option['num_classes']}\n"
            f"默认权重：{option['weights']}\n"
            f"推荐说明：{option.get('recommendation', '用于当前任务的表情识别演示。')}\n\n"
            f"{option['description']}"
        )

    def _missing_weight_message(self, option: dict) -> str:
        """返回当前模型的权重缺失提示，特殊模型可定制更明确的中文说明。"""
        return option.get("missing_weight_message", "模型权重文件不存在，请检查路径。")

    def _default_result_text(self, option: dict | None = None) -> str:
        """根据任务类型生成预测结果区域的默认文字。"""
        option = option or self._current_model_option()
        task_type = option.get("task_type", "basic_7")
        if task_type == "compound_11":
            return "复合表情结果：等待识别"
        if task_type == "rafml_6":
            return "混合表情结果：等待识别"
        return "预测结果：等待识别"

    def _refresh_probability_panel(self, option: dict | None = None) -> None:
        """根据当前任务动态刷新概率条标题和类别行。"""
        option = option or self._current_model_option()
        self.prob_group.setTitle(option.get("probability_title", "表情概率"))
        self._rebuild_probability_bars(option["labels_cn"], option["labels_en"])
        self._reset_probability_bars()

    def _rebuild_probability_bars(self, labels_cn: list[str], labels_en: list[str]) -> None:
        """按当前任务标签动态生成概率条，支持 7 类、11 类和 6 维输出。"""
        while self.probability_layout.count():
            item = self.probability_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.probability_rows = []
        self.probability_name_labels = []
        self.probability_bars = []
        self.probability_value_labels = []

        for label_cn, label_en in zip(labels_cn, labels_en):
            row_widget = QWidget()
            row_widget.setObjectName("ProbabilityRow")
            row_widget.setProperty("active", False)
            row_widget.setFixedHeight(42)
            row = QVBoxLayout(row_widget)
            row.setContentsMargins(6, 2, 6, 3)
            row.setSpacing(2)

            header_layout = QHBoxLayout()
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(8)
            label = QLabel(f"{label_cn} {label_en}")
            label.setObjectName("ProbabilityLabel")
            label.setProperty("active", False)
            label.setWordWrap(False)
            label.setToolTip(f"{label_cn} {label_en}")
            value_label = QLabel("0.00%")
            value_label.setObjectName("ProbabilityValueLabel")
            value_label.setProperty("active", False)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setFixedWidth(68)

            bar = QProgressBar()
            bar.setProperty("active", False)
            bar.setRange(0, 10000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(15)

            header_layout.addWidget(label, 1)
            header_layout.addWidget(value_label)
            row.addLayout(header_layout)
            row.addWidget(bar)
            self.probability_layout.addWidget(row_widget)
            self.probability_rows.append(row_widget)
            self.probability_name_labels.append(label)
            self.probability_bars.append(bar)
            self.probability_value_labels.append(value_label)

        self.probability_layout.addStretch(1)

    def _on_mode_changed(self, index: int) -> None:
        """切换图片上传识别或摄像头实时识别模式。"""
        mode = self.mode_combo.itemData(index)
        if mode == "image" and self.camera_running:
            self.stop_camera(switch_to_image=False)
        self.recognition_mode = mode or "image"
        if self.recognition_mode == "camera" and not self.camera_running:
            self.result_label.setText(self._default_result_text())
            self.confidence_label.setText("置信度：--")
        elif self.recognition_mode == "image" and not self._has_predictable_image():
            self.result_label.setText(self._default_result_text())
            self.confidence_label.setText("置信度：--")
        self._apply_mode_state()

    def _on_face_location_changed(self, index: int) -> None:
        """切换摄像头人脸定位方式。"""
        if self.recognition_mode != "camera":
            return
        self.face_location_mode = self.face_location_combo.itemData(index) or "yunet"
        if self.face_location_mode != "yunet":
            self.yunet_detector = None
            self.yunet_input_size = None
        if self.face_location_mode != "mediapipe":
            self._close_mediapipe_detector()
        self.last_mediapipe_face_box = None
        self.last_mediapipe_detect_frame = -1
        self.last_camera_prediction = None
        self.last_camera_status = None
        if self.recognition_mode == "camera":
            if self.face_location_mode == "mediapipe" and self.camera_running:
                self._load_mediapipe_face_detector()
            if self.camera_running:
                self.result_label.setText("预测结果：摄像头识别中...")
            else:
                self.result_label.setText(self._default_result_text())
            self.confidence_label.setText("置信度：--")

    def _set_mode_combo(self, mode: str) -> None:
        """同步模式下拉框，不重复触发切换逻辑。"""
        target_index = 0 if mode == "image" else 1
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(target_index)
        self.mode_combo.blockSignals(False)
        self.recognition_mode = mode

    def _apply_mode_state(self) -> None:
        """根据当前识别模式更新按钮可见性和可用性。"""
        is_image_mode = self.recognition_mode == "image"
        self.image_button_widget.setVisible(is_image_mode)
        self.camera_button_widget.setVisible(not is_image_mode)
        self._update_face_location_combo_state()
        self.mode_status_label.setText(
            "当前模式：图片上传识别" if is_image_mode else "当前模式：摄像头实时识别"
        )
        self.preview_group.setTitle("图片预览" if is_image_mode else "摄像头实时画面")
        if self.current_pixmap is None and self.current_camera_pixmap is None:
            self.image_label.setText("请加载图片或开启摄像头" if is_image_mode else "摄像头实时画面")

        self.load_image_button.setEnabled(is_image_mode and not self.camera_running)
        self.predict_button.setEnabled(is_image_mode and not self.camera_running and self._has_predictable_image())
        self.start_camera_button.setEnabled((not is_image_mode) and not self.camera_running)
        self.stop_camera_button.setEnabled((not is_image_mode) and self.camera_running)

    def _update_face_location_combo_state(self) -> None:
        """根据识别模式更新人脸定位方式下拉框的内容和提示。"""
        is_camera_mode = self.recognition_mode == "camera"
        self.face_location_combo.blockSignals(True)
        if is_camera_mode:
            if self.face_location_combo.count() != 3 or self.face_location_combo.itemData(0) == "unavailable":
                self.face_location_combo.clear()
                self.face_location_combo.addItem("YuNet 人脸检测（推荐）", "yunet")
                self.face_location_combo.addItem("MediaPipe 人脸检测", "mediapipe")
                self.face_location_combo.addItem("中心区域识别", "center")
                self.face_location_combo.setCurrentIndex(0)
                self.face_location_mode = "yunet"
            self.face_location_combo.setEnabled(True)
            face_location_tip = "选择摄像头实时识别时使用的人脸定位方式。"
        else:
            if self.face_location_combo.count() != 1 or self.face_location_combo.itemData(0) != "unavailable":
                self.face_location_combo.clear()
                self.face_location_combo.addItem("当前模式不可用", "unavailable")
                self.face_location_combo.setCurrentIndex(0)
            self.face_location_combo.setEnabled(False)
            face_location_tip = "人脸定位方式仅用于摄像头实时识别，图片上传识别不使用该选项。"
        self.face_location_combo.blockSignals(False)
        self.face_location_combo.setToolTip(face_location_tip)
        self.face_location_label.setToolTip(face_location_tip)

    def _has_predictable_image(self) -> bool:
        """判断当前是否有图片文件或摄像头最后一帧可用于预测。"""
        has_image_file = self.current_image_source == "image_file" and bool(self.current_image_path)
        has_camera_frame = self.current_image_source == "camera_frame" and self.last_camera_frame_rgb is not None
        return has_image_file or has_camera_frame

    def load_image(self) -> None:
        """打开文件对话框并加载图片。"""
        if self.camera_running:
            self.stop_camera()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择人脸图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return
        self.current_image_path = file_path
        self.current_image_source = "image_file"
        self.current_pixmap = QPixmap(file_path)
        if self.current_pixmap.isNull():
            QMessageBox.warning(self, "图片加载失败", "无法加载该图片，请选择有效的图片文件。")
            self.current_image_path = None
            self.current_image_source = None
            self.current_pixmap = None
            return
        self.preview_mode = "image"
        self.current_camera_pixmap = None
        self._update_image_preview()
        self.result_label.setText(self._default_result_text())
        self.confidence_label.setText("置信度：--")
        self._reset_probability_bars()
        self._apply_mode_state()

    def browse_weights(self) -> None:
        """选择模型权重文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型权重",
            "checkpoints",
            "PyTorch 权重文件 (*.pth *.pt);;所有文件 (*)",
        )
        if file_path:
            self.weights_edit.setText(file_path)
            self.weights_edit.setToolTip(file_path)
            self.cached_model = None
            self.cached_model_key = None

    def predict_current_image(self) -> None:
        """对当前图片进行表情预测。"""
        if self.camera_running:
            QMessageBox.information(self, "提示", "请先停止摄像头，再进行图片预测。")
            return
        image_source = self._get_current_prediction_source()
        if image_source is None:
            QMessageBox.information(self, "提示", "请先加载图片或开启摄像头保留最后一帧。")
            return

        option = self._current_model_option()
        model_name = option["model_name"]
        weights_path = self.weights_edit.text().strip()
        if not weights_path:
            QMessageBox.warning(self, "提示", "请先选择模型权重文件。")
            return
        if not Path(weights_path).exists():
            QMessageBox.warning(self, "模型权重不存在", self._missing_weight_message(option))
            return

        try:
            self._validate_model_option(option)
            model = self._get_or_load_model(model_name, weights_path, option["num_classes"])
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", f"模型加载失败：\n{exc}")
            return

        try:
            result = predict_image(
                image_source,
                model,
                model_name,
                option["task_type"],
                option["output_type"],
                option["labels_cn"],
                option["labels_en"],
                self.device,
                threshold=option.get("threshold", 0.5),
            )
        except Exception as exc:
            QMessageBox.critical(self, "预测失败", f"预测失败：\n{exc}")
            return

        self._update_prediction_result(result)
        self._apply_mode_state()

    def _get_current_prediction_source(self):
        """返回预测输入；图片用原始文件，摄像头用原始帧中的识别区域。"""
        if self.current_image_source == "image_file" and self.current_image_path:
            # 图片上传识别始终使用原始文件路径，不使用 QLabel 中缩放后的显示图。
            return self.current_image_path
        if self.current_image_source == "camera_frame" and self.last_camera_frame_rgb is not None:
            # 停止摄像头后的单张预测使用原始摄像头帧，优先 YuNet 人脸框，失败后回退中心区域。
            return self._get_camera_prediction_region(self.last_camera_frame_rgb)
        return None

    def start_camera(self) -> None:
        """打开默认摄像头并启动实时识别。"""
        if self.camera_running:
            return
        if self.recognition_mode != "camera":
            self._set_mode_combo("camera")
        self._apply_mode_state()
        if cv2 is None:
            QMessageBox.critical(self, "缺少 OpenCV", "缺少 OpenCV，请先运行：pip install opencv-python")
            return

        option = self._current_model_option()
        model_name = option["model_name"]
        weights_path = self.weights_edit.text().strip()
        if not weights_path:
            QMessageBox.warning(self, "提示", "请先选择模型权重文件。")
            return
        if not Path(weights_path).exists():
            QMessageBox.warning(self, "模型权重不存在", self._missing_weight_message(option))
            return

        try:
            self._validate_model_option(option)
            self._get_or_load_model(model_name, weights_path, option["num_classes"])
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", f"模型加载失败：\n{exc}")
            return

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            QMessageBox.critical(
                self,
                "摄像头打开失败",
                "无法打开默认摄像头，请检查摄像头是否被占用、系统权限是否开启。",
            )
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if self.face_location_mode == "mediapipe":
            # 在启动定时器前预先初始化 MediaPipe，避免第一帧检测时才加载造成卡顿。
            self._load_mediapipe_face_detector()

        self.camera_capture = capture
        self.camera_running = True
        self.camera_frame_count = 0
        self.last_camera_prediction = None
        self.last_camera_status = None
        self.last_camera_frame_rgb = None
        self.last_mediapipe_face_box = None
        self.last_mediapipe_detect_frame = -1
        self.current_image_path = None
        self.current_image_source = None
        self.current_pixmap = None
        self.preview_mode = "camera"
        self.result_label.setText(self._default_result_text())
        self.confidence_label.setText("置信度：--")
        self._reset_probability_bars()
        self._set_camera_controls_state(True)
        self.camera_timer.start()

    def stop_camera(self, switch_to_image: bool = True) -> None:
        """停止摄像头并释放资源。"""
        self._release_camera()
        if self.last_camera_frame_rgb is not None:
            self.current_image_source = "camera_frame"
            self.preview_mode = "camera"
            self.result_label.setText("预测结果：摄像头已关闭，可使用最后一帧继续预测")
        else:
            self.current_image_source = None
            self.result_label.setText("预测结果：摄像头已关闭")
        self.confidence_label.setText("置信度：--")
        if switch_to_image:
            self._set_mode_combo("image")
        self._apply_mode_state()

    def _release_camera(self) -> None:
        """释放摄像头和定时器资源。"""
        if self.camera_timer.isActive():
            self.camera_timer.stop()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.yunet_detector = None
        self.yunet_input_size = None
        self._close_mediapipe_detector()
        self.last_mediapipe_face_box = None
        self.last_mediapipe_detect_frame = -1
        self.camera_running = False

    def _set_camera_controls_state(self, running: bool) -> None:
        """根据摄像头状态切换按钮可用性。"""
        self.camera_running = running
        self._apply_mode_state()

    def _print_yunet_warning_once(self, message: str) -> None:
        """YuNet 异常只在终端提示一次，避免摄像头刷新时刷屏。"""
        if message not in self.yunet_warning_messages:
            print(message)
            self.yunet_warning_messages.add(message)

    def _print_mediapipe_warning_once(self, message: str) -> None:
        """MediaPipe 异常只在终端提示一次，避免摄像头刷新时刷屏。"""
        if message not in self.mediapipe_warning_messages:
            print(message)
            self.mediapipe_warning_messages.add(message)

    def _close_mediapipe_detector(self) -> None:
        """释放 MediaPipe Tasks 人脸检测器。"""
        if self.mediapipe_face_detector is None:
            return
        try:
            close_method = getattr(self.mediapipe_face_detector, "close", None)
            if callable(close_method):
                close_method()
        except Exception as exc:
            self._print_mediapipe_warning_once(f"MediaPipe Face Detector 释放失败，已忽略：{exc}")
        finally:
            self.mediapipe_face_detector = None

    def _disable_yunet_detection(self, message: str) -> None:
        """禁用 YuNet 检测并回退到中心区域识别。"""
        self._print_yunet_warning_once(message)
        self.face_location_mode = "center"
        self.yunet_detector = None
        self.yunet_input_size = None
        if hasattr(self, "face_location_combo"):
            self.face_location_combo.blockSignals(True)
            self.face_location_combo.setCurrentIndex(2)
            self.face_location_combo.blockSignals(False)

    def _disable_mediapipe_detection(self, message: str) -> None:
        """禁用 MediaPipe 检测并回退到中心区域识别。"""
        self._print_mediapipe_warning_once(message)
        self.face_location_mode = "center"
        self._close_mediapipe_detector()
        self.last_mediapipe_face_box = None
        self.last_mediapipe_detect_frame = -1
        if hasattr(self, "face_location_combo"):
            self.face_location_combo.blockSignals(True)
            self.face_location_combo.setCurrentIndex(2)
            self.face_location_combo.blockSignals(False)

    def _resolve_yunet_model_path(self) -> Path:
        """解析 YuNet 模型路径，兼容从项目根目录或脚本路径启动。"""
        model_path = Path(self.yunet_model_path)
        if model_path.is_absolute():
            return model_path
        return Path(__file__).resolve().parent / model_path

    def _resolve_mediapipe_model_path(self) -> Path:
        """解析 MediaPipe Tasks 模型路径，兼容从项目根目录或脚本路径启动。"""
        model_path = Path(self.mediapipe_model_path)
        if model_path.is_absolute():
            return model_path
        return Path(__file__).resolve().parent / model_path

    def _load_yunet_detector(self, frame_width: int, frame_height: int) -> bool:
        """按需加载 YuNet 人脸检测器，并同步当前摄像头帧尺寸。"""
        if self.face_location_mode != "yunet":
            return False
        if cv2 is None:
            return False
        if not hasattr(cv2, "FaceDetectorYN"):
            self._disable_yunet_detection("当前 OpenCV 版本不支持 FaceDetectorYN，已自动切换为中心区域识别。")
            return False

        model_path = self._resolve_yunet_model_path()
        if not model_path.exists():
            self._disable_yunet_detection("未找到 YuNet 人脸检测模型，已自动切换为中心区域识别。")
            return False

        input_size = (int(frame_width), int(frame_height))
        try:
            if self.yunet_detector is None:
                initial_size = (320, 320)
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=str(model_path),
                    config="",
                    input_size=initial_size,
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self.yunet_input_size = initial_size
            if self.yunet_input_size != input_size:
                self.yunet_detector.setInputSize(input_size)
                self.yunet_input_size = input_size
            return True
        except Exception as exc:
            self._disable_yunet_detection(f"YuNet 人脸检测器初始化失败，已自动切换为中心区域识别：{exc}")
            return False

    def _load_mediapipe_face_detector(self) -> bool:
        """按需加载新版 MediaPipe Tasks API 人脸检测器。"""
        if self.face_location_mode != "mediapipe":
            return False
        if mp is None or mp_tasks_python is None or mp_vision is None:
            self._disable_mediapipe_detection("mediapipe 未安装或 MediaPipe Tasks API 不可用，已自动回退到中心区域识别。")
            return False
        if not hasattr(mp, "Image") or not hasattr(mp, "ImageFormat"):
            self._disable_mediapipe_detection("当前 mediapipe 版本缺少 Tasks 图像接口，已自动回退到中心区域识别。")
            return False
        if not all(
            hasattr(mp_vision, attr)
            for attr in ("FaceDetector", "FaceDetectorOptions", "RunningMode")
        ) or not hasattr(mp_tasks_python, "BaseOptions"):
            self._disable_mediapipe_detection("当前 mediapipe 版本不支持新版 Face Detector Tasks API，已自动回退到中心区域识别。")
            return False

        model_path = self._resolve_mediapipe_model_path()
        if not model_path.exists():
            self._disable_mediapipe_detection("未找到 MediaPipe Face Detector 模型文件，已自动回退到中心区域识别。")
            return False

        if self.mediapipe_face_detector is not None:
            return True

        try:
            base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            self.mediapipe_face_detector = mp_vision.FaceDetector.create_from_options(options)
            return True
        except Exception as exc:
            self._disable_mediapipe_detection(f"MediaPipe Face Detector 初始化失败，已自动回退到中心区域识别：{exc}")
            return False

    def _update_camera_frame(self) -> None:
        """读取摄像头帧，按当前定位方式查找人脸，失败后回退中心区域预测。"""
        if cv2 is None or self.camera_capture is None:
            self._release_camera()
            return

        success, frame_bgr = self.camera_capture.read()
        if not success or frame_bgr is None:
            self._release_camera()
            self._set_camera_controls_state(False)
            self.result_label.setText("预测结果：摄像头帧读取失败")
            self.confidence_label.setText("置信度：--")
            QMessageBox.warning(self, "摄像头读取失败", "摄像头帧读取失败，已停止摄像头。")
            return

        self.camera_frame_count += 1
        # 摄像头预测始终基于 VideoCapture 读取到的原始帧，不使用 QLabel 缩放后的显示尺寸。
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.last_camera_frame_rgb = frame_rgb.copy()
        display_rgb = frame_rgb.copy()
        center_box = self._get_center_region_box(frame_rgb.shape[1], frame_rgb.shape[0])
        predict_box = center_box
        box_source = "center"

        if self.face_location_mode == "yunet":
            try:
                face_box = self._detect_largest_face_yunet(frame_bgr)
                if face_box is not None:
                    predict_box = face_box
                    box_source = "yunet_face"
            except Exception as exc:
                self._disable_yunet_detection(f"YuNet 人脸检测异常，已自动切换为中心区域识别：{exc}")
        elif self.face_location_mode == "mediapipe":
            try:
                should_detect_mediapipe = (
                    self.last_mediapipe_detect_frame < 0
                    or self.camera_frame_count - self.last_mediapipe_detect_frame >= self.mediapipe_detect_interval
                )
                if should_detect_mediapipe:
                    self.last_mediapipe_face_box = self._detect_largest_face_mediapipe(frame_rgb)
                    self.last_mediapipe_detect_frame = self.camera_frame_count
                face_box = self.last_mediapipe_face_box
                if face_box is not None:
                    predict_box = face_box
                    box_source = "mediapipe_face"
            except Exception as exc:
                self._disable_mediapipe_detection(f"MediaPipe 人脸检测异常，已自动回退到中心区域识别：{exc}")

        should_predict = (
            self.last_camera_prediction is None
            or self.last_camera_status != box_source
            or self.camera_frame_count % self.camera_predict_interval == 0
        )
        if should_predict:
            try:
                self.last_camera_prediction = self._predict_camera_face(
                    frame_rgb,
                    predict_box,
                    expand=box_source in ("yunet_face", "mediapipe_face"),
                )
                self._update_prediction_result(self.last_camera_prediction)
                self.last_camera_status = box_source
            except Exception as exc:
                self._release_camera()
                self._set_camera_controls_state(False)
                self.result_label.setText("预测结果：摄像头预测失败")
                self.confidence_label.setText("置信度：--")
                QMessageBox.critical(self, "摄像头预测失败", f"摄像头预测失败：\n{exc}")
                return

        self._draw_camera_result(
            display_rgb,
            predict_box,
            self.last_camera_prediction,
            is_center_box=box_source == "center",
        )
        self._display_camera_frame(display_rgb)

    def _crop_frame_to_preview_ratio(self, frame_bgr):
        """将摄像头帧裁剪为接近 4:3，减少横向空白区域。"""
        height, width = frame_bgr.shape[:2]
        target_ratio = 4 / 3
        current_ratio = width / height
        if abs(current_ratio - target_ratio) < 0.03:
            return frame_bgr
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            x1 = max(0, (width - new_width) // 2)
            return frame_bgr[:, x1 : x1 + new_width]
        new_height = int(width / target_ratio)
        y1 = max(0, (height - new_height) // 2)
        return frame_bgr[y1 : y1 + new_height, :]

    def _detect_largest_face_yunet(self, frame_bgr):
        """使用 OpenCV YuNet 检测最大人脸区域。"""
        frame_height, frame_width = frame_bgr.shape[:2]
        if not self._load_yunet_detector(frame_width, frame_height):
            return None

        try:
            detect_result = self.yunet_detector.detect(frame_bgr)
        except Exception as exc:
            self._disable_yunet_detection(f"YuNet 人脸检测运行失败，已自动切换为中心区域识别：{exc}")
            return None

        faces = detect_result[1] if isinstance(detect_result, tuple) else detect_result
        if faces is None or len(faces) == 0:
            return None

        face_boxes: list[tuple[int, int, int, int]] = []
        for face in faces:
            if len(face) < 4:
                continue
            x, y, w, h = face[:4]
            if w <= 0 or h <= 0:
                continue
            if len(face) >= 15 and face[14] < 0.5:
                continue

            x1 = max(0, min(frame_width - 1, int(round(x))))
            y1 = max(0, min(frame_height - 1, int(round(y))))
            x2 = max(0, min(frame_width, int(round(x + w))))
            y2 = max(0, min(frame_height, int(round(y + h))))
            if x2 > x1 and y2 > y1:
                face_boxes.append((x1, y1, x2 - x1, y2 - y1))

        if not face_boxes:
            return None
        return max(face_boxes, key=lambda face: face[2] * face[3])

    def _detect_largest_face_mediapipe(self, frame_rgb):
        """使用新版 MediaPipe Tasks API 检测最大人脸区域。"""
        frame_height, frame_width = frame_rgb.shape[:2]
        if not self._load_mediapipe_face_detector():
            return None

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb.copy())
            result = self.mediapipe_face_detector.detect(mp_image)
        except Exception as exc:
            self._disable_mediapipe_detection(f"MediaPipe 单帧人脸检测失败，已自动回退到中心区域识别：{exc}")
            return None

        detections = getattr(result, "detections", None) or []
        if not detections:
            return None

        face_boxes: list[tuple[int, int, int, int]] = []
        for detection in detections:
            bounding_box = getattr(detection, "bounding_box", None)
            if bounding_box is None:
                continue
            x = int(round(getattr(bounding_box, "origin_x", 0)))
            y = int(round(getattr(bounding_box, "origin_y", 0)))
            w = int(round(getattr(bounding_box, "width", 0)))
            h = int(round(getattr(bounding_box, "height", 0)))
            if w <= 0 or h <= 0:
                continue

            x1 = max(0, min(frame_width - 1, x))
            y1 = max(0, min(frame_height - 1, y))
            x2 = max(0, min(frame_width, x + w))
            y2 = max(0, min(frame_height, y + h))
            if x2 > x1 and y2 > y1:
                face_boxes.append((x1, y1, x2 - x1, y2 - y1))

        if not face_boxes:
            return None
        return max(face_boxes, key=lambda face: face[2] * face[3])

    def _get_center_region_box(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """取画面中心正方形识别区域。"""
        size = int(min(frame_width, frame_height) * 0.65)
        x = max(0, (frame_width - size) // 2)
        y = max(0, (frame_height - size) // 2)
        return x, y, size, size

    def _crop_rgb_center_region(self, frame_rgb):
        """裁剪 RGB 图像中央识别框区域。"""
        height, width = frame_rgb.shape[:2]
        x, y, w, h = self._get_center_region_box(width, height)
        return frame_rgb[y : y + h, x : x + w]

    def _get_camera_prediction_region(self, frame_rgb):
        """从摄像头原始 RGB 帧中取得单张预测区域。"""
        face_box = None
        if self.face_location_mode == "yunet" and cv2 is not None:
            try:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                face_box = self._detect_largest_face_yunet(frame_bgr)
            except Exception as exc:
                self._disable_yunet_detection(f"YuNet 最后一帧检测异常，已自动切换为中心区域识别：{exc}")
        elif self.face_location_mode == "mediapipe":
            try:
                face_box = self._detect_largest_face_mediapipe(frame_rgb)
            except Exception as exc:
                self._disable_mediapipe_detection(f"MediaPipe 最后一帧检测异常，已自动回退到中心区域识别：{exc}")

        if face_box is not None:
            x, y, w, h = self._expand_face_box(face_box, frame_rgb.shape[1], frame_rgb.shape[0])
            face_rgb = frame_rgb[y : y + h, x : x + w]
            if face_rgb.size != 0:
                return face_rgb
        return self._crop_rgb_center_region(frame_rgb)

    def _predict_camera_face(self, frame_rgb, face_box, expand: bool = True) -> dict:
        """裁剪摄像头识别区域，并复用统一预测函数。"""
        if expand:
            x, y, w, h = self._expand_face_box(face_box, frame_rgb.shape[1], frame_rgb.shape[0])
        else:
            x, y, w, h = [int(value) for value in face_box]
        face_rgb = frame_rgb[y : y + h, x : x + w]
        if face_rgb.size == 0:
            raise ValueError("检测到的人脸区域为空。")

        option = self._current_model_option()
        model_name = option["model_name"]
        weights_path = self.weights_edit.text().strip()
        if not weights_path:
            raise ValueError("请先选择模型权重文件。")
        if not Path(weights_path).exists():
            raise FileNotFoundError("模型权重文件不存在，请检查路径。")

        self._validate_model_option(option)
        model = self._get_or_load_model(model_name, weights_path, option["num_classes"])
        face_image = Image.fromarray(face_rgb)
        return predict_image(
            face_image,
            model,
            model_name,
            option["task_type"],
            option["output_type"],
            option["labels_cn"],
            option["labels_en"],
            self.device,
            threshold=option.get("threshold", 0.5),
        )

    def _expand_face_box(self, face_box, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """适度扩大人脸框，保留更多脸部边缘信息。"""
        x, y, w, h = [int(value) for value in face_box]
        margin_x = int(w * 0.20)
        margin_y = int(h * 0.22)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_width, x + w + margin_x)
        y2 = min(frame_height, y + h + margin_y)
        return x1, y1, x2 - x1, y2 - y1

    def _draw_camera_result(self, frame_rgb, face_box, result: dict | None, is_center_box: bool = True) -> None:
        """在摄像头画面中绘制中心识别框或可选人脸框。"""
        if cv2 is None:
            return
        x, y, w, h = [int(value) for value in face_box]
        color = (47, 111, 237) if is_center_box else (16, 185, 129)
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 3 if is_center_box else 2)
        if is_center_box:
            self._draw_camera_message(frame_rgb, "Put face inside the box", x, max(24, y - 10), color)
        if not result:
            return

        label_text = f"{result['predicted_label_en']} {result['confidence']:.2f}%"
        label_y = y + h + 28 if is_center_box else max(24, y - 10)
        self._draw_camera_message(frame_rgb, label_text, x, label_y, color)

    def _draw_camera_message(self, frame_rgb, label_text: str, x: int = 12, y: int = 32, color=None) -> None:
        """在视频画面中绘制英文状态提示，避免 OpenCV 中文乱码。"""
        if cv2 is None:
            return
        if color is None:
            color = (47, 111, 237)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_x = max(0, x)
        text_y = max(24, y)
        bg_x2 = min(frame_rgb.shape[1], text_x + text_size[0] + 10)
        bg_y1 = max(0, text_y - text_size[1] - baseline - 8)
        cv2.rectangle(frame_rgb, (text_x, bg_y1), (bg_x2, text_y + baseline), color, -1)
        cv2.putText(
            frame_rgb,
            label_text,
            (text_x + 5, text_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _display_camera_frame(self, frame_rgb) -> None:
        """将 RGB 摄像头帧显示到左侧预览区域。"""
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.current_camera_pixmap = QPixmap.fromImage(image)
        self.preview_mode = "camera"
        self._update_image_preview()

    def _get_or_load_model(self, model_name: str, weights_path: str, num_classes: int):
        """按模型类型、类别数和权重路径缓存模型，避免重复加载。"""
        key = (model_name, str(Path(weights_path)), int(num_classes))
        if self.cached_model is None or self.cached_model_key != key:
            self.cached_model = load_emotion_model(
                model_name,
                weights_path,
                self.device,
                num_classes=num_classes,
            )
            self.cached_model_key = key
        return self.cached_model

    def _update_prediction_result(self, result: dict) -> None:
        """刷新预测结果和概率条。"""
        predicted_index = result["predicted_index"]
        output_type = result.get("output_type", "softmax")
        task_type = result.get("task_type", self._current_model_option().get("task_type", "basic_7"))

        if output_type == "sigmoid":
            active_labels = result.get("active_labels", [])
            if active_labels:
                lines = ["混合表情结果："]
                lines.extend(
                    f"{item['label_cn']} {item['label_en']}：{item['probability']:.2f}%"
                    for item in active_labels
                )
                self.result_label.setText("\n".join(lines))
            else:
                self.result_label.setText("混合表情结果：未检测到明显混合表情成分")
            self.confidence_label.setText(
                f"最高候选：{result['predicted_label_cn']} {result['predicted_label_en']} {result['confidence']:.2f}%"
            )
            active_indices = [item["index"] for item in active_labels]
        elif task_type == "compound_11":
            self.result_label.setText(
                f"复合表情结果：{result['predicted_label_cn']} {result['predicted_label_en']}"
            )
            self.confidence_label.setText(f"置信度：{result['confidence']:.2f}%")
            active_indices = [predicted_index]
        else:
            self.result_label.setText(
                f"预测结果：{result['predicted_label_cn']} {result['predicted_label_en']}"
            )
            self.confidence_label.setText(f"置信度：{result['confidence']:.2f}%")
            active_indices = [predicted_index]

        for index, probability in enumerate(result["probability_list"]):
            if index >= len(self.probability_bars):
                continue
            value = int(round(probability * 100))
            self.probability_bars[index].setValue(value)
            self.probability_value_labels[index].setText(f"{probability:.2f}%")
        self._highlight_probability_row(active_indices)

    def _reset_probability_bars(self) -> None:
        """清空当前任务的概率条。"""
        for bar, value_label in zip(self.probability_bars, self.probability_value_labels):
            bar.setValue(0)
            value_label.setText("0.00%")
        self._highlight_probability_row(None)

    def _highlight_probability_row(self, active_indices) -> None:
        """高亮当前任务的重点概率行，softmax 高亮最高类，sigmoid 高亮超过阈值的维度。"""
        if active_indices is None:
            active_set: set[int] = set()
        elif isinstance(active_indices, int):
            active_set = {active_indices}
        else:
            active_set = {int(index) for index in active_indices}
        for index, (row, name_label, bar, value_label) in enumerate(
            zip(
                self.probability_rows,
                self.probability_name_labels,
                self.probability_bars,
                self.probability_value_labels,
            )
        ):
            active = index in active_set
            for widget in (row, name_label, bar, value_label):
                widget.setProperty("active", active)
                self._refresh_widget_style(widget)

    def _refresh_widget_style(self, widget: QWidget) -> None:
        """刷新动态属性对应的 QSS 样式。"""
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _update_image_preview(self) -> None:
        """按显示区域等比例缩放图片。"""
        pixmap = self.current_camera_pixmap if self.preview_mode == "camera" else self.current_pixmap
        if pixmap is None:
            return
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """窗口缩放时同步刷新图片显示。"""
        super().resizeEvent(event)
        self._update_image_preview()

    def closeEvent(self, event) -> None:  # noqa: N802
        """关闭窗口时释放摄像头资源。"""
        self._release_camera()
        super().closeEvent(event)


def main() -> None:
    """启动 PyQt5 GUI。"""
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = EmotionRecognitionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
