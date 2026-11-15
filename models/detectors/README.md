# 可选人脸检测器

仓库不分发第三方检测器二进制文件，可自行从原项目获取：

- YuNet：`face_detection_yunet_2023mar.onnx`，来源为 https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet 。
- MediaPipe：`blaze_face_short_range.tflite`，来源为 https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector 。

将文件放在本目录，使用前查阅原项目提供的许可。缺少检测器时，GUI 可回退到中心区域模式；表情分类模型的权重仍需单独准备。
