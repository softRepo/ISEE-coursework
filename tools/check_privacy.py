"""Check publication content without printing sensitive matched values.

Use --tree before Git initialization, --tracked before publishing, and
--staged in a pre-commit hook. Git modes inspect index blobs, including
files added with git add --force, rather than trusting .gitignore.
Only the exact reviewed reports, presentations, images and audio inputs
are allowed as binary assets.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXACT_PATHS = {
    ".gitignore", ".gitattributes", "README.md", 
    "tools/check_privacy.py", ".githooks/pre-commit", ".github/workflows/privacy.yml",
    "人工智能/README.md", "电子工程训练（甲）/README.md",
}
FACE_PREFIX = "人工智能/人脸表情识别/"
LAB_PREFIX = "人工智能/AILAB/"
SIGNALS_PREFIX = "信号与系统课程设计/"
SIGNALS_TEXT_PATHS = {'README.md',
 '课题ABC_音乐信号综合处理/MATLAB源码/README.md',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/audio_metrics.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/compute_fft_spectrum.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/compute_stft_manual.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/estimate_effective_bandwidth.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/extract_audio_segment.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/find_audio_by_keywords.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/load_audio_segment.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/normalize_audio_peak.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/plot_audio_overview.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/plot_audio_spectrum.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/plot_time_frequency_overview.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/preprocess_audio_basic.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/read_audio_robust.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/safe_sound_play.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/save_figure_safely.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/write_latex_table.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/common/write_table_utf8.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/config_ABC_experiments.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/run_all_ABC.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/topic_A/main_A_music_effect_analysis.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/topic_B/main_B_sampling_reconstruction.m',
 '课题ABC_音乐信号综合处理/MATLAB源码/topic_C/main_C_noise_filter_design.m',
 '课题ABC_音乐信号综合处理/README.md',
 '课题ABC_音乐信号综合处理/results/A_music_effect/tables/A_effect_metrics.csv',
 '课题ABC_音乐信号综合处理/results/B_sampling/tables/B_sampling_metrics.csv',
 '课题ABC_音乐信号综合处理/results/C_denoising/tables/C_filter_metrics.csv',
 '课题D_图像处理与频域分析/MATLAB源码/README.md',
 '课题D_图像处理与频域分析/MATLAB源码/functions/apply_frequency_filter.m',
 '课题D_图像处理与频域分析/MATLAB源码/functions/find_given_image.m',
 '课题D_图像处理与频域分析/MATLAB源码/functions/gaussian_frequency_filters.m',
 '课题D_图像处理与频域分析/MATLAB源码/functions/normalize_for_display.m',
 '课题D_图像处理与频域分析/MATLAB源码/functions/read_and_preprocess_image.m',
 '课题D_图像处理与频域分析/MATLAB源码/functions/save_figure_safely.m',
 '课题D_图像处理与频域分析/MATLAB源码/functions/show_filtering_results.m',
 '课题D_图像处理与频域分析/MATLAB源码/main_topicD_frequency_filter.m',
 '课题D_图像处理与频域分析/README.md',
 '课题D_图像处理与频域分析/results/tables/filter_parameter_record.csv',
 '课题E_孤立数字语音识别/MATLAB源码/README.md',
 '课题E_孤立数字语音识别/MATLAB源码/config_topicE.m',
 '课题E_孤立数字语音识别/MATLAB源码/dataset/README.md',
 '课题E_孤立数字语音识别/MATLAB源码/functions/classify_by_dtw.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/compute_delta_features.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/compute_mfcc_manual.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/dct2_manual.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/dtw_distance_manual.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/endpoint_detect_energy_zcr.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/frame_signal.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/hz2mel_local.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/load_audio_mono.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/mel2hz_local.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/mel_filterbank.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/normalize_audio.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/parse_fsdd_filename.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/plot_confusion_matrix_local.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/plot_dtw_example.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/preprocess_speech.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/save_figure_safely.m',
 '课题E_孤立数字语音识别/MATLAB源码/functions/scan_fsdd_dataset.m',
 '课题E_孤立数字语音识别/MATLAB源码/main_topicE_dtw_speech_recognition.m',
 '课题E_孤立数字语音识别/README.md',
 '课题E_孤立数字语音识别/results/tables/class_accuracy.csv',
 '课题E_孤立数字语音识别/results/tables/confusion_matrix.csv',
 '课题E_孤立数字语音识别/results/tables/prediction_results.csv'}
LAB_TEXT_PATHS = {'AI_Lab1/README.md',
 'AI_Lab1/eight_puzzle.py',
 'AI_Lab1/modified_8_puzzle.py',
 'AI_Lab1/requirements.txt',
 'AI_Lab1/test.py',
 'AI_Lab2/README.md',
 'AI_Lab2/alpha_beta.py',
 'AI_Lab2/compare.py',
 'AI_Lab2/debug.py',
 'AI_Lab2/requirements.txt',
 'AI_Lab3/README.md',
 'AI_Lab3/baseline/bayes.py',
 'AI_Lab3/baseline/lenet.py',
 'AI_Lab3/report/preamble.tex',
 'AI_Lab3/report/report.tex',
 'AI_Lab3/requirements.txt',
 'AI_Lab3/results/bayes_bernoulli_results.csv',
 'AI_Lab3/results/bayes_bernoulli_results.json',
 'AI_Lab3/results/bayes_pca_results.csv',
 'AI_Lab3/results/bayes_pca_results.json',
 'AI_Lab3/results/lenet_base_metrics.json',
 'AI_Lab3/results/lenet_plus_metrics.json',
 'AI_Lab3/src/bayes_bernoulli.py',
 'AI_Lab3/src/bayes_pca.py',
 'AI_Lab3/src/common_fashion.py',
 'AI_Lab3/src/lenet_base.py',
 'AI_Lab3/src/lenet_plus.py',
 'README.md'}
PATTERNS = {
    "email address": re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"),
    "mainland mobile number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "identity number": re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    "student identifier": re.compile(r"(?i)(?:学\s*号|student[_ -]?(?:id|number))\s*[:：=]\s*[\"']?[a-z0-9]{6,18}"),
    "absolute Windows path": re.compile(r"(?i)(?<![a-z])[a-z]:[\\/][^\s\"'<>]+"),
    "personal Windows path": re.compile(r"(?i)(?<![a-z])[a-z]:[\\/](?:users|documents and settings)[\\/][^\s\"'<>]+"),
    "personal Unix path": re.compile(r"/(?:Users|home)/[a-zA-Z0-9_.-]+/"),
    "GitHub credential": re.compile(r"(?:gh[pousr]_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{30,})"),
    "cloud credential": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[0-9A-Za-z_-]{30,}"),
    "OpenAI credential": re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*[\"'][a-zA-Z0-9_+/=.-]{12,}[\"']"),
    "URL with credentials": re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
}
MAX_TEXT_SIZE = 2 * 1024 * 1024
MAX_APPROVED_BINARY_SIZE = 64 * 1024 * 1024
# Updating these files requires another content and metadata review before
# replacing a digest. Other binary files and modified bytes remain blocked.
APPROVED_BINARIES = {'人工智能/AILAB/AI_Lab1/report/report.pdf': 'f4f7672b5114fd2f8a7a11188c57a41d7d014f3c907394274e35192deb80caae',
 '人工智能/AILAB/AI_Lab2/report/report.pdf': 'af1fed4e814f0492572f3c2c25ebf31dae13ae67a0b2d573da633e5a9f4e7da5',
 '人工智能/AILAB/AI_Lab3/report/report.pdf': '5bd51467cd7c48b5d770382fb905ef92de0a2ce2106a1bc38a5d39dd289a1eab',
 '人工智能/人脸表情识别/docs/presentation.pptx': '6afcb22bff59b2b90f9ddf645ad20f93d7faa88179eba2029f01fc6b3624d5cf',
 '人工智能/人脸表情识别/docs/report.pdf': '990692cfd2f3d5017e72dcdf6d12fdf345b8cf76d18fd904606e478538266af7',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/MATLAB源码/segments/A_fur_elise_segment.wav': 'f58d1fe621aa1ce2426724406de38847d32207d487a0a7db3b507bfe56705e10',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/MATLAB源码/segments/B_hotel_california_segment.wav': '5d3c5b4189609b6ba52768829c98753f9df8a9c768063ca12b114e7a77e05a45',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/MATLAB源码/segments/C_faded_segment.wav': 'd7d813bff7de2419120fa06201662cf1ca0fc76721fc8012a41e464e8cf901cc',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_echo_reverb_waveform_compare.png': '2605aa20e52b1c15b865fed5df0c671b5ae7445015f198a7729b879c646a5fc1',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_fft_compare_original_echo_reverb.png': '0af75cd419e98763584a45aacab2ba82a56b54a9cd57fcd8c7efde35c83b0643',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_original_waveform_spectrum.png': '4788027f148e7750bb2ce065760613893fd645281ad6a43d647d8a1f4aa7c461',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_stft_echo.png': '9fe8a1d7f6b1e7c55206413ca9a916d7b1288271be7042f77c144929a1013524',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_stft_original.png': 'b4a280abbbd298c4cfe8a6facd0eb7794178716f7b17c30fedd7d282995c20fd',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_stft_reverb.png': '1bcc0388f9ad168cb01dfa732bd5dac25af98e0b946b38ee016034741d3a2b07',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/A_music_effect/figures/A_stft_window_compare.png': 'b24cde2b31f58a4e31ac2747a888a4ce3f0d8fa7bec0a091ae991b272322634c',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/B_sampling/figures/B_aliasing_direct_decimation.png': '8b3ab1b00d5fd7287010723f9eafe355d5e336c30724ecc8f64cb985d37417e1',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/B_sampling/figures/B_original_spectrum_bandwidth.png': 'ce6a4a828c7de4288d87a2f3c11c17e6ea58823986de6724140f0fdb2415cdef',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/B_sampling/figures/B_reconstruction_error_compare.png': '0cfeb616ece199660f0e8a024e1a874ac9179ff7294b58333d261558c84eb40d',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/B_sampling/figures/B_sampling_scheme_overview.png': '3e8ee62979c8dae0bdd4d5751eb678c7f01945cd3d83388b4b1b1ae0ca1d779f',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/B_sampling/figures/B_sampling_spectrum_compare.png': '8c3b82c1d2607ef4f8980efbd7512777535ca8e0f31e5c3a2c69f236998e1fe5',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/B_sampling/figures/B_sampling_time_compare.png': 'e744a90a7631f491b492104ffcd42805feeb39cb71b17abc836c01b7baedbcb0',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/C_denoising/figures/C_denoise_spectrum_compare.png': '0abfc649ac01b7ed09b86b80d87ad9388a4e6f3c280d4670c81b475261b9ab6e',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/C_denoising/figures/C_denoise_waveform_compare.png': '0d652d04e13665acb58778ee2392e6bd39890395071cded14f9b8de4bb8f12a0',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/C_denoising/figures/C_filter_frequency_response.png': '2eb28cf286ccba7aa46d20f9d7971664385f409219fd8f98a6af0758508af37c',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/C_denoising/figures/C_filter_parameter_compare.png': 'b7487e0b0ea5e00612538d70d1cd763c4fc57fa9744215ec0b4168ea64356842',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/C_denoising/figures/C_noise_waveform_spectrum_compare.png': '2bb940af4c550744c4f98b6714396f4ab8427ac7b0c78bc8734460136c963b7c',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/results/C_denoising/figures/C_snr_rmse_compare.png': '7bb98596647ccd9340b4ba906a3cf859c18d4c20c03bf14f1e51fc5e39ea4124',
 '信号与系统课程设计/课题ABC_音乐信号综合处理/设计报告.pdf': 'ddc98ac98a6d7ff3fa439a410b2cb2a3758d022089d665f9a47ceca313471bfa',
 '信号与系统课程设计/课题D_图像处理与频域分析/MATLAB源码/课题D给定图像.png': '0dff6272e47f1dfb0c630a23aa8137ec9b2940b87bf51dd94a386164dc7ab310',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/figures/figure1_original_and_spectrum.png': 'ca0fc29ac879244d94603b9c32056e409dcf315a52a43b42b75c261424bbc1e2',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/figures/figure2_filtering_results_sigma_30.png': 'f90724811cf8ac6f446d5210b3c7187bb7f35070031c09d7fff638b25f27ed8f',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/figures/figure3_filter_3d_spectrum_sigma_30.png': '0f50bc7aedffa8cb186f3555f741637664dd50508581fbab3bb3f9961e3e87b5',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/figures/figure4_lowpass_sigma_comparison.png': 'fce3e4daeacf5a3a86233b7d9e5ae010601c066a2b80cf94c7671c6898cc9116',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/figures/figure5_highpass_sigma_comparison.png': '27f0b406ba28eeac7f22c77a5483d0d88e5307d5e045f1066d7c00d2c7e2cca6',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/figures/figure6_frequency_spectrum_comparison_sigma_30.png': 'e5177dee2f90784c3013f0fc568d1eb24fd8fa6d6a0300a0d0f5ef505e1bc75d',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/gaussian_highpass_filter_sigma_10.png': '0b1b228d659ab530daa210c36f0ae976ad94c269e03460fb5470a603de77a71f',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/gaussian_highpass_filter_sigma_30.png': 'da22a44e9b8689c3d80062d4cf8c1200cc116671cbf2a75503784fb023186759',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/gaussian_highpass_filter_sigma_80.png': 'a8f2c707d22d8afa9f5edcf36223c518f081f380896d31c96c3c1710b355b1be',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/gaussian_lowpass_filter_sigma_10.png': '3fdeaec7543b738d93d05ebc60ba50c361fefe12a2e6c5fc0cdb94f1fa8d5f2a',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/gaussian_lowpass_filter_sigma_30.png': '52be74ba36a373a6562efc2360c4ce525b48408fe241c8cf2f701fc45f33af19',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/gaussian_lowpass_filter_sigma_80.png': '64c07e4b29b449b2e9184b813938c5c6455cd5c3c8fcddb0fdd62b86633340a4',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/highpass_result_sigma_10.png': 'd5b072bdbbb98da275efdac143dfb3361e13c6c66758812df564b28da546131f',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/highpass_result_sigma_30.png': '9c3ee69be1831e5369ba381d7d5c290c899d05c3256fe5a6020920adbc5c71e3',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/highpass_result_sigma_80.png': '0eaeb43fd25e52894645bd8a4d51bcef51f9f5b18330f72852d216f45c0d6dff',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/lowpass_result_sigma_10.png': '533bb8788d1173c184852ef4af3c859bcbda87003f25ec9e66082b1188bbdb6d',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/lowpass_result_sigma_30.png': '8a02cf1eef6f4f05762c60fb59a5197821844cc668ad0020e37fb98f494f5dea',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/lowpass_result_sigma_80.png': 'c4d32f475019a5dfd2fe3de4d1886d7a795df6b780349a250efa6de0684f6a74',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/original_gray.png': '93dfb651c6c083eeca2fb20eb68c7ea19e9752f85ef12f609d9a41f4ea479151',
 '信号与系统课程设计/课题D_图像处理与频域分析/results/images/original_spectrum_log.png': '6f2c69dea99c666de6570031da457ffc5f66f87f77ad5c18505efce69fb0d163',
 '信号与系统课程设计/课题D_图像处理与频域分析/设计报告.pdf': '7cb0a1ba111e99fdf94552951d1edf2f0dff4eddf677dcd15da606ef845a3bbb',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/class_accuracy_bar.png': '2eabeef190eece6369b23c7faa8a51391eda2d664158950f22f89d04cf44b5f2',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/class_distance_bar_example.png': '92d0c5837cc96ff1f7ce0c95ab59d76359560558b8929224c42ee201c3696858',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/confusion_matrix.png': '71309fc4ea3ede2bf8bb7f59e0d3daa406b238c9865795ff1a045c039d1dc5a5',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/dataset_digit_counts.png': '2e3467fc9be254fef9dce3a9bc7e5a59f9285c873af9af4dedfa92fd5b400b97',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/dtw_example_accumulated_path.png': '650feec60c6a7171e57e0c2424e434d29d6b6a7516aadeac9048f77d5139df27',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/dtw_example_local_distance.png': 'e41b073092605ea4158e848103fb701697d22c0e4331766c67165c2f24f50350',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/dtw_example_mfcc_sequence_compare.png': '5cff49f520ab1b49c6cfd6c3246666628eb86712ea9fc133a6f8e6087d46f5ce',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/mfcc_heatmap_example.png': '1e16ab35b20cfc69f93f4ff2f2a1958ccb84b9f858ee5126a7b35c79ddb2aacd',
 '信号与系统课程设计/课题E_孤立数字语音识别/results/figures/preprocess_example.png': '8331c59f9a738e8fdad7dff4808527e1bb4e4a64ba1a19e0cb3552aa91c4d487',
 '信号与系统课程设计/课题E_孤立数字语音识别/设计报告.pdf': 'f9eb1b31c9301bf29f56966eb498a3e9187ef0fbeb80cc5bf7726fd5023c67c6',
 '电子工程训练（甲）/基础实验/基础部分实验报告.pdf': '0f8dc3dbce39236cef5309d9d13f059fccb63c0787298752035626e0ce17e955',
 '电子工程训练（甲）/小车/最终照片/01-上视图.png': '903ca3d1c2e11591b36b4d9d9122f064c0985a47531ac5b1107706152bd1afb7',
 '电子工程训练（甲）/小车/最终照片/02-底部结构.png': 'c49b190c5a090ecc91de3eaa4ea8d409154ce15b9768c015f09cb4b65bf0cc55',
 '电子工程训练（甲）/小车/最终照片/03-正面结构.png': '089b2c53887d5420782dc87579285486265b6fb341bd17c06adbcadfde3f062a',
 '电子工程训练（甲）/小车/最终照片/04-侧面全貌.png': '9e94ff91ac407ae74c5e65ff32417f0c25c2ce974a890f7299aa06c9811a90dc',
 '电子工程训练（甲）/智能插座/智能插座测试报告.pdf': 'cc68b01f1f50afe471369fb935009f045cf77b1f8fc1267458d983c06140264d',
 '电子工程训练（甲）/智能插座/照片/01-PCB元件面.png': 'd656f0356ce88f56b4a2343c88d4978e30d213ccfa09947dc26c7e0f87793c6d',
 '电子工程训练（甲）/智能插座/照片/02-PCB焊接面.png': '30db4649fa16854093f4c6153cf5b9907bc6b3c220af32ca95ba5539e46e1177'}


def allowed_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if name in EXACT_PATHS or name in APPROVED_BINARIES:
        return True
    if name.startswith(SIGNALS_PREFIX):
        return name[len(SIGNALS_PREFIX):] in SIGNALS_TEXT_PATHS
    if name.startswith(LAB_PREFIX):
        return name[len(LAB_PREFIX):] in LAB_TEXT_PATHS
    if not name.startswith(FACE_PREFIX):
        return False
    path = PurePosixPath(name[len(FACE_PREFIX):])
    if str(path) in {"README.md", "requirements.txt", "docs/report.tex", "models/detectors/README.md"}:
        return True
    if path.suffix == ".py" and len(path.parts) == 1:
        return True
    if path.suffix == ".py" and len(path.parts) == 2 and path.parts[0] in {"models", "utils"}:
        return True
    return len(path.parts) == 3 and path.parts[:2] == ("docs", "results") and path.suffix in {".txt", ".csv", ".md"}



def inspect_content(name: str, data: bytes, mode: str = "100644") -> list[str]:
    findings = []
    if not allowed_path(name):
        findings.append("file outside publication allowlist")
    if mode not in {"100644", "100755"}:
        findings.append("symlink or submodule is not allowed")
    limit = MAX_APPROVED_BINARY_SIZE if name in APPROVED_BINARIES else MAX_TEXT_SIZE
    if len(data) > limit:
        return findings + ["oversized publication file"]
    if name in APPROVED_BINARIES:
        if hashlib.sha256(data).hexdigest() != APPROVED_BINARIES[name]:
            findings.append("binary document differs from the manually audited version")
        if name.endswith(".pdf") and (not data.startswith(b"%PDF-") or not data.rstrip().endswith(b"%%EOF")):
            findings.append("invalid report PDF envelope")
        if name.endswith(".pptx") and not data.startswith(b"PK\x03\x04"):
            findings.append("invalid presentation envelope")
        if name.endswith(".png") and (not data.startswith(b"\x89PNG\r\n\x1a\n") or not data.endswith(b"IEND\xaeB`\x82")):
            findings.append("invalid approved PNG envelope")
        if name.endswith(".wav") and (data[:4] != b"RIFF" or data[8:12] != b"WAVE" or int.from_bytes(data[4:8], "little") + 8 != len(data)):
            findings.append("invalid approved WAV envelope")
        return findings
    if b"\0" in data:
        return findings + ["binary file"]
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return findings + ["file is not UTF-8 text"]
    for label, pattern in PATTERNS.items():
        if pattern.search(name):
            findings.append("filename: " + label)
        for line_number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                findings.append(f"line {line_number}: {label}")
    return findings


def git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise RuntimeError("Git inspection failed; initialize and stage the publication tree first.")
    return result.stdout


def index_files():
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, object_id, stage = header.decode("ascii").split()
        if stage != "0":
            raise RuntimeError("Resolve Git index conflicts before publishing.")
        name = raw_name.decode("utf-8")
        if mode == "160000":
            yield name, b"", mode
        else:
            yield name, git("cat-file", "blob", object_id), mode


def tree_files():
    # Skip only Git internals and Python bytecode created by local verification.
    def walk(directory: Path):
        for path in sorted(directory.iterdir()):
            if path.name in {".git", "__pycache__"}:
                continue
            name = path.relative_to(ROOT).as_posix()
            if path.is_symlink():
                yield name, b"", "120000"
            elif path.is_dir():
                yield from walk(path)
            else:
                yield name, path.read_bytes(), "100644"
    yield from walk(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--tree", action="store_true", help="Inspect publication files on disk")
    modes.add_argument("--tracked", action="store_true", help="Inspect all Git index blobs")
    modes.add_argument("--staged", action="store_true", help="Inspect all Git index blobs before committing")
    args = parser.parse_args()
    count = 0
    failed = False
    try:
        for name, data, mode in tree_files() if args.tree else index_files():
            count += 1
            findings = inspect_content(name, data, mode)
            if findings:
                failed = True
                # Do not print filenames: they may themselves contain private data.
                print(f"File #{count}: " + "; ".join(findings), file=sys.stderr)
        if count == 0:
            raise RuntimeError("No publication files found; refusing an empty check.")
    except (OSError, RuntimeError, UnicodeError) as exc:
        print(f"Privacy check could not complete: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(f"Checked {count} files; " + ("review required." if failed else "no configured privacy patterns found."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
