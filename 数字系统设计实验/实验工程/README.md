# 实验工程

| 目录 | 实验内容 | 顶层模块 |
| --- | --- | --- |
| `lab1` | 流水灯与可逆计数器 | `FlowingWaterLight` |
| `lab3_IPCore` | 图像 ROM、时钟管理与视频输出 | `IPCore` |
| `lab5` | 加法器、比较器、绝对值与模式比较 | `abs_dif` |
| `lab7` | 计数器与定时器 | `TimerTop` |
| `lab11_button_process_unit` | 同步、防颤与单周期按键脉冲 | `button_unit` |
| `lab15_dds` | 相位累加与正弦查表 | `dds` |
| `lab19_MusicPlayer` | 四首乐曲的播放、暂停与切换 | `MusicPLayerTop` |

每组实验的 `src/` 包含 Verilog 源码和测试平台。带板级设计的实验同时提供 `.xdc` 引脚约束。原工程使用 Artix-7 `xc7a200tfbg484-1` 器件。

在对应实验目录中执行以下命令可创建 Vivado 工程，输出目录为 `build/`：

```sh
vivado -mode batch -source create_project.tcl
```

已有的 `.xpr` 文件位于 `vivado/`。图像 ROM 和音频时钟的 IP 配置以 `.xci` 保存，需要在 Vivado 中生成输出产品。`lab3_IPCore/fig/fig.coe` 是图像 ROM 的初始化数据。

ModelSim 仿真时，将本实验的设计文件和需要的 `*_tb.v` 加入独立的工作库。`*_func_synth.v` 是接口网表的功能模型，不能与同名的接口声明重复编译。视频工程的仿真还需要 Vivado 生成的 IP 模型和器件库。

音乐播放器中的 `AudioInterface` 和 `button_process_unit` 使用课程提供的接口声明及 EDIF 网表。DDS、主控制器、乐曲读取器和音符播放器的测试平台可分别仿真。
