.. _improving-cuda-compilation-time:

5.5. 优化 CUDA 编译时间
=======================

.. _compile-time-introduction:

5.5.1. 简介
-----------

本节介绍用于测量和诊断编译时间瓶颈的技术、编译选项，以及用于缩短 CUDA 代码编译时间的源代码重构策略。

以下是应对编译时间问题的推荐工作流程：

.. figure:: /_static/images/workflow.png
   :alt: 用于测量和缩短编译时间的推荐工作流程
   :width: 80%
   :align: center

   用于测量和缩短编译时间的推荐工作流程

测量编译时间有两种方式：以编译器为中心（compiler-centric）和以源代码为中心（source-centric）。在
以编译器为中心的视图中，nvcc/NVRTC 的 ``-time`` 选项会报告编译时间在各个编译阶段
（例如解析器、优化器）之间的细分。在以源代码为中心的视图中，用户可以生成按源代码元素
（例如头文件和模板实例化）细分编译时间的编译时间火焰图（flame chart）。

或者，`CTAdvisor <https://docs.nvidia.com/cuda/cuda-compile-time-advisor/index.html>`__ 工具
会生成按源代码元素细分的编译时间摘要，以及可用于缩短编译时间的适用编译选项的建议。
改善编译时间通常是一个迭代过程，需要相继解决下一个编译时间瓶颈，直到编译时间令人满意为止。

本文档的其余部分介绍如何测量编译时间，以及各种可用于缩短编译时间的编译选项和源代码修改。

.. _compile-time-measurement:

5.5.2. 测量
-----------

.. _compile-time-using-the-time-flag:

5.5.2.1. 使用 -time 选项
^^^^^^^^^^^^^^^^^^^^^^^^

nvcc 和 NVRTC 支持 ``-time <filename>`` 选项，用于测量编译每个阶段所花费的编译时间。
输出会追加到指定文件的末尾。

示例：

以下是 NVRTC 使用 ``-time`` 选项生成的示例输出：

.. code-block:: text

   File name, phase name, metric, unit
   j4.cu.ptx, CUDA C++ Frontend, 30, ms
   j4.cu.ptx, NVVM Optimizer, 1000, ms
   j4.cu.ptx, PTX Translation, 14, ms

按阶段细分的时间数据可以指出改善编译时间的潜在途径。在上述示例中，NVVM Optimizer 阶段
占据了大部分时间，因此可以通过调整已知能够加速 NVVM 编译的编译选项
（例如 ``--split-compile``）来潜在地改善编译时间。

.. _compile-time-using-compilation-flame-chart-fdevice-time-trace:

5.5.2.2. 使用编译火焰图（-fdevice-time-trace）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于许多程序，修改源代码而不是添加编译选项可能带来显著的编译时间改善。要了解需要修改哪些
源代码，我们首先需要获取占用最多编译时间的源代码元素的信息。火焰图以图形化的方式展示按
源代码元素（例如模板）标注的编译时间。使用 ``-fdevice-time-trace <filename>`` 选项调用
nvcc/NVRTC 生成设备编译火焰图。这会生成一个遵循 Trace Event 格式的 .json 文件，
Trace Event 是一种广泛用于性能分析的格式。可以使用 Perfetto 跟踪查看器
（`https://ui.perfetto.dev/ <https://ui.perfetto.dev/>`__）打开该 .json 跟踪文件。

火焰图让我们能够可视化编译过程的端到端工作流程，提供所有主要阶段的完整时间线：

- 预处理
- 主机代码和设备代码编译
- 设备链接
- 二进制文件生成

这种整体视角对于理解不同阶段如何相互作用并共同影响总编译时间特别有价值。通过检查这一
可视化结果，你可以找出在工作流程中占主导地位的阶段，判断延迟是源于特定的代码构造还是
编译器流程本身，并确定优化的优先次序。

示例：

.. figure:: /_static/images/compilation_flame_chart.png
   :alt: 多目标架构的编译火焰图
   :width: 80%
   :align: center

   多目标架构的编译火焰图

该火焰图展示了针对多个架构（compute_90、compute_100）的编译，其中大量时间花费在前端。
可以通过允许 nvcc 使用多个线程进行编译（``-threads N``）来重叠多个架构的编译。

火焰图还可用于查找占用大量编译时间的头文件和模板。

示例：

.. figure:: /_static/images/template_instantiation_flame_chart.png
   :alt: 显示递归模板实例化的火焰图
   :width: 80%
   :align: center

   显示递归模板实例化的火焰图

``recursiveTree`` 类具有深层嵌套的模板实例化。在本例中，重构代码以避免过多的实例化将
改善整体编译时间。

有关如何使用编译时间火焰图的更多详细信息，请参阅这篇
`博客文章 <https://developer.nvidia.com/blog/optimizing-compile-times-for-cuda-c/>`__。

.. _compile-time-using-compile-time-advisor:

5.5.2.3. 使用 Compile Time Advisor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compile Time Advisor 是 CUDA Toolkit 中包含的一个命令行工具。它处理由 nvcc/NVRTC 的
``--fdevice-time-trace`` 选项生成的 .json 文件，并将编译时间诊断信息和建议汇总到生成的
文本文件中。

目前，Compile Time Advisor 提供五种不同类型的建议：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 建议
     - 描述
   * - Expensive Templates
     - 识别实例化耗时最长的模板函数/类以及实例化的数量。
   * - Expensive Headers
     - 识别包含处理耗时最长的头文件，并细分处理每个子头文件所花费的时间。
   * - Parallelizable Compilation
     - 检查源文件是否为多个架构（多个 ``-gencode`` 目标）编译，以及是否未使用 ``--threads``
       选项（使用 ``--threads`` 可以重叠按架构的编译，从而缩短总体编译时间）。
   * - Split Compilation
     - 识别优化耗时较高、可以通过 ``--split-compile`` 选项并行运行优化而受益的源文件。
   * - Compile Time Breakdown
     - 将每个编译单元的编译时间细分为各个编译步骤，列出编译耗时最长的源文件，并帮助确定
       编译的哪些阶段是瓶颈。

例如，以下是 Compile Time Advisor 的示例输出，仅截取了 Expensive Templates 和
Expensive Headers 两条建议：

.. code-block:: text

   //-------------------------------------------------------------
   Trace file loading is complete
   ***********************************************
   Start expensive template advisor
   Top template functions/classes by instantiation time:
     [0.00s]: (2 times in 2 files) std::__detail::__hyperg
       [0.00s]: (1 times in 1 files) std::__detail::__hyperg<float>(float, float, float, float)
         [0.00s]: (1x) saxpy.cu, recursively instantiates 12 templates with max depth 5
   …
   ***********************************************
   Start expensive headers advisor
   Top header files by include processing time:
    [0.05s]: (1 times in 1 files) /usr/local/cuda-13.0/bin/../targets/x86_64-linux/include/cuda_runtime.h
        Processing this header itself takes 0.00s (1.65%)
        includes /usr/local/cuda-13.0/bin/../targets/x86_64-linux/include/crt/common_functions.h (1x) taking 0.03s (58.73%)
   …
   //-------------------------------------------------------------

有关如何使用该工具的更多详细信息，请参阅
`Compile Time Advisor 文档 <https://docs.nvidia.com/cuda/cuda-compile-time-advisor/index.html>`__。

.. _compile-time-compiler-flags-to-improve-compile-time:

5.5.3. 用于改善编译时间的编译选项
---------------------------------

使用上一节介绍的方法，我们可以识别影响编译时间的编译阶段和程序特征。然后可以尝试本节介绍的
编译选项来缩短编译时间。

下表总结了可用的编译选项：

.. list-table::
   :header-rows: 1
   :widths: 22 22 16 28 12

   * - 选项
     - 描述
     - 工具
     - 典型用例
     - 运行时性能影响
   * - ``--threads <N>``
     - 编译使用的线程数。
     - nvcc
     - 配合 ``--gencode`` 进行多架构编译。
     - 无
   * - ``--split-compile <N>``
     - 并发优化使用的线程数。
     - nvcc、nvrtc、nvlink、nvJitLink、ptxas
     - 包含多个 kernel 或未内联设备函数的编译单元。
     - 极小
   * - ``--split-compile-extended <N>``
     - 并发优化使用的线程数。是 split-compile 的更激进形式。仅限 LTO。
     - nvcc、nvrtc、nvlink、nvJitLink
     - LTO（链接时优化）处理，可以接受一定程度的运行时性能下降的场合。
     - 取决于程序
   * - ``--Ofast-compile <min/mid/max>``
     - 设备代码优化级别
     - nvcc、nvrtc、nvlink、nvJitLink、ptxas
     - 以适度到显著的运行时性能增加为代价，换取编译时间的大幅缩短。
     - 取决于程序
   * - ``--gen-opt-lto``
     - 在保存 LTO-IR 之前运行 NVVM 优化器
     - nvcc、nvrtc
     - LTO-IR 片段基本自包含；将优化开销从运行时移到构建时。
     - 取决于程序
   * - ``--pch`` ``--use-pch``
     - 预编译头支持。自动（``-pch``）与显式（``-use-pch``）。
     - nvrtc
     - 编译使用公共头文件前缀的多个程序，且前端（解析）时间是瓶颈。注意：会强制前端串行化，
       因此可能影响并发 NVRTC 调用的扩展性。
     - 无
   * - ``--minimal``
     - 禁用某些很少使用的语言特性
     - nvrtc
     - 前端时间是瓶颈。在现代机器上可节省数十毫秒量级的固定开销。
     - 无
   * - ``--opt-level <N>``
     - ptxas 的优化级别
     - ptxas
     - 以显著的运行时性能增加为代价，换取更快的编译时间。
     - 取决于程序
   * - ``--allow-expensive-optimizations=false``
     - 禁止激进优化
     - ptxas
     - 以可能显著的运行时性能增加为代价，换取更快的编译时间。
     - 取决于程序
   * - ``--register-usage-level <N>``
     - 控制影响寄存器使用的优化的激进程度
     - ptxas
     - 以可能显著的运行时性能增加为代价，换取更快的编译时间。
     - 取决于程序
   * - ``--jobserver``
     - 启用 GNU jobserver 支持
     - nvcc、ptxas
     - 避免超额订阅 CPU 核心数量
     - 无

以下是各选项按类别分组的详细说明。

.. _compile-time-concurrent-compilation-options:

5.5.3.1. 并发编译选项
^^^^^^^^^^^^^^^^^^^^^

.. _compile-time-threads:

5.5.3.1.1. 线程
"""""""""""""""

``--threads <N> (-t)``

**描述**

指定为多个架构编译时并行创建的最大线程数。如果 ``<N>`` 为 1，或者只为单个架构编译，
则忽略此选项。如果 ``<N>`` 为 0，线程数将等于机器上的 CPU 数量。

**支持的工具**

nvcc

**何时使用**

- 使用 ``-gencode`` 选项为多个目标架构编译 CUDA 文件。在这种情况下，可以使用不同的线程
  重叠多个目标的编译，从而缩短总体编译时间。

  使用 GNU make 或 Ninja 构建系统时，可以与 ``--jobserver`` 选项配合使用，以避免超额订阅
  系统上的 CPU 核心数量。

**运行时性能影响**

无 —— 此选项不会影响生成代码的质量。

.. _compile-time-split-compile:

5.5.3.1.2. 拆分编译
"""""""""""""""""""

``--split-compile <N> (-split-compile)``

**描述**

指定运行编译器优化时使用的最大并发线程数。如果 ``<N>`` 为 1，则忽略此选项。
如果 ``<N>`` 为 0，线程数将等于机器上的 CPU 数量。

**支持的工具**

nvcc、nvrtc、nvlink、nvJitLink、ptxas

**何时使用**

- NVVM 或 PTXAS 编译步骤是编译时间瓶颈。
- 代码包含多个 kernel 或未内联的设备函数。

  编译器优化器将使用指定数量的线程并发运行按函数的优化。使用 GNU make 或 Ninja 构建系统时，
  可以与 ``--jobserver`` 选项配合使用，以避免超额订阅系统上的 CPU 核心数量。

**运行时性能影响**

极小 —— 在大多数情况下，这不应影响生成的设备机器代码的质量。

.. _compile-time-extended-split-compile:

5.5.3.1.3. 扩展拆分编译
"""""""""""""""""""""""

``--extended-split-compile <N> (-split-compile)``

**描述**

指定在 LTO 模式下运行编译器优化时使用的最大并发线程数。如果 ``<N>`` 为 1，则忽略此选项。
如果 ``<N>`` 为 0，线程数将等于机器上的 CPU 数量。

此选项是拆分编译的更激进形式，可能会影响编译后二进制的性能。它仅在 LTO 模式下可用。

**支持的工具**

nvcc、nvrtc、nvlink、nvJitLink

**何时使用**

- 已开启 LTO 优化。
- NVVM 处理是瓶颈。
- 可以接受程序运行时增加的潜在可能性。

  使用 GNU make 或 Ninja 构建系统时，可以与 ``--jobserver`` 选项配合使用，以避免超额订阅
  系统上的 CPU 核心数量。

**运行时性能影响**

取决于程序 —— 对于主要包含内联代码的 kernel，影响可能是极小的。

.. _compile-time-jobserver:

5.5.3.1.4. Jobserver
""""""""""""""""""""

``--jobserver (-jobserver)``

**描述**

启用 GNU Make Jobserver 支持。GNU jobserver 协议由 GNU Make 和 Ninja 构建系统支持。
当同时使用并发编译选项（``--threads``/``--split-compile``/``--extended-split-compile``）
和并发构建命令（例如 GNU make 的 ``-j`` 选项）时，此选项可防止 CPU 超额订阅。

**支持的工具**

nvcc、ptxas

**何时使用**

- 构建使用 GNU make 或 ninja 构建系统，并进行并发构建（例如 GNU make 的 ``-j`` 选项）。
- 并发编译选项处于激活状态（``--threads``/``--split-compile``/``--extended-split-compile``）。

**运行时性能影响**

无 —— 此选项不会影响生成代码的质量。

.. _compile-time-nvrtc-specific-options:

5.5.3.2. NVRTC 专用选项
^^^^^^^^^^^^^^^^^^^^^^^

.. _compile-time-precompiled-headers:

5.5.3.2.1. 预编译头
"""""""""""""""""""

``--pch (-pch)``

``--use-pch <filename> (-use-pch)``

**描述**

启用自动预编译头（PCH）模式（``--pch``）或显式 PCH 模式（``--use-pch``）。
详细信息和示例代码请参阅 NVRTC 文档。

**支持的工具**

nvrtc

**何时使用**

- 有多个需要编译的 NVRTC 程序，并且它们使用公共的头文件前缀。例如：

  .. code-block:: cpp

     #include "foo.h"
     #include "bar.h"

- 前端编译时间占比较高。

注意：使用预编译头会使前端处理串行化，当来自不同主机线程的多个并发 NVRTC 调用正在进行时，
会对扩展性产生负面影响。随着并发 NVRTC 调用数量的增加，这种影响会变得更加显著；
在这种情况下，可以尝试关闭 PCH 以消除扩展性瓶颈。

**运行时性能影响**

无 —— 此选项不会影响生成代码的质量。

.. _compile-time-minimal:

5.5.3.2.2. Minimal
""""""""""""""""""

``--minimal (-minimal)``

**描述**

省略某些语言特性，以缩短小型程序的编译时间。具体来说，将省略以下内容：

- 纹理（texture）和表面对象（surface）函数及相关类型，例如 ``cudaTextureObject_t``。
- 由 cudadevrt 设备代码库提供的 CUDA Runtime 函数，通常以 "cuda" 前缀命名，
  例如 ``cudaMalloc``。
- 从设备代码启动 kernel。
- 与 CUDA Runtime 和 Driver API 相关的类型和宏，由 cuda/tools/cudart/driver_types.h
  提供，通常以 "cuda" 前缀命名，例如 ``cudaError_t``。

**支持的工具**

nvrtc

**何时使用**

- 前端处理时间占比较高的小型源程序。
- 减少编译器的一些固定开销，通常可节省数十微秒量级的时间。

**运行时性能影响**

无 —— 此选项不会影响生成代码的质量。

.. _compile-time-ptxas-specific-options:

5.5.3.3. PTXAS 专用选项
^^^^^^^^^^^^^^^^^^^^^^^

``--opt-level <N> (-O)``

``--allow-expensive-optimizations=false (-allow-expensive-optimizations)``

``--register-usage-level <0…10> (-regUsageLevel)``

**描述**

这些选项控制 PTXAS 优化器。

- ``--opt-level`` 设置总体优化级别（默认值：3）。
- ``--allow-expensive-optimizations=false`` 关闭耗时的编译优化（默认值：在优化级别 >= 2 时
  自动开启）。
- ``--register-usage-level`` 控制针对寄存器使用的优化的激进程度。较高的值会激进地优化
  源程序，以增加寄存器使用为代价换取生成代码的潜在改进。（默认值：5）。

**支持的工具**

ptxas

**何时使用**

- PTX 编译是编译时间瓶颈。
- 实测的运行时影响在可接受的限度内。

**运行时性能影响**

取决于程序 —— 尝试不同的选项值，测量编译时间加速与运行时增加的对比，以找到可接受的折中。

.. _compile-time-miscellaneous-options:

5.5.3.4. 其他选项
^^^^^^^^^^^^^^^^^

.. _compile-time-fast-compile:

5.5.3.4.1. 快速编译
"""""""""""""""""""

``--Ofast-compile <level> (-Ofc)``

**描述**

指定设备代码的快速编译（fast-compile）级别，该级别通过在不同程度上禁用某些优化，
控制编译速度与运行时性能之间的折中。

级别：

- ``max``：只关注最快的编译速度，禁用许多优化。
- ``mid``：平衡编译时间和运行时，禁用代价高昂的优化。
- ``min``：对编译时间和运行时的影响都更有限，最小化某些代价高昂的优化。
- ``0``：禁用 fast-compile。

此选项默认处于禁用状态。

**支持的工具**

nvcc、nvrtc、nvlink、nvJitLink、ptxas

**何时使用**

- 编译时间比峰值运行时更重要（例如，先进行一次快速的初始编译以降低用户可感知的延迟，
  然后在后台进行一次较慢的、完全优化的编译）。

**运行时性能影响**

取决于程序 —— 尝试不同的级别，测量编译时间加速与运行时增加的对比，以找到可接受的折中。

.. _compile-time-optimized-lto-ir:

5.5.3.4.2. 优化的 LTO-IR
""""""""""""""""""""""""

``--gen-opt-lto (-gen-opt-lto)``

**描述**

默认情况下，nvcc 和 NVRTC 会生成未经优化的 LTO-IR 代码，并且 NVVM 优化器只在 LTO-IR
片段链接在一起之后才运行。然而，对于使用 nvJitLink 库链接 LTO-IR 片段以在线生成机器代码的
工作流，运行 NVVM 优化器所带来的构建时间开销可能是显著的。

作为一种替代方案，可以使用此选项指示 nvcc 和 NVRTC 在输出 LTO-IR 之前先在 LTO-IR 上运行
NVVM 优化器传递。这使得一些优化可以从在线阶段移到 LTO-IR 生成阶段，从而减少整体的在线
处理时间。

注意：此选项会导致优化器运行两次（一次在 LTO-IR 生成期间，另一次在 LTO-IR 片段链接之后），
因此生成的机器代码可能与默认的 LTO-IR 处理模式不同。在默认模式中，所有优化都推迟到
LTO-IR 片段链接之后进行。

**支持的工具**

nvcc、nvrtc

**何时使用**

- 工作流使用 nvJitLink 库进行在线 LTO-IR 链接。
- 被链接的 LTO-IR 片段大体上是自包含的，并且只有粗粒度的交互，例如 FFT 流水线的不同阶段。
  这使离线优化器能够更有效地优化代码，从而减少 LTO-IR 片段链接之后在线调用的 NVVM
  优化器需要完成的工作。
- 另一方面，如果 LTO 片段之间存在细粒度的交互，那么独立作用于每个 LTO-IR 片段的离线
  优化器可能无法有效地优化代码，大部分优化工作仍将留给在线调用的优化器完成。
- 如前所述，与默认的 LTO 工作流相比，运行优化器两次可能会改变生成的机器代码，因此请确保
  生成代码的性能没有意外下降。

**运行时性能影响**

取决于程序 —— 通过实验确认运行时影响是否可以接受。

.. _compile-time-user-code-and-workflow-changes:

5.5.4. 用户代码与工作流程修改
-----------------------------

在本节中，我们讨论对用户代码和编译器调用方式的修改，以加快编译速度。

.. _compile-time-remove-unused-headers:

5.5.4.1. 移除未使用的头文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^

在大型软件项目中，通常存在一个每个编译单元都会包含的公共头文件「前缀」。例如，某些机器学习
实现会生成无条件包含定义 ``__half`` 和 ``__bfloat16`` 类型的 CUDA C++ 头文件的 NVRTC 代码，
即使生成的 NVRTC 程序中实际上并未使用这些类型：

.. code-block:: cpp

   #include <cuda_fp16.h>
   #include <cuda_bf16.h>
   /* other includes */

遗憾的是，这些头文件包含大量内联（inline）的 ``__device__`` 函数，这会增加解析时间。

应对措施：

查看 ``-time`` 选项报告的按阶段编译时间细分。或者，使用 ``-fdevice-time-trace`` 生成
编译时间跟踪并运行 Compile Time Advisor，检查报告中的 "Expensive Headers Advisor"
以找出开销较大的头文件，并检查 "Compile Time Breakdown Advisor" 以确定前端处理时间
是否占比较高。

移除未使用的头文件和样板代码（boilerplate code），例如类型定义、static/inline 辅助函数。

.. _compile-time-avoid-excessive-forceinline-functions:

5.5.4.2. 避免过多的 __forceinline__ 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一般而言，CUDA 编译器会积极地进行内联，因为内联能够实现一些重要的优化，例如将局部变量
提升到寄存器。此外，使用 ``__forceinline__`` 标记函数会指示编译器忽略内联启发式规则，
尽全力内联目标函数。

然而，不加选择地内联可能导致代码爆炸（code explosion），使 NVVM 和 PTX 转换阶段变慢。

应对措施：

- 尽可能避免使用 ``__forceinline__`` 注解，让编译器的启发式规则自行决定是否内联该函数。
- 对预计不会被频繁调用的函数应用 ``__noinline__`` 注解。

示例：

.. code-block:: cpp

   __device__ int  large_func(int in) { ... } // a large function
   __device__ void parent() {
     #pragma unroll
     for (int i = 0; i < 100; ++i) {
       if (unlikely_condition) {
             result += large_func(i); // <- Code explosion due
                                       // to inlining “large_func”
           }
         }
       }

- 在该示例中，对 large_func() 的调用很少发生，然而由于外层循环被 ``#pragma unroll`` 标记，
  当循环被展开且 large_func 被内联时，就发生了代码爆炸。
- 通过将 large_func 标记为 ``__noinline__`` 来修复编译时间问题。

.. _compile-time-avoid-excessive-loop-unrolling:

5.5.4.3. 避免过度的循环展开
^^^^^^^^^^^^^^^^^^^^^^^^^^^

一般而言，完全展开具有已知边界的循环会生成更优化的代码。然而，展开会显著增加需要由 NVVM
和 PTXAS 优化器处理的代码量，导致编译时间变慢。

应对措施：

- 在最外层循环上使用 ``#pragma unroll 1`` 禁用循环展开，并测量其对运行时性能的影响。
  在许多情况下，编译时间的提升对于实测的运行时性能下降而言可能是可以接受的折中。
- 在这种情形下，查找数组下标表达式依赖输入或线程 ID 的最外层循环；这类循环可能是禁用展开的
  良好候选对象（因为即使完全展开循环，这些数组也无法提升到寄存器）。

示例：

.. code-block:: cpp

   int arr[32]{};

   #pragma unroll    // problematic unroll for compile time
   for(int i = 0; i < 32; ++i) {
     arr[i%threadIdx.x] = large_func(i); // runtime dependent
                                         // index
   }
   ...
   #pragma unroll
   for (int i = 0; i < 32; ++i) {
     result[i] = arr[i];
   }

- 在这个示例中，展开第一个循环导致代码爆炸，因为 large_func() 在展开后的循环体中被多次
  内联。然而，由于 arr 的下标表达式依赖线程 ID（i%threadIdx.x），展开循环体并不能使 arr
  的元素提升到寄存器；该数组仍驻留在局部内存中。
- 因此，使用 ``#pragma unroll 1`` 禁用循环展开将提升编译时间，同时不太可能明显降低
  运行时性能。

.. _compile-time-use-concurrent-nvrtc-and-nvjitlink-invocation:

5.5.4.4. 并发调用 NVRTC 和 nvJitLink
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在某些场景下，用户代码可能有大量需要使用 NVRTC 和 nvJitLink 编译的 CUDA C++ 程序，
例如在处理对应于深度学习模型的图时，其中每个节点对应一个独立的 CUDA C++ kernel。

应对措施：

使用多个主机线程为不同的程序并发调用 NVRTC（或 nvJitLink），以缩短总体编译时间。

.. _compile-time-be-careful-when-using-llvm-splitmodule-for-partitioning:

5.5.4.5. 使用 llvm::SplitModule() 进行分区时需要注意
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 为了获得良好的吞吐量，一些客户端可能会选择将 LLVM 代码拆分为独立的 LLVM 模块，然后使用
  独立的线程并发处理每个模块，以生成 PTX 和 GPU 机器代码
  （例如 `openxla/xla <https://github.com/openxla/xla>`__）。
- 如果在使用并发编译时遇到加速效果不佳的情况，请检查划分出的各个模块是否均匀，即不应有少数
  模块明显大于其他模块，导致相应的编译线程需要长得多的时间才能完成。例如，当使用
  llvm::SplitModule() 进行分区、且有多个 kernel 函数调用同一个公共设备函数时，就可能出现
  这种情况（在这种情况下，这些 kernel 与该设备函数最终会被划分到同一个分区中）。

应对措施：

一种可能的解决方案是在调用 llvm::SplitModule() 之前先进行激进的内联 —— 这最终会创建不再
调用其他函数的函数（kernel），从而允许 llvm::SplitModule() 将这些函数拆分到可以更高效地
并发处理的独立模块中。

示例：

.. code-block:: cpp

   __device__  static int get() { /* some code */ }
    __global__ void kernel1(int *ptr) { *ptr = get(); }
    __global__ void kernel2(int *ptr) { *ptr = get()+1; }

llvm::SplitModule() 很可能会把所有函数放入同一个模块，从而无法进行并发编译。在调用
llvm::SplitModule() 之前先进行内联，get() 会被内联到 kernel1() 和 kernel2() 中，
之后 kernel1() 和 kernel2() 就可以处于不同的分区，从而能够被并发编译。
