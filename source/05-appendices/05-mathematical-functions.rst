.. _mathematical-functions:

5.5. 浮点计算
=============

.. _floating-point-introduction:

5.5.1. 浮点介绍
---------------

自 1985 年采用 `IEEE-754 标准 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__ 进行二进制浮点运算以来，几乎所有主流计算系统（包括 NVIDIA 的 CUDA 架构）都已实现了该标准。IEEE-754 标准规定了浮点运算结果应如何近似。

要在所需精度下获得准确结果并实现最高性能，重要的是考虑浮点行为的许多方面。这在异构计算环境中尤为重要，因为操作在不同类型的硬件上执行。

以下章节回顾浮点计算的基本属性，并介绍融合乘加（FMA）运算和点积。这些示例说明不同的实现选择如何影响精度。

.. _floating-point-format:

5.5.1.1. 浮点格式
^^^^^^^^^^^^^^^^^

浮点格式和功能在 `IEEE-754 标准 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__ 中定义。

该标准规定二进制浮点数据在三个字段上编码：

- **符号（Sign）**：一位，指示正数或负数。
- **指数（Exponent）**：编码基数为 2 的指数，偏移一个数值偏差。
- **有效数字（Significand）** （也称为*尾数*或*小数*）：编码数值的小数部分。

.. figure:: /_static/images/floating-point-encoding.drawio.png
   :alt: 浮点编码
   :height: 24px

最新的 IEEE-754 标准定义了以下二进制格式的编码和属性：

- 16 位，也称为半精度，对应 CUDA 中的 ``__half`` 数据类型。
- 32 位，也称为单精度，对应 C、C++ 和 CUDA 中的 ``float`` 数据类型。
- 64 位，也称为双精度，对应 C、C++ 和 CUDA 中的 ``double`` 数据类型。
- 128 位，也称为四精度，对应 CUDA 中的 ``__float128`` 或 ``_Float128`` 数据类型。

这些类型具有以下的位长度：

.. figure:: /_static/images/floating-point-ieee.drawio.png
   :alt: IEEE-754 浮点编码
   :align: center

正规值的浮点编码关联的数值计算如下：

.. math::

   (-1)^{\mathrm{sign}} \times 1.\mathrm{mantissa} \times 2^{\mathrm{exponent} - \mathrm{bias}}

对于次正规值，公式修改为：

.. math::

   (-1)^{\mathrm{sign}} \times 0.\mathrm{mantissa} \times 2^{1-\mathrm{bias}}

指数分别对单精度和双精度偏移 127 和 1023。 ``1.`` 的整数部分在小数中是隐含的。

例如，值 :math:`-192 = (-1)^1 \times 2^7 \times 1.5` 编码为负号、指数 7 和小数部分 0.5。因此指数 7 用 ``float`` 的值 ``7 + 127 = 134 = 10000110`` 和 ``double`` 的值 ``7 + 1023 = 1030 = 10000000110`` 的位字符串表示。尾数 ``0.5 = 2^{-1}`` 用第一个位置为 ``1`` 的二进制值表示。-192 在单精度和双精度中的二进制编码如下图所示：

.. figure:: /_static/images/floating-point-192.drawio.png
   :alt: -192 的浮点表示
   :align: center

由于小数字段使用有限数量的位，并非所有实数都可以精确表示。例如，小数 :math:`2 / 3` 的数学值的二进制表示是 ``0.10101010...`` ，在二进制小数点后有无限多位。因此，:math:`2 / 3` 必须在可以表示为有限精度的浮点数之前舍入。舍入规则和模式在 IEEE-754 中指定。最常用的模式是*向偶数舍入*，缩写为向最近舍入。

.. _normal-and-subnormal-values:

5.5.1.2. 正规和次正规值
^^^^^^^^^^^^^^^^^^^^^^^

任何指数字段既不全为零也不全为一的浮点值都称为*正规*值。

浮点值的一个重要方面是最小可表示的正正规数 ``FLT_MIN`` 与零之间的巨大差距。这个差距比 ``FLT_MIN`` 与第二小正规数之间的差距大得多。

浮点*次正规*数（也称为*非正规数*）是为了解决这个问题而引入的。次正规浮点值用指数中所有位都设置为零且有效数字中至少设置一位来表示。次正规数是 IEEE-754 浮点标准的必要部分。

次正规数允许逐渐丢失精度，作为突然向零舍入的替代方案。然而，次正规数的计算成本更高。因此，不需要严格精度的应用程序可能会选择避免它们以提高性能。 ``nvcc`` 编译器允许通过设置 ``-ftz=true`` 选项（刷新为零）来禁用次正规数，这也包含在 ``--use_fast_math`` 中。

单精度下最小正规值和次正规值编码的简化图示如下图所示：

.. figure:: /_static/images/floating-point-subnormal.drawio.png
   :alt: 最小正规值和次正规值表示
   :align: center

其中 X 表示 0 和 1。

.. _special-values:

5.5.1.3. 特殊值
^^^^^^^^^^^^^^^

IEEE-754 标准为浮点数定义了三种特殊值：

**零：**

- 数学零。
- 注意浮点零有两种可能的表示： ``+0`` 和 ``-0`` 。这与整数零的表示不同。
- ``+0 == -0`` 计算为 ``true`` 。
- 零用指数和有效数字中所有位都设置为 0 来编码。

**无穷大：**

- 浮点数根据饱和算术行为，其中超出可表示范围的操作结果为 ``+Infinity`` 或 ``-Infinity`` 。
- 无穷大用指数中所有位都设置为 1 且有效数字中所有位都设置为 0 来编码。无穷大值正好有两个编码。
- 涉及无穷大和非零有限值的算术运算通常结果为无穷大。不确定形式如 ``Inf * 0.0`` 、 ``Inf - Inf`` 、 ``Inf / Inf`` 和 ``0.0 / 0.0`` 结果为 NaN。

**非数字（NaN）：**

- NaN 是表示未定义或不可表示值的特殊符号。常见示例是 ``0.0 / 0.0`` 、 ``sqrt(-1.0)`` 或 ``+Inf - Inf`` 。
- NaN 用指数中所有位都设置为 1 且有效数字中有任何位模式（除了所有位都设置为 0）来编码。有 :math:`2^{\mathrm{mantissa} + 1} - 2` 个可能的编码。
- 任何涉及 NaN 的算术运算都将结果为 NaN。
- 任何涉及 NaN 的有序比较（ ``<`` 、 ``<=`` 、 ``>`` 、 ``>=`` 、 ``==`` ）都将结果为 ``false`` ，包括 ``NaN == NaN`` （非自反）。无序比较 ``NaN != NaN`` 返回 ``true`` 。
- NaN 有两种形式：

  - 静默 NaN（ ``qNaN`` ）用于传播无效操作或值导致的错误。无效算术运算通常产生静默 NaN。它们用有效数字的最高有效位设置为 1 来编码。
  - 信号 NaN（ ``sNaN`` ）旨在引发无效操作异常。信号 NaN 通常显式创建。它们用有效数字的最高有效位设置为 0 来编码。
  - 静默和信号 NaN 的确切位模式是实现定义的。CUDA 提供 `cuda::std::numeric_limits<T>::quiet_NaN <https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN.html>`__ 和 `cuda::std::numeric_limits<T>::signaling_NaN <https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN.html>`__ 常量来获取其特殊值。

特殊值编码的简化图示如下图所示：

.. figure:: /_static/images/floating-point-special-values.drawio.png
   :alt: 无穷大和 NaN 的浮点表示
   :align: center

其中 X 表示 0 和 1。

.. _associativity:

5.5.1.4. 结合律
^^^^^^^^^^^^^^^

重要的是要注意，数学算术的规则和属性由于其有限精度而不能直接应用于浮点算术。以下示例显示单精度值 ``A`` 、 ``B`` 和 ``C`` 以及使用不同结合律计算的数学精确值。

.. math::

   \begin{split}\begin{aligned}
   A           &= 2^{1} \times 1.00000000000000000000001 \\
   B           &= 2^{0} \times 1.00000000000000000000001 \\
   C           &= 2^{3} \times 1.00000000000000000000001 \\
   (A + B) + C &= 2^{3} \times 1.01100000000000000000001011 \\
   A + (B + C) &= 2^{3} \times 1.01100000000000000000001011
   \end{aligned}\end{split}

数学上，:math:`(A + B) + C` 等于 :math:`A + (B + C)`。

设 :math:`\mathrm{rn}(x)` 表示对 :math:`x` 的一次舍入步骤。根据 IEEE-754 在向最近舍入模式下以单精度浮点算术执行相同计算，我们得到：

对于参考，上面还计算了数学精确结果。根据 IEEE-754 计算的结果与数学精确结果不同。此外，对应于和 :math:`\mathrm{rn}(\mathrm{rn}(A + B) + C)` 和 :math:`\mathrm{rn}(A + \mathrm{rn}(B + C))` 的结果彼此不同。在这种情况下，:math:`\mathrm{rn}(A + \mathrm{rn}(B + C))` 比键 :math:`\mathrm{rn}(\mathrm{rn}(A + B) + C)` 更接近正确的数学结果。

此示例表明，看似相同的计算可以产生不同的结果，即使所有基本运算都符合 IEEE-754。

.. _fused-multiply-add-fma:

5.5.1.5. 融合乘加（FMA）
^^^^^^^^^^^^^^^^^^^^^^^^

融合乘加（FMA）运算仅用一次舍入步骤计算结果。没有 FMA，结果需要两次舍入步骤：一次用于乘法，一次用于加法。因为 FMA 只使用一次舍入步骤，它产生更准确的结果。

融合乘加运算可能以不同于两个单独运算的方式影响 NaN 的传播。然而，FMA NaN 处理在所有目标上并非普遍相同。具有多个 NaN 操作数的不同实现可能倾向于静默 NaN 或传播一个操作数的有效负载。此外，当存在多个 NaN 操作数时，IEEE-754 不严格要求确定性的有效负载选择顺序。NaN 也可能出现在中间计算中，例如 :math:`\infty \times 0 + 1` 或 :math:`1 \times \infty - \infty`，导致实现定义的 NaN 有效负载。

为清晰起见，首先考虑一个使用十进制算术的示例来说明 FMA 运算如何工作。我们将使用总共五位精度（小数点后四位）计算 :math:`x^2 - 1`。

- 对于 :math:`x = 1.0008`，正确的数学结果是 :math:`x^2 - 1 = 1.60064 \times 10^{-4}`。使用小数点后四位的最接近数字是 :math:`1.6006 \times 10^{-4}`。
- 融合乘加运算仅用一次舍入步骤实现正确结果 :math:`\mathrm{rn}(x \times x - 1) = 1.6006 \times 10^{-4}`。
- 替代方案是分别计算乘法和加法步骤。:math:`x^2 = 1.00160064` 转换为 :math:`\mathrm{rn}(x \times x) = 1.0016`。最终结果是 :math:`\mathrm{rn}(\mathrm{rn}(x \times x) -1) = 1.6000 \times 10^{-4}`。

分别舍入乘法和加法产生的结果偏差 :math:`0.00064`。相应的 FMA 计算仅偏差 :math:`0.00004`，其结果最接近正确的数学答案。结果总结如下：

下面是另一个使用二进制单精度值的示例：

- 分别计算乘法和加法会导致所有精度位丢失，产生 :math:`0`。
- 另一方面，计算 FMA 提供等于数学值的结果。

融合乘加有助于防止相减抵消期间的精度丢失。当添加具有相反符号的相似数量级时，会发生相减抵消。在这种情况下，许多前导位相互抵消，导致有意义位更少。融合乘加在乘法期间计算双宽度乘积。因此，即使在加法期间发生相减抵消，乘积中仍有足够的有效位剩余以产生精确结果。

**CUDA 中的融合乘加支持：**

CUDA 以多种方式为 ``float`` 和 ``double`` 数据类型提供融合乘加运算：

- 使用标志 ``-fmad=true`` 或 ``--use_fast_math`` 编译时的 ``x * y + z`` 。
- ``fma(x, y, z)`` 和 ``fmaf(x, y, z)`` `C 标准库函数 <https://en.cppreference.com/w/c/numeric/math/fma>`__。
- ``__fmaf_[rd, rn, ru, rz]`` 、 ``__fmaf_ieee_[rd, rn, ru, rz]`` 和 ``__fma_[rd, rn, ru, rz]`` `CUDA 数学内建函数 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html>`__。
- ``cuda::std::fma(x, y, z)`` 和 ``cuda::std::fmaf(x, y, z)`` `CUDA C++ 标准库函数 <https://en.cppreference.com/w/cpp/numeric/math/fma.html>`__。

.. _dot-product-example:

5.5.1.6. 点积示例
^^^^^^^^^^^^^^^^^

考虑找到两个短向量 :math:`\vec{a}` 和 :math:`\vec{b}` 的点积的问题，两者都有四个元素。

尽管这个运算在数学上很容易写下来，但在软件中实现它涉及几种可能导致略有不同结果的替代方案。这里提出的所有策略都使用完全符合 IEEE-754 的运算。

**示例算法 1：** 计算点积的最简单方法是使用乘积的顺序和，保持乘法和加法分离。

> 最终结果可以表示为 :math:`((((a_1 \times b_1) + (a_2 \times b_2)) + (a_3 \times b_3)) + (a_4 \times b_4))`。

**示例算法 2：** 使用融合乘加顺序计算点积。

> 最终结果可以表示为 :math:`(a_4 \times b_4) + ((a_3 \times b_3) + ((a_2 \times b_2) + (a_1 \times b_1 + 0)))`。

**示例算法 3：** 使用分治策略计算点积。首先，我们找到向量前半部分和后半部分的点积。然后，我们使用加法组合这些结果。此算法称为"并行算法"，因为两个子问题可以并行计算，因为它们彼此独立。然而，该算法不需要并行实现；它可以用单个线程实现。

> 最终结果可以表示为 :math:`((a_1 \times b_1) + (a_2 \times b_2)) + ((a_3 \times b_3) + (a_4 \times b_4))`。

.. _rounding:

5.5.1.7. 舍入
^^^^^^^^^^^^^

IEEE-754 标准要求支持几种运算。这些包括算术运算如加法、减法、乘法、除法、平方根、融合乘加、求余数、转换、缩放、符号和比较运算。对于给定的格式和舍入模式，这些运算的结果保证在标准的所有实现中一致。

**舍入模式**

IEEE-754 标准定义了四种舍入模式：*向最近舍入*、*向正无穷舍入*、*向负无穷舍入*和*向零舍入*。CUDA 支持所有四种模式。默认情况下，运算使用*向最近舍入*。`内建数学函数 <#mathematical-functions-appendix-intrinsic-functions>`__ 可用于为单个运算选择其他舍入模式。

.. list-table:: 舍入模式
   :header-rows: 1
   :widths: 20 60

   * - 舍入模式
     - 解释
   * - ``rn``
     - 向最近舍入，平局向偶数
   * - ``rz``
     - 向零舍入
   * - ``ru``
     - 向 :math:`\infty` 舍入
   * - ``rd``
     - 向 :math:`-\infty` 舍入

.. _notes-on-host-device-computation-accuracy:

5.5.1.8. 主机/设备计算精度说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

浮点计算结果的精度受多种因素影响。本节总结了在浮点计算中获得可靠结果的重要考虑因素。其中一些方面在前面章节中有更详细的描述。

这些方面在比较 CPU 和 GPU 之间的结果时也很重要。主机和设备执行之间的差异必须仔细解释。存在差异并不一定意味着 GPU 的结果不正确或 GPU 有问题。

**结合律**：

> 有限精度中的浮点加法和乘法不是`可结合的 <#associativity>`__，因为它们经常产生无法直接以目标格式表示的数学值，需要舍入。评估这些运算的顺序会影响舍入误差的累积方式，并可能显著改变最终结果。

**融合乘加**：

> `融合乘加 <#fused-multiply-add>`__ 在单个运算中计算 :math:`a \times b + c`，从而产生更高的精度和更快的执行时间。最终结果的精度可能受其使用影响。融合乘加依赖于硬件支持，可以通过调用相关函数显式启用或通过编译器优化标志隐式启用。

**精度**：

> 增加浮点精度可能潜在地提高结果的精度。更高的精度减少了有效数字的丢失，并能够表示更广泛的值范围。然而，更高精度类型的吞吐量更低，消耗更多寄存器。此外，使用它们显式存储输入和输出会增加内存使用和数据移动。

**编译器标志和优化**：

> 所有主要编译器都提供各种优化标志来控制浮点运算的行为。
>
> - GCC（ ``-O3`` ）、Clang（ ``-O3`` ）、nvcc（ ``-O3`` ）和 Microsoft Visual Studio（ ``/O2`` ）的最高优化级别不影响浮点语义。然而，内联、循环展开、向量化和公共子表达式消除可能会影响结果。NVC++ 编译器还需要标志 ``-Kieee -Mnofma`` 才能实现 IEEE-754 兼容语义。
> - 请参阅 `GCC <https://gcc.gnu.org/wiki/FloatingPointMath>`__、`Clang <https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior>`__、`Microsoft Visual Studio 编译器 <https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior>`__、`nvc++ <https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu>`__ 和 `Arm C/C++ 编译器 <https://developer.arm.com/documentation/101458/2404/Compiler-options?lang=en>`__ 文档，获取影响浮点行为的选项的详细信息。
> - 另请参阅 ``nvcc`` `用户手册 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math>`__，获取专门影响 CUDA 设备代码中浮点行为的编译器标志的详细描述： ``-ftz`` 、 ``-prec-div`` 、 ``-prec-sqrt`` 、 ``-fmad`` 、 ``--use_fast_math`` 。除了这些浮点选项外，在用户程序的上下文中验证其他编译器优化的效果也很重要。鼓励用户通过广泛的测试验证其结果的正确性，并比较启用优化与禁用所有设备代码优化时获得的结果；另请参阅 ``-G`` 编译器标志。

**库实现**：

> IEEE-754 标准之外定义的函数不能保证正确舍入，并且取决于实现定义的行为。因此，结果可能因不同平台而异，包括主机、设备和不同设备架构之间。

**确定性结果**：

> 确定性结果是指在相同指定条件下使用相同输入运行时每次计算相同的逐位数值输出。这些条件包括：
>
> - 硬件依赖性，例如在同一 CPU 处理器或 GPU 设备上执行。
> - 编译器方面，例如编译器版本和`编译器标志和优化 <#compiler-flags-and-optimizations>`__。
> - 影响计算的运行时条件，例如 :ref:`舍入模式 <rounding>` 或环境变量。
> - 计算的相同输入。
> - 线程配置，包括参与计算的线程数量及其组织，例如块和 grid 大小。
> - `算术原子操作 <cpp-language-extensions.html#atomic-functions>`__ 的排序取决于硬件调度，这可能因运行而异。

**利用 CUDA 库**：

> `CUDA 数学库 <https://developer.nvidia.com/gpu-accelerated-libraries>`__、`C 标准库数学函数 <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 和 `C++ 标准库数学函数 <https://nvidia.github.io/cccl/libcudacxx/standard_api.html>`__ 旨在为常见功能提高开发人员生产力，特别是对于浮点数学和数值密集型例程。这些功能提供一致的高级接口，经过优化，并在平台和边缘情况下广泛测试。鼓励用户充分利用这些库，避免繁琐的手动重新实现。

.. _floating-point-data-types:

5.5.2. 浮点数据类型
-------------------

CUDA 支持 Bfloat16、半精度、单精度、双精度和四精度浮点数据类型。下表总结了 CUDA 中支持的浮点数据类型及其要求。

.. list-table:: 支持的浮点类型
   :header-rows: 1
   :widths: 15 15 10 25 35

   * - 精度/名称
     - 数据类型
     - IEEE-754
     - 头文件/内建
     - 要求
   * - Bfloat16
     - ``__nv_bfloat16``
     - ❌
     - ``<cuda_bf16.h>``
     - 计算能力 8.0 或更高。
   * - 半精度
     - ``__half``
     - ✅
     - ``<cuda_fp16.h>``
     -
   * - 单精度
     - ``float``
     - ✅
     - 内建
     -
   * - 双精度
     - ``double``
     - ✅
     - 内建
     -
   * - 四精度
     - ``__float128``/``_Float128``
     - ✅
     - 内建

       数学函数需要 ``<crt/device_fp128_functions.h>``
     - 主机编译器支持以及计算能力 10.0 或更高。

       C 或 C++ 拼写（分别为 ``_Float128`` 和 ``__float128`` ）也取决于主机编译器支持。

CUDA 还支持 `TensorFloat-32 <https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/>`__（ ``TF32`` ）、`微缩放（MX） <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__ 浮点类型和其他 `低精度数值格式 <https://resources.nvidia.com/en-us-blackwell-architecture>`__，这些不用于通用计算，而是用于涉及张量核心的专用目的。这些包括 4、6 和 8 位浮点类型。有关更多详细信息，请参阅 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html>`__。

下图列出了所支持的浮点数据类型的尾数和指数大小。

.. figure:: /_static/images/floating-point.drawio.png
   :alt: 所支持的浮点数据类型的尾数和指数大小
   :align: center

下表列出了所支持的浮点数据类型的范围。

.. list-table:: 支持的浮点类型属性
   :header-rows: 1
   :widths: 14 14 14 14 14 15 15

   * - 精度/名称
     - 最大值
     - 最大值
     - 最小正值
     - 最小正值
     - 最小正次正规值
     - Epsilon
   * - Bfloat16
     - :math:`\approx 2^{128}`
     - :math:`\approx 3.39 \cdot 10^{38}`
     - :math:`2^{-126}`
     - :math:`\approx 1.18 \cdot 10^{-38}`
     - :math:`2^{-133}`
     - :math:`2^{-7}`
   * - 半精度
     - :math:`\approx 2^{16}`
     - :math:`65504`
     - :math:`2^{-14}`
     - :math:`\approx 6.1 \cdot 10^{-5}`
     - :math:`2^{-24}`
     - :math:`2^{-10}`
   * - 单精度
     - :math:`\approx 2^{128}`
     - :math:`\approx 3.40 \cdot 10^{38}`
     - :math:`2^{-126}`
     - :math:`\approx 1.18 \cdot 10^{-38}`
     - :math:`2^{-149}`
     - :math:`2^{-23}`
   * - 双精度
     - :math:`\approx 2^{1024}`
     - :math:`\approx 1.8 \cdot 10^{308}`
     - :math:`2^{-1022}`
     - :math:`\approx 2.22 \cdot 10^{-308}`
     - :math:`2^{-1074}`
     - :math:`2^{-52}`
   * - 四精度
     - :math:`\approx 2^{16384}`
     - :math:`\approx 1.19 \cdot 10^{4932}`
     - :math:`2^{-16382}`
     - :math:`\approx 3.36 \cdot 10^{-4932}`
     - :math:`2^{-16494}`
     - :math:`2^{-112}`

.. hint::

   `CUDA C++ 标准库 <cpp-language-support.html#cpp-standard-library>`__ 在 ``<cuda/std/limits>`` 头文件中提供 ``cuda::std::numeric_limits`` 来查询支持的浮点类型的属性和范围，包括 `微缩放格式（MX） <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__ 。有关可查询属性的列表，请参阅 `C++ 参考 <https://en.cppreference.com/w/cpp/types/numeric_limits.html>`__。

**复数支持：**

- `CUDA C++ 标准库 <cpp-language-support.html#cpp-standard-library>`__ 在 ``<cuda/std/complex>`` 头文件中通过 `cuda::std::complex <https://en.cppreference.com/w/cpp/numeric/complex>`__ 类型支持复数。有关更多详细信息，另请参阅 `libcu++ 文档 <https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html>`__。

- CUDA 还在 ``cuComplex.h`` 头文件中通过 ``cuComplex`` 和 ``cuDoubleComplex`` 类型提供对复数的基本支持。

.. _cuda-and-ieee-754-compliance:

5.5.3. CUDA 和 IEEE-754 兼容性
------------------------------

所有 GPU 设备都遵循 `IEEE 754-2019 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__ 二进制浮点运算标准，但有以下限制：

- 没有动态可配置的舍入模式；然而，大多数运算支持多种常量 IEEE 舍入模式，可通过特定命名的`设备内建函数 <#mathematical-functions-appendix-intrinsic-functions>`__ 选择。

- 没有检测浮点异常的机制，因此所有运算的行为就像 IEEE-754 异常总是被屏蔽一样。如果发生异常事件，则传递 IEEE-754 定义的默认屏蔽响应。因此，虽然支持信号 NaN（ ``SNaN`` ）编码，但它们不是信号的，而是作为静默异常处理。

- 浮点运算可能会更改输入 NaN 有效负载的位模式。绝对值和取反等运算也可能不符合 IEEE 754 要求，这可能导致 NaN 的符号以实现定义的方式更新。

为了最大化结果的可移植性，建议用户使用 ``nvcc`` 编译器浮点选项的默认设置： ``-ftz=false`` 、 ``-prec-div=true`` 和 ``-prec-sqrt=true`` ，并且不使用 ``--use_fast_math`` 选项。请注意，浮点表达式重关联和收缩默认是允许的，类似于 ``--fmad=true`` 选项。另请参阅 ``nvcc`` `用户手册 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math>`__ 获取这些编译标志的详细描述。

IEEE-754 和 C/C++ 语言标准没有明确解决在舍入到整数值超出目标整数格式范围的情况下将浮点值转换为整数值的问题。GPU 设备到范围的钳位行为在 `PTX ISA 转换指令 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt>`__ 部分中描述。然而，当超出范围的转换不是直接通过 PTX 指令调用时，编译器优化可能会利用未指定行为条款，从而导致未定义行为和无效的 CUDA 程序。CUDA Math 文档在每个函数/内建基础上向用户发出警告。例如，考虑 `__double2int_rz() <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__double2int_rzd>`__ 指令。这可能不同于主机编译器和库实现的行为方式。

**原子函数次正规值行为**：

原子运算关于浮点次正规值具有以下行为，无论编译器标志 ``-ftz`` 的设置如何：

- 全局内存上的原子单精度浮点加法始终在刷新为零模式下运算，即行为等同于 PTX ``add.rn.ftz.f32`` 语义。

- 共享内存上的原子单精度浮点加法始终支持次正规值，即行为等同于 PTX ``add.rn.f32`` 语义。

.. _cuda-and-c-c-compliance:

5.5.4. CUDA 和 C/C++ 兼容性
---------------------------

**浮点异常：**

与主机实现不同，设备代码中支持的数学运算符和函数不设置全局 ``errno`` 变量，也不报告 `浮点异常 <https://en.cppreference.com/w/cpp/numeric/fenv/FE_exceptions>`__ 来指示错误。
因此，如果需要错误诊断机制，用户应为函数实现额外的输入和输出筛选。

**浮点运算的未定义行为：**

数学运算的未定义行为的常见条件包括：

- 数学运算符和函数的无效参数：

  - 使用未初始化的浮点变量。
  - 在其生存期之外使用浮点变量。
  - 有符号整数溢出。
  - 解引用无效指针。

- 浮点特定的未定义行为：

  - 将浮点值转换为结果不可表示的整数类型是未定义行为。这也包括 NaN 和无穷大。

用户有责任确保 CUDA 程序的有效性。无效参数可能导致未定义行为，并受编译器优化影响。

与整数除以零相反，浮点除以零不是未定义行为，也不受编译器优化影响；相反，它是实现特定的行为。
符合 `IEC-60559 <https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559.html>`__ （IEEE-754）的 C++ 实现（包括 CUDA）产生无穷大。
请注意，无效浮点运算产生 NaN，不应误解为未定义行为。示例包括零除以零和无穷大除以无穷大。

**浮点字面量可移植性：**

C 和 C++ 都允许以十进制或十六进制表示法表示浮点值。
`C99 <https://en.cppreference.com/w/c/language/floating_constant.html>`__ 和 `C++17 <https://en.cppreference.com/w/cpp/language/floating_literal.html>`__ 支持的十六进制浮点字面量以科学记数法表示实数值，可以精确地以基数为 2 表示。
然而，这并不保证字面量将映射到目标变量中存储的实际值（见下一段）。
相反，十进制浮点字面量可能表示无法以基数为 2 表示的数值。

根据 `C++ 标准规则 <https://eel.is/c++draft/lex.fcon#3>`__，十六进制和十进制浮点字面量舍入到最接近的可表示值，较大或较小，以实现定义的方式选择。此舍入行为可能在主机和设备之间不同。

相同浮点表达式的运行时和编译时评估受以下可移植性问题影响：

- 浮点表达式的运行时评估可能受选定的舍入模式、浮点收缩（FMA）和重关联编译器设置以及浮点异常的影响。
  请注意，CUDA 不支持浮点异常， :ref:`舍入模式 <rounding>` 默认设置为 *向偶数舍入* 。
  可以使用 :ref:`内建函数 <intrinsic-functions>` 选择其他舍入模式。

- 编译器可能使用更高精度的内部表示来处理常量表达式。

- 编译器可能执行优化，如常量折叠、常量传播和公共子表达式消除，这可能导致不同的最终值或比较结果。

.. note::

   有关浮点计算的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>`_。

.. _floating-point-functionality-exposure:

5.5.5. 浮点功能暴露
-------------------

CUDA 支持的数学函数通过以下方式暴露：

:ref:`内建 C/C++ 语言算术运算符 <built-in-arithmetic-operators>` ：

- ``x + y`` 、 ``x - y`` 、 ``x * y`` 、 ``x / y`` 、 ``x++`` 、 ``x--`` 、 ``x += y`` 、 ``x -= y`` 、 ``x *= y`` 、 ``x /= y`` 。

- 支持单精度、双精度和四精度类型，分别为 ``float`` 、 ``double`` 和 ``__float128/_Float128`` 。

  - 通过分别包含 ``<cuda_fp16.h>`` 和 ``<cuda_bf16.h>`` 头文件，也支持 ``__half`` 和 ``__nv_bfloat16`` 类型。

  - ``__float128/_Float128`` 类型支持依赖于主机编译器和设备计算能力，参阅 :ref:`支持的浮点类型 <floating-point-data-types>` 表。

- 它们在主机和设备代码中均可用。

- 其行为受 ``nvcc`` `优化标志 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math>`__ 影响。

`CUDA C++ 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__ ：

- 通过 ``<cuda/std/cmath>`` 头文件和 ``cuda::std::`` 命名空间暴露完整的 C++ ``<cmath>`` `头文件函数 <https://en.cppreference.com/w/cpp/header/cmath>`__ 。

- 支持 IEEE-754 标准浮点类型 ``__half`` 、 ``float`` 、 ``double`` 、 ``__float128`` ，以及 Bfloat16 ``__nv_bfloat16`` 。

  - ``__float128`` 支持依赖于主机编译器和设备计算能力，参阅 :ref:`支持的浮点类型 <floating-point-data-types>` 表。

- 它们在主机和设备代码中均可用。

- 它们通常依赖 `CUDA Math API 函数 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html>`__ 。因此，主机和设备代码之间可能存在不同的精度级别。

- 其行为受 ``nvcc`` :ref:`优化标志 <optimization-options>` 影响。

- 根据 C++23 和 C++26 标准规范，一部分功能也在常量表达式（例如 ``constexpr`` 函数）中受支持。

`CUDA C 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__ （ `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ ）：

- 暴露 C ``<math.h>`` `头文件函数 <https://en.cppreference.com/w/c/header/math.html>`__ 的子集。

- 支持单精度和双精度类型，分别为 ``float`` 和 ``double`` 。

  - 它们在主机和设备代码中均可用。

  - 它们不需要额外的头文件。

  - 其行为受 ``nvcc`` :ref:`优化标志 <optimization-options>` 影响。

- ``<math.h>`` 头文件功能的一个子集也适用于 ``__half`` 、 ``__nv_bfloat16`` 和 ``__float128/_Float128`` 类型。这些函数的名称与 C 标准库中的名称相似。

  - ``__half`` 和 ``__nv_bfloat16`` 类型分别需要 ``<cuda_fp16.h>`` 和 ``<cuda_bf16.h>`` 头文件。它们在主机和设备代码中的可用性按函数逐一确定。

  - ``__float128/_Float128`` 类型支持依赖于主机编译器和设备计算能力，参阅 :ref:`支持的浮点类型 <floating-point-data-types>` 表。相关函数需要 ``crt/device_fp128_functions.h`` 头文件，并且仅在设备代码中可用。

- 它们在主机和设备代码之间可能具有不同的精度。

:ref:`非标准 CUDA 数学函数 <non-standard-cuda-mathematical-functions>` （ `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ ）：

- 暴露不属于 C/C++ 标准库的数学功能。

- 主要支持单精度和双精度类型，分别为 ``float`` 和 ``double`` 。

  - 它们在主机和设备代码中的可用性按函数逐一确定。

  - 它们不需要额外的头文件。

  - 它们在主机和设备代码之间可能具有不同的精度。

- ``__nv_bfloat16`` 、 ``__half`` 、 ``__float128/_Float128`` 仅在有限的函数集合中受支持。

  - ``__half`` 和 ``__nv_bfloat16`` 类型分别需要 ``<cuda_fp16.h>`` 和 ``<cuda_bf16.h>`` 头文件。

  - ``__float128/_Float128`` 类型支持依赖于主机编译器和设备计算能力，参阅 :ref:`支持的浮点类型 <floating-point-data-types>` 表。相关函数需要 ``crt/device_fp128_functions.h`` 头文件。

  - 它们仅在设备代码中可用。

- 其行为受 ``nvcc`` :ref:`优化标志 <optimization-options>` 影响。

`内建数学函数 <#mathematical-functions-appendix-intrinsic-functions>`__ （ `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ ）：

- 支持单精度和双精度类型，分别为 ``float`` 和 ``double`` 。

- 它们仅在设备代码中可用。

- 它们比相应的 `CUDA Math API 函数 <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 更快，但精度更低。

- 其行为不受 ``nvcc`` :ref:`浮点优化标志 <optimization-options>` ``-prec-div=false`` 、 ``-prec-sqrt=false`` 和 ``-fmad=true`` 的影响。唯一的例外是 ``-ftz=true`` ，它也包含在 ``-use_fast_math`` 中。

.. list-table:: 数学函数功能特性总结
   :header-rows: 1
   :widths: 22 33 9 9 27

   * - 功能
     - 支持的类型
     - 主机
     - 设备
     - | 受浮点优化标志影响
       | （仅针对 ``float`` 和 ``double`` ）
   * - :ref:`内建 C/C++ 语言算术运算符 <built-in-arithmetic-operators>`
     - | ``float``
       | ``double``
       | ``__half``
       | ``__nv_bfloat16``
       | ``__float128/_Float128``
       | ``cuda::std::complex``
     - ✅
     - ✅
     - ✅
   * - `CUDA C++ 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__
     - ``float`` 、 ``double`` 、 ``__half`` 、 ``__nv_bfloat16`` 、 ``__float128`` 、 ``cuda::std::complex``
     - ✅
     - ✅
     - ✅
   * - `CUDA C++ 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__
     - ``__nv_fp8_e4m3`` 、 ``__nv_fp8_e5m2`` 、 ``__nv_fp8_e8m0`` 、 ``__nv_fp6_e2m3`` 、 ``__nv_fp6_e3m2`` 、 ``__nv_fp4_e2m1`` **\***
     - ✅
     - ✅
     - ✅
   * - `CUDA C 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__
     - ``float`` 、 ``double``
     - ✅
     - ✅
     - ✅
   * - `CUDA C 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__
     - ``__nv_bfloat16`` 、 ``__half`` ，有限支持且名称相似
     - 按函数逐一确定
     - 按函数逐一确定
     - ✅
   * - `CUDA C 标准库数学函数 <#mathematical-functions-appendix-cxx-standard-functions>`__
     - ``__float128/_Float128`` ，有限支持且名称相似
     - ❌
     - ✅
     - ✅
   * - :ref:`非标准 CUDA 数学函数 <non-standard-cuda-mathematical-functions>`
     - ``float`` 、 ``double``
     - 按函数逐一确定
     - 按函数逐一确定
     - ✅
   * - :ref:`非标准 CUDA 数学函数 <non-standard-cuda-mathematical-functions>`
     - ``__nv_bfloat16`` 、 ``__half`` 、 ``__float128/_Float128`` ，有限支持
     - ❌
     - ✅
     - ✅
   * - `内建函数 <#mathematical-functions-appendix-intrinsic-functions>`__
     - ``float`` 、 ``double``
     - ❌
     - ✅
     - 仅在 ``-ftz=true`` 时，也包含在 ``-use_fast_math`` 中

**\*** `CUDA C++ 标准库函数 <cpp-language-support.html#cpp-standard-library>`__ 支持对小浮点类型的查询，例如 `numeric_limits\<T\> <https://en.cppreference.com/w/cpp/types/numeric_limits.html>`__ 、 `fpclassify() <https://en.cppreference.com/w/cpp/numeric/math/fpclassify>`__ 、 `isfinite() <https://en.cppreference.com/w/cpp/numeric/math/isfinite.html>`__ 、 `isnormal() <https://en.cppreference.com/w/cpp/numeric/math/isnormal.html>`__ 、 `isinf() <https://en.cppreference.com/w/cpp/numeric/math/isinf.html>`__ 和 `isnan() <https://en.cppreference.com/w/cpp/numeric/math/isnan.html>`__ 。

以下各节在适用时提供其中一些函数的精度信息，并使用 ULP 进行量化。有关 `最后一位单位（ULP） <https://en.wikipedia.org/wiki/Unit_in_the_last_place>`__ 定义的更多信息，请参阅 Jean-Michel Muller 的论文 `On the definition of ulp(x) <https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf>`__ 。

.. _built-in-arithmetic-operators:

5.5.6. 内建算术运算符
---------------------

内建 C/C++ 语言运算符，例如 ``x + y`` 、 ``x - y`` 、 ``x * y`` 、 ``x / y`` 、 ``x++`` 、 ``x--`` 以及倒数 ``1 / x`` ，对于单精度、双精度和四精度类型符合 IEEE-754 标准。它们在使用*向偶数舍入*舍入模式时保证最大 ULP 误差为零。它们在主机和设备代码中均可用。

``nvcc`` 编译标志 ``-fmad=true`` （也包含在 ``--use_fast_math`` 中）启用将浮点乘法和加法/减法收缩为浮点乘加运算，并对单精度类型 ``float`` 的最大 ULP 误差产生以下影响：

- ``x * y + z`` → `__fmaf_rn(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff>`__ ：0 ULP

``nvcc`` 编译标志 ``-prec-div=false`` （也包含在 ``--use_fast_math`` 中）对单精度类型 ``float`` 的除法运算符 ``/`` 的最大 ULP 误差产生以下影响：

- ``x / y`` → `__fdividef(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__cuda__math__intrinsic__single_1gac996beec34f94f6376d0674a6860e107>`__ ：2 ULP
- ``1 / x`` ：1 ULP

.. _cuda-c-mathematical-standard-library-functions:

5.5.7. CUDA C++ 标准库数学函数
------------------------------

CUDA 通过 ``cuda::std::`` 命名空间为 `C++ 标准库数学函数 <https://en.cppreference.com/w/cpp/header/cmath.html>`__ 提供全面支持。这些功能是 ``<cuda/std/cmath>`` 标头的一部分。它们在主机和设备代码中均可用。

以下各节规定了与 `CUDA Math APIs <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 的映射，以及每个函数在设备上执行时的误差界限。

- 最大 ULP 误差表述为：函数返回值与按照*向偶数舍入*（round-to-nearest ties-to-even）舍入模式获得的相应精度的正确舍入结果之间，以 ULP 计的差异的最大观测绝对值。

- 误差界限来自广泛但并非详尽的测试。因此，它们无法得到保证。

.. _basic-operations:

5.5.7.1. 基本运算
^^^^^^^^^^^^^^^^^

基本运算的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 在主机和设备代码中均可用，但 ``__float128`` 除外。

以下所有函数的最大 ULP 误差均为零。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射 —— 基本运算
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `fabs(x) <https://en.cppreference.com/w/cpp/numeric/math/fabs.html>`__
     - :math:`|x|`
     - `__habs(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#_CPPv46__habsK13__nv_bfloat16>`__
     - `__habs(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#_CPPv46__habsK6__half>`__
     - `fabsf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fabsff>`__
     - `fabs(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fabsd>`__
     - `__nv_fp128_fabs(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fabsg>`__
   * - `fmod(x, y) <https://en.cppreference.com/w/cpp/numeric/math/fmod.html>`__
     - :math:`\dfrac{x}{y}` 的余数，按 :math:`x - \mathrm{trunc}\left(\dfrac{x}{y}\right) \cdot y` 计算
     - N/A
     - N/A
     - `fmodf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmodfff>`__
     - `fmod(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmoddd>`__
     - `__nv_fp128_fmod(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmodgg>`__
   * - `remainder(x, y) <https://en.cppreference.com/w/cpp/numeric/math/remainder.html>`__
     - :math:`\dfrac{x}{y}` 的余数，按 :math:`x - \mathrm{rint}\left(\dfrac{x}{y}\right) \cdot y` 计算
     - N/A
     - N/A
     - `remainderf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410remainderfff>`__
     - `remainder(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49remainderdd>`__
     - `__nv_fp128_remainder(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv420__nv_fp128_remaindergg>`__
   * - `remquo(x, y, iptr) <https://en.cppreference.com/w/cpp/numeric/math/remquo.html>`__
     - :math:`\dfrac{x}{y}` 的余数和商
     - N/A
     - N/A
     - `remquof(x, y, iptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47remquofffPi>`__
     - `remquo(x, y, iptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46remquoddPi>`__
     - N/A
   * - `fma(x, y, z) <https://en.cppreference.com/w/cpp/numeric/math/fma.html>`__
     - :math:`x \cdot y + z`
     - `__hfma(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#_CPPv46__hfmaK13__nv_bfloat16K13__nv_bfloat16K13__nv_bfloat16>`__ ，仅设备端
     - `__hfma(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#_CPPv46__hfmaK6__halfK6__halfK6__half>`__ ，仅设备端
     - `fmaf(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff>`__
     - `fma(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43fmaddd>`__
     - `__nv_fp128_fma(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_fmaggg>`__
   * - `fmax(x, y) <https://en.cppreference.com/w/cpp/numeric/math/fmax.html>`__
     - :math:`\max(x, y)`
     - `__hmax(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv46__hmaxK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hmax(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv46__hmaxK6__halfK6__half>`__
     - `fmaxf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmaxfff>`__
     - `fmax(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmaxdd>`__
     - `__nv_fp128_fmax(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmaxgg>`__
   * - `fmin(x, y) <https://en.cppreference.com/w/cpp/numeric/math/fmin.html>`__
     - :math:`\min(x, y)`
     - `__hmin(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv46__hminK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hmin(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv46__hminK6__halfK6__half>`__
     - `fminf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fminfff>`__
     - `fmin(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmindd>`__
     - `__nv_fp128_fmin(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmingg>`__
   * - `fdim(x, y) <https://en.cppreference.com/w/cpp/numeric/math/fdim.html>`__
     - :math:`\max(x-y, 0)`
     - N/A
     - N/A
     - `fdimf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fdimfff>`__
     - `fdim(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fdimdd>`__
     - `__nv_fp128_fdim(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fdimgg>`__
   * - `nan(str) <https://en.cppreference.com/w/cpp/numeric/math/nan.html>`__
     - 由字符串表示生成的 NaN 值
     - N/A
     - N/A
     - `nanf(str) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44nanfPKc>`__
     - `nan(str) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43nanPKc>`__
     - N/A

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

.. _exponential-functions:

5.5.7.2. 指数函数
^^^^^^^^^^^^^^^^^

指数函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 仅对 ``float`` 和 ``double`` 类型在主机和设备代码中可用。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射与精度（最大 ULP） —— 指数函数
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `exp(x) <https://en.cppreference.com/w/cpp/numeric/math/exp.html>`__
     - :math:`e^x`
     - `hexp(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hexpK13__nv_bfloat16>`__

       0 ULP
     - `hexp(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hexpK6__half>`__

       0 ULP
     - `expf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff>`__

       2 ULP
     - `exp(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43expd>`__

       1 ULP
     - `__nv_fp128_exp(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_expg>`__

       1 ULP
   * - `exp2(x) <https://en.cppreference.com/w/cpp/numeric/math/exp2.html>`__
     - :math:`2^x`
     - `hexp2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hexp2K13__nv_bfloat16>`__

       0 ULP
     - `hexp2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hexp2K6__half>`__

       0 ULP
     - `exp2f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45exp2ff>`__

       2 ULP
     - `exp2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44exp2d>`__

       1 ULP
     - `__nv_fp128_exp2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_exp2g>`__

       1 ULP
   * - `expm1(x) <https://en.cppreference.com/w/cpp/numeric/math/expm1.html>`__
     - :math:`e^x - 1`
     - N/A
     - N/A
     - `expm1f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46expm1ff>`__

       1 ULP
     - `expm1(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45expm1d>`__

       1 ULP
     - `__nv_fp128_expm1(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_expm1g>`__

       1 ULP
   * - `log(x) <https://en.cppreference.com/w/cpp/numeric/math/log.html>`__
     - :math:`\ln(x)`
     - `hlog(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hlogK13__nv_bfloat16>`__

       0 ULP
     - `hlog(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hlogK6__half>`__

       0 ULP
     - `logf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff>`__

       1 ULP
     - `log(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43logd>`__

       1 ULP
     - `__nv_fp128_log(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_logg>`__

       1 ULP
   * - `log10(x) <https://en.cppreference.com/w/cpp/numeric/math/log10.html>`__
     - :math:`\log_{10}(x)`
     - `hlog10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hlog10K13__nv_bfloat16>`__

       0 ULP
     - `hlog10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hlog10K6__half>`__

       0 ULP
     - `log10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log10ff>`__

       2 ULP
     - `log10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45log10d>`__

       1 ULP
     - `__nv_fp128_log10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_log10g>`__

       1 ULP
   * - `log2(x) <https://en.cppreference.com/w/cpp/numeric/math/log2.html>`__
     - :math:`\log_2(x)`
     - `hlog2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hlog2K13__nv_bfloat16>`__

       0 ULP
     - `hlog2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hlog2K6__half>`__

       0 ULP
     - `log2f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45log2ff>`__

       1 ULP
     - `log2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44log2d>`__

       1 ULP
     - `__nv_fp128_log2(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_log2g>`__

       1 ULP
   * - `log1p(x) <https://en.cppreference.com/w/cpp/numeric/math/log1p.html>`__
     - :math:`\ln(1+x)`
     - N/A
     - N/A
     - `log1pf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log1pff>`__

       1 ULP
     - `log1p(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45log1pd>`__

       1 ULP
     - `__nv_fp128_log1p(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_log1pg>`__

       1 ULP

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

.. _power-functions:

5.5.7.3. 幂函数
^^^^^^^^^^^^^^^

幂函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 仅对 ``float`` 和 ``double`` 类型在主机和设备代码中可用。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射与精度（最大 ULP） —— 幂函数
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `pow(x, y) <https://en.cppreference.com/w/cpp/numeric/math/pow.html>`__
     - :math:`x^y`
     - N/A
     - N/A
     - `powf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff>`__

       4 ULP
     - `pow(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43powdd>`__

       2 ULP
     - `__nv_fp128_pow(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_powgg>`__

       1 ULP
   * - `sqrt(x) <https://en.cppreference.com/w/cpp/numeric/math/sqrt.html>`__
     - :math:`\sqrt{x}`
     - `hsqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hsqrtK13__nv_bfloat16>`__

       0 ULP
     - `hsqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hsqrtK6__half>`__

       0 ULP
     - `sqrtf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sqrtff>`__

       ▪ 0 ULP

       ▪ 使用 ``--use_fast_math`` 时为 1 ULP
     - `sqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sqrtd>`__

       0 ULP
     - `__nv_fp128_sqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sqrtg>`__

       0 ULP
   * - `cbrt(x) <https://en.cppreference.com/w/cpp/numeric/math/cbrt.html>`__
     - :math:`\sqrt[3]{x}`
     - N/A
     - N/A
     - `cbrtf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45cbrtff>`__

       1 ULP
     - `cbrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44cbrtd>`__

       1 ULP
     - N/A
   * - `hypot(x, y) <https://en.cppreference.com/w/cpp/numeric/math/hypot.html>`__
     - :math:`\sqrt{x^2 + y^2}`
     - N/A
     - N/A
     - `hypotf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46hypotfff>`__

       3 ULP
     - `hypot(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45hypotdd>`__

       2 ULP
     - `__nv_fp128_hypot(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_hypotgg>`__

       1 ULP

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

.. _trigonometric-functions:

5.5.7.4. 三角函数
^^^^^^^^^^^^^^^^^

三角函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 仅对 ``float`` 和 ``double`` 类型在主机和设备代码中可用。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射与精度（最大 ULP） —— 三角函数
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `sin(x) <https://en.cppreference.com/w/cpp/numeric/math/sin.html>`__
     - :math:`\sin(x)`
     - `hsin(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hsinK13__nv_bfloat16>`__

       0 ULP
     - `hsin(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hsinK6__half>`__

       0 ULP
     - `sinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44sinff>`__

       2 ULP
     - `sin(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43sind>`__

       2 ULP
     - `__nv_fp128_sin(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_sing>`__

       1 ULP
   * - `cos(x) <https://en.cppreference.com/w/cpp/numeric/math/cos.html>`__
     - :math:`\cos(x)`
     - `hcos(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hcosK13__nv_bfloat16>`__

       0 ULP
     - `hcos(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hcosK6__half>`__

       0 ULP
     - `cosf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44cosff>`__

       2 ULP
     - `cos(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43cosd>`__

       2 ULP
     - `__nv_fp128_cos(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_cosg>`__

       1 ULP
   * - `tan(x) <https://en.cppreference.com/w/cpp/numeric/math/tan.html>`__
     - :math:`\tan(x)`
     - N/A
     - N/A
     - `tanf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44tanff>`__

       4 ULP
     - `tan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43tand>`__

       2 ULP
     - `__nv_fp128_tan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_tang>`__

       1 ULP
   * - `asin(x) <https://en.cppreference.com/w/cpp/numeric/math/asin.html>`__
     - :math:`\sin^{-1}(x)`
     - N/A
     - N/A
     - `asinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45asinff>`__

       2 ULP
     - `asin(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44asind>`__

       2 ULP
     - `__nv_fp128_asin(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_asing>`__

       1 ULP
   * - `acos(x) <https://en.cppreference.com/w/cpp/numeric/math/acos.html>`__
     - :math:`\cos^{-1}(x)`
     - N/A
     - N/A
     - `acosf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45acosff>`__

       2 ULP
     - `acos(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44acosd>`__

       2 ULP
     - `__nv_fp128_acos(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_acosg>`__

       1 ULP
   * - `atan(x) <https://en.cppreference.com/w/cpp/numeric/math/atan.html>`__
     - :math:`\tan^{-1}(x)`
     - N/A
     - N/A
     - `atanf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45atanff>`__

       2 ULP
     - `atan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44atand>`__

       2 ULP
     - `__nv_fp128_atan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_atang>`__

       1 ULP
   * - `atan2(y, x) <https://en.cppreference.com/w/cpp/numeric/math/atan2.html>`__
     - :math:`\tan^{-1}\left(\dfrac{y}{x}\right)`
     - N/A
     - N/A
     - `atan2f(y, x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46atan2fff>`__

       3 ULP
     - `atan2(y, x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45atan2dd>`__

       2 ULP
     - N/A

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

.. _hyperbolic-functions:

5.5.7.5. 双曲函数
^^^^^^^^^^^^^^^^^

双曲函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 仅对 ``float`` 和 ``double`` 类型在主机和设备代码中可用。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射与精度（最大 ULP） —— 双曲函数
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `sinh(x) <https://en.cppreference.com/w/cpp/numeric/math/sinh.html>`__
     - :math:`\sinh(x)`
     - N/A
     - N/A
     - `sinhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sinhff>`__

       3 ULP
     - `sinh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sinhd>`__

       2 ULP
     - `__nv_fp128_sinh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sinhg>`__

       1 ULP
   * - `cosh(x) <https://en.cppreference.com/w/cpp/numeric/math/cosh.html>`__
     - :math:`\cosh(x)`
     - N/A
     - N/A
     - `coshf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45coshff>`__

       2 ULP
     - `cosh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44coshd>`__

       1 ULP
     - `__nv_fp128_cosh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_coshg>`__

       1 ULP
   * - `tanh(x) <https://en.cppreference.com/w/cpp/numeric/math/tanh.html>`__
     - :math:`\tanh(x)`
     - `htanh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45htanhK13__nv_bfloat16>`__

       0 ULP
     - `htanh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45htanhK6__half>`__

       0 ULP
     - `tanhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45tanhff>`__

       2 ULP
     - `tanh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44tanhd>`__

       1 ULP
     - `__nv_fp128_tanh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_tanhg>`__

       1 ULP
   * - `asinh(x) <https://en.cppreference.com/w/cpp/numeric/math/asinh.html>`__
     - :math:`\operatorname{sinh}^{-1}(x)`
     - N/A
     - N/A
     - `asinhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46asinhff>`__

       3 ULP
     - `asinh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45asinhd>`__

       3 ULP
     - `__nv_fp128_asinh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_asinhg>`__

       1 ULP
   * - `acosh(x) <https://en.cppreference.com/w/cpp/numeric/math/acosh.html>`__
     - :math:`\operatorname{cosh}^{-1}(x)`
     - N/A
     - N/A
     - `acoshf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46acoshff>`__

       4 ULP
     - `acosh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45acoshd>`__

       3 ULP
     - `__nv_fp128_acosh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_acoshg>`__

       1 ULP
   * - `atanh(x) <https://en.cppreference.com/w/cpp/numeric/math/atanh.html>`__
     - :math:`\operatorname{tanh}^{-1}(x)`
     - N/A
     - N/A
     - `atanhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46atanhff>`__

       3 ULP
     - `atanh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45atanhd>`__

       2 ULP
     - `__nv_fp128_atanh(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_atanhg>`__

       1 ULP

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

.. _error-and-gamma-functions:

5.5.7.6. 误差函数和伽马函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^

误差函数和伽马函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 对 ``float`` 和 ``double`` 类型在主机和设备代码中可用。

误差函数和伽马函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射与精度（最大 ULP） —— 误差函数和伽马函数
   :header-rows: 1
   :widths: 25 35 20 20

   * - ``cuda::std`` 函数
     - 含义
     - ``float``
     - ``double``
   * - `erf(x) <https://en.cppreference.com/w/cpp/numeric/math/erf.html>`__
     - :math:`\dfrac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt`
     - `erff(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44erfff>`__

       2 ULP
     - `erf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43erfd>`__

       2 ULP
   * - `erfc(x) <https://en.cppreference.com/w/cpp/numeric/math/erfc.html>`__
     - :math:`1 - \mathrm{erf}(x)`
     - `erfcf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45erfcff>`__

       4 ULP
     - `erfc(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44erfcd>`__

       5 ULP
   * - `tgamma(x) <https://en.cppreference.com/w/cpp/numeric/math/tgamma.html>`__
     - :math:`\Gamma(x)`
     - `tgammaf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47tgammaff>`__

       5 ULP
     - `tgamma(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46tgammad>`__

       10 ULP
   * - `lgamma(x) <https://en.cppreference.com/w/cpp/numeric/math/lgamma.html>`__
     - :math:`\ln |\Gamma(x)|`
     - `lgammaf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lgammaff>`__

       ▪ 当 :math:`x \notin [-10.001, -2.264]` 时为 6 ULP

       ▪ 否则更大
     - `lgamma(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lgammad>`__

       ▪ 当 :math:`x \notin [-23.0001, -2.2637]` 时为 4 ULP

       ▪ 否则更大

.. _nearest-integer-floating-point-operations:

5.5.7.7. 最近整数浮点运算
^^^^^^^^^^^^^^^^^^^^^^^^^

最近整数浮点运算的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 仅对 ``float`` 和 ``double`` 类型在主机和设备代码中可用。

以下所有函数的最大 ULP 误差均为零。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射 —— 最近整数浮点运算
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `ceil(x) <https://en.cppreference.com/w/cpp/numeric/math/ceil.html>`__
     - :math:`\lceil x \rceil`
     - `hceil(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hceilK13__nv_bfloat16>`__
     - `hceil(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hceilK6__half>`__
     - `ceilf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45ceilff>`__
     - `ceil(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44ceild>`__
     - `__nv_fp128_ceil(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_ceilg>`__
   * - `floor(x) <https://en.cppreference.com/w/cpp/numeric/math/floor.html>`__
     - :math:`\lfloor x \rfloor`
     - `hfloor(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hfloorK13__nv_bfloat16>`__
     - `hfloor(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hfloorK6__half>`__
     - `floorf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46floorff>`__
     - `floor(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45floord>`__
     - `__nv_fp128_floor(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_floorg>`__
   * - `trunc(x) <https://en.cppreference.com/w/cpp/numeric/math/trunc.html>`__
     - 截断为整数
     - `htrunc(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46htruncK13__nv_bfloat16>`__
     - `htrunc(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46htruncK6__half>`__
     - `truncf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46truncff>`__
     - `trunc(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45truncd>`__
     - `__nv_fp128_trunc(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_truncg>`__
   * - `round(x) <https://en.cppreference.com/w/cpp/numeric/math/round.html>`__
     - 向最近整数舍入，平局远离零
     - N/A
     - N/A
     - `roundf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46roundff>`__
     - `round(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45roundd>`__
     - `__nv_fp128_round(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_roundg>`__
   * - `nearbyint(x) <https://en.cppreference.com/w/cpp/numeric/math/nearbyint.html>`__
     - 向最近整数舍入，平局向偶数
     - N/A
     - N/A
     - `nearbyintf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410nearbyintff>`__
     - `nearbyint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49nearbyintd>`__
     - N/A
   * - `rint(x) <https://en.cppreference.com/w/cpp/numeric/math/rint.html>`__
     - 向最近整数舍入，平局向偶数
     - `hrint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hrintK13__nv_bfloat16>`__
     - `hrint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hrintK6__half>`__
     - `rintf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45rintff>`__
     - `rint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44rintd>`__
     - `__nv_fp128_rint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_rintg>`__
   * - `lrint(x) <https://en.cppreference.com/w/cpp/numeric/math/rint.html>`__
     - 向最近整数舍入，平局向偶数（返回 ``long int`` ）
     - N/A
     - N/A
     - `lrintf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46lrintff>`__
     - `lrint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45lrintd>`__
     - N/A
   * - `llrint(x) <https://en.cppreference.com/w/cpp/numeric/math/rint.html>`__
     - 向最近整数舍入，平局向偶数（返回 ``long long int`` ）
     - N/A
     - N/A
     - `llrintf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47llrintff>`__
     - `llrint(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46llrintd>`__
     - N/A
   * - `lround(x) <https://en.cppreference.com/w/cpp/numeric/math/round.html>`__
     - 向最近整数舍入，平局远离零（返回 ``long int`` ）
     - N/A
     - N/A
     - `lroundf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lroundff>`__
     - `lround(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lroundd>`__
     - N/A
   * - `llround(x) <https://en.cppreference.com/w/cpp/numeric/math/round.html>`__
     - 向最近整数舍入，平局远离零（返回 ``long long int`` ）
     - N/A
     - N/A
     - `llroundf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48llroundff>`__
     - `llround(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47llroundd>`__
     - N/A

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

**性能注意事项**

将单精度或双精度浮点操作数舍入为整数的推荐方式是使用函数 ``rintf()`` 和 ``rint()`` ，而不是 ``roundf()`` 和 ``round()`` 。这是因为 ``roundf()`` 和 ``round()`` 在设备代码中映射为多条指令，而 ``rintf()`` 和 ``rint()`` 映射为单条指令。 ``truncf()`` 、 ``trunc()`` 、 ``ceilf()`` 、 ``ceil()`` 、 ``floorf()`` 和 ``floor()`` 也各自映射为单条指令。

.. _floating-point-manipulation-functions:

5.5.7.8. 浮点操作函数
^^^^^^^^^^^^^^^^^^^^^

浮点操作函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 在主机和设备代码中均可用，但 ``__float128`` 除外。

浮点操作函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。在这些情况下，通过转换为 ``float`` 类型计算后再将结果转换回来，对这些函数进行模拟。

以下所有函数的最大 ULP 误差均为零。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射 —— 浮点操作函数
   :header-rows: 1
   :widths: 22 30 16 16 16

   * - ``cuda::std`` 函数
     - 含义
     - ``float``
     - ``double``
     - ``__float128``
   * - `frexp(x, exp) <https://en.cppreference.com/w/cpp/numeric/math/frexp.html>`__
     - 提取尾数和指数
     - `frexpf(x, exp) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46frexpffPi>`__
     - `frexp(x, exp) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45frexpdPi>`__
     - `__nv_fp128_frexp(x, nptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_frexpgPi>`__
   * - `ldexp(x, n) <https://en.cppreference.com/w/cpp/numeric/math/ldexp.html>`__
     - :math:`x \cdot 2^{\mathrm{n}}`
     - `ldexpf(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46ldexpffi>`__
     - `ldexp(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45ldexpdi>`__
     - `__nv_fp128_ldexp(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_ldexpgi>`__
   * - `modf(x, iptr) <https://en.cppreference.com/w/cpp/numeric/math/modf.html>`__
     - 提取整数部分和小数部分
     - `modff(x, iptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45modfffPf>`__
     - `modf(x, iptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44modfdPd>`__
     - `__nv_fp128_modf(x, iptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_modfgPg>`__
   * - `scalbn(x, n) <https://en.cppreference.com/w/cpp/numeric/math/scalbn.html>`__
     - :math:`x \cdot 2^n`
     - `scalbnf(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47scalbnffi>`__
     - `scalbn(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46scalbndi>`__
     - N/A
   * - `scalbln(x, n) <https://en.cppreference.com/w/cpp/numeric/math/scalbn.html>`__
     - :math:`x \cdot 2^n`
     - `scalblnf(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48scalblnffl>`__
     - `scalbln(x, n) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47scalblndl>`__
     - N/A
   * - `ilogb(x) <https://en.cppreference.com/w/cpp/numeric/math/ilogb.html>`__
     - :math:`\lfloor \log_2(|x|) \rfloor`
     - `ilogbf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46ilogbff>`__
     - `ilogb(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45ilogbd>`__
     - `__nv_fp128_ilogb(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_ilogbg>`__
   * - `logb(x) <https://en.cppreference.com/w/cpp/numeric/math/logb.html>`__
     - :math:`\lfloor \log_2(|x|) \rfloor`
     - `logbf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45logbff>`__
     - `logb(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44logbd>`__
     - N/A
   * - `nextafter(x, y) <https://en.cppreference.com/w/cpp/numeric/math/nextafter.html>`__
     - 朝 :math:`y` 方向的下一个可表示值
     - `nextafterf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410nextafterfff>`__
     - `nextafter(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49nextafterdd>`__
     - N/A
   * - `copysign(x, y) <https://en.cppreference.com/w/cpp/numeric/math/copysign.html>`__
     - 将 :math:`y` 的符号复制到 :math:`x`
     - `copysignf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv49copysignfff>`__
     - `copysign(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48copysigndd>`__
     - `__nv_fp128_copysign(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv419__nv_fp128_copysigngg>`__

.. _classification-and-comparison:

5.5.7.9. 分类和比较
^^^^^^^^^^^^^^^^^^^

分类和比较函数的 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 在主机和设备代码中均可用，但 ``__float128`` 除外。

以下所有函数的最大 ULP 误差均为零。

.. list-table:: C++ 标准库数学函数 —— C Math API 映射 —— 分类和比较函数
   :header-rows: 1
   :widths: 20 25 11 11 11 11 11

   * - ``cuda::std`` 函数
     - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``float``
     - ``double``
     - ``__float128``
   * - `fpclassify(x) <https://en.cppreference.com/w/cpp/numeric/math/fpclassify.html>`__
     - 对 :math:`x` 进行分类
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
   * - `isfinite(x) <https://en.cppreference.com/w/cpp/numeric/math/isfinite.html>`__
     - 检查 :math:`x` 是否为有限值
     - N/A
     - N/A
     - `isfinite(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48isfinitef>`__
     - `isfinite(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48isfinited>`__
     - N/A
   * - `isinf(x) <https://en.cppreference.com/w/cpp/numeric/math/isinf.html>`__
     - 检查 :math:`x` 是否为无穷大
     - `__hisinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv48__hisinfK13__nv_bfloat16>`__
     - `__hisinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv48__hisinfK6__half>`__
     - `isinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45isinff>`__
     - `isinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45isinfd>`__
     - N/A
   * - `isnan(x) <https://en.cppreference.com/w/cpp/numeric/math/isnan.html>`__
     - 检查 :math:`x` 是否为 NaN
     - `__hisnan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv48__hisnanK13__nv_bfloat16>`__
     - `__hisnan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv48__hisnanK6__half>`__
     - `isnan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45isnanf>`__
     - `isnan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45isnand>`__
     - `__nv_fp128_isnan(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_isnang>`__
   * - `isnormal(x) <https://en.cppreference.com/w/cpp/numeric/math/isnormal.html>`__
     - 检查 :math:`x` 是否为规格化数
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
   * - `signbit(x) <https://en.cppreference.com/w/cpp/numeric/math/signbit.html>`__
     - 检查符号位是否已置位
     - N/A
     - N/A
     - `signbit(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47signbitf>`__
     - `signbit(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47signbitd>`__
     - N/A
   * - `isgreater(x, y) <https://en.cppreference.com/w/cpp/numeric/math/isgreater.html>`__
     - 检查是否 :math:`x > y`
     - `__hgt(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hgtK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hgt(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hgtK6__halfK6__half>`__
     - N/A
     - N/A
     - N/A
   * - `isgreaterequal(x, y) <https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal.html>`__
     - 检查是否 :math:`x \geq y`
     - `__hge(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hgeK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hge(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hgeK6__halfK6__half>`__
     - N/A
     - N/A
     - N/A
   * - `isless(x, y) <https://en.cppreference.com/w/cpp/numeric/math/isless.html>`__
     - 检查是否 :math:`x < y`
     - `__hlt(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hltK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hlt(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hltK6__halfK6__half>`__
     - N/A
     - N/A
     - N/A
   * - `islessequal(x, y) <https://en.cppreference.com/w/cpp/numeric/math/islessequal.html>`__
     - 检查是否 :math:`x \leq y`
     - `__hle(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hleK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hle(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hleK6__halfK6__half>`__
     - N/A
     - N/A
     - N/A
   * - `islessgreater(x, y) <https://en.cppreference.com/w/cpp/numeric/math/islessgreater.html>`__
     - 检查是否 :math:`x < y` 或 :math:`x > y`
     - `__hne(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hneK13__nv_bfloat16K13__nv_bfloat16>`__
     - `__hne(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hneK6__halfK6__half>`__
     - N/A
     - N/A
     - N/A
   * - `isunordered(x, y) <https://en.cppreference.com/w/cpp/numeric/math/isunordered.html>`__
     - 检查 :math:`x` 、 :math:`y` 或两者是否为 NaN
     - N/A
     - N/A
     - N/A
     - N/A
     - `__nv_fp128_isunordered(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv422__nv_fp128_isunorderedgg>`__

**\*** 标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 ``__half`` 和 ``__nv_bfloat16`` ）并非原生可用。

.. _non-standard-cuda-mathematical-functions:

5.5.8. 非标准 CUDA 数学函数
---------------------------

CUDA 提供了不属于 C/C++ 标准库的数学函数，它们作为扩展提供。对于单精度和双精度函数，主机和设备代码的可用性按函数逐一确定。

本节规定每个函数在设备上执行时的误差界限。

- 最大 ULP 误差表述为：函数返回值与按照*向偶数舍入*（round-to-nearest ties-to-even）舍入模式获得的相应精度的正确舍入结果之间，以 ULP 计的差异的最大观测绝对值。

- 误差界限来自广泛但并非详尽的测试。因此，它们无法得到保证。

.. list-table:: 非标准 CUDA 数学函数 ``float`` 和 ``double`` 的映射与精度（最大 ULP）
   :header-rows: 1
   :widths: 20 40 40

   * - 含义
     - ``float``
     - ``double``
   * - :math:`\dfrac{x}{y}`
     - `fdividef(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48fdividefff>`__ ，仅设备端

       0 ULP，与 ``x / y`` 相同
     - N/A
   * - :math:`10^x`
     - `exp10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46exp10ff>`__

       2 ULP
     - `exp10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45exp10d>`__

       1 ULP
   * - :math:`\sqrt{x^2 + y^2 + z^2}`
     - `norm3df(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47norm3dffff>`__ ，仅设备端

       3 ULP
     - `norm3d(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46norm3dddd>`__ ，仅设备端

       2 ULP
   * - :math:`\sqrt{x^2 + y^2 + z^2 + t^2}`
     - `norm4df(x, y, z, t) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47norm4dfffff>`__ ，仅设备端

       3 ULP
     - `norm4d(x, y, z, t) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46norm4ddddd>`__ ，仅设备端

       2 ULP
   * - :math:`\sqrt{\sum_{i=0}^{\mathrm{dim}-1} p_i^{2}}`
     - `normf(dim, p) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45normfiPKf>`__ ，仅设备端

       无法提供误差界限，因为使用了存在舍入精度损失的快速算法
     - `norm(dim, p) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44normiPKd>`__ ，仅设备端

       无法提供误差界限，因为使用了存在舍入精度损失的快速算法
   * - :math:`\dfrac{1}{\sqrt{x}}`
     - `rsqrtf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rsqrtff>`__

       2 ULP
     - `rsqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rsqrtd>`__

       1 ULP
   * - :math:`\dfrac{1}{\sqrt[3]{x}}`
     - `rcbrtf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rcbrtff>`__

       1 ULP
     - `rcbrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rcbrtd>`__

       1 ULP
   * - :math:`\dfrac{1}{\sqrt{x^2 + y^2}}`
     - `rhypotf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47rhypotfff>`__ ，仅设备端

       2 ULP
     - `rhypot(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46rhypotdd>`__ ，仅设备端

       1 ULP
   * - :math:`\dfrac{1}{\sqrt{x^2 + y^2 + z^2}}`
     - `rnorm3df(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48rnorm3dffff>`__ ，仅设备端

       2 ULP
     - `rnorm3d(x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47rnorm3dddd>`__ ，仅设备端

       1 ULP
   * - :math:`\dfrac{1}{\sqrt{x^2 + y^2 + z^2 + t^2}}`
     - `rnorm4df(x, y, z, t) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48rnorm4dfffff>`__ ，仅设备端

       2 ULP
     - `rnorm4d(x, y, z, t) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47rnorm4ddddd>`__ ，仅设备端

       1 ULP
   * - :math:`\dfrac{1}{\sqrt{\sum_{i=0}^{\mathrm{dim}-1} p_i^{2}}}`
     - `rnormf(dim, p) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rnormfiPKf>`__ ，仅设备端

       无法提供误差界限，因为使用了存在舍入精度损失的快速算法
     - `rnorm(dim, p) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rnormiPKd>`__ ，仅设备端

       无法提供误差界限，因为使用了存在舍入精度损失的快速算法
   * - :math:`\cos(\pi x)`
     - `cospif(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46cospiff>`__

       1 ULP
     - `cospi(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45cospid>`__

       2 ULP
   * - :math:`\sin(\pi x)`
     - `sinpif(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46sinpiff>`__

       1 ULP
     - `sinpi(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45sinpid>`__

       2 ULP
   * - :math:`\sin(\pi x), \cos(\pi x)`
     - `sincospif(x, sptr, cptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv49sincospiffPfPf>`__

       1 ULP
     - `sincospi(x, sptr, cptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48sincospidPdPd>`__

       2 ULP
   * - :math:`\Phi(x)`
     - `normcdff(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48normcdfff>`__

       5 ULP
     - `normcdf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47normcdfd>`__

       5 ULP
   * - :math:`\Phi^{-1}(x)`
     - `normcdfinvf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv411normcdfinvff>`__

       5 ULP
     - `normcdfinv(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv410normcdfinvd>`__

       8 ULP
   * - :math:`\mathrm{erfc}^{-1}(x)`
     - `erfcinvf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48erfcinvff>`__

       4 ULP
     - `erfcinv(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47erfcinvd>`__

       6 ULP
   * - :math:`e^{x^2}\mathrm{erfc}(x)`
     - `erfcxf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46erfcxff>`__

       4 ULP
     - `erfcx(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45erfcxd>`__

       4 ULP
   * - :math:`\mathrm{erf}^{-1}(x)`
     - `erfinvf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47erfinvff>`__

       2 ULP
     - `erfinv(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46erfinvd>`__

       5 ULP
   * - :math:`I_0(x)`
     - `cyl_bessel_i0f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv414cyl_bessel_i0ff>`__ ，仅设备端

       6 ULP
     - `cyl_bessel_i0(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv413cyl_bessel_i0d>`__ ，仅设备端

       6 ULP
   * - :math:`I_1(x)`
     - `cyl_bessel_i1f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv414cyl_bessel_i1ff>`__ ，仅设备端

       6 ULP
     - `cyl_bessel_i1(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv413cyl_bessel_i1d>`__ ，仅设备端

       6 ULP
   * - :math:`J_0(x)`
     - `j0f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43j0ff>`__

       ▪ 当 :math:`|x| < 8` 时为 9 ULP

       ▪ 否则最大绝对误差 :math:`= 2.2 \cdot 10^{-6}`
     - `j0(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42j0d>`__

       ▪ 当 :math:`|x| < 8` 时为 7 ULP

       ▪ 否则最大绝对误差 :math:`= 5 \cdot 10^{-12}`
   * - :math:`J_1(x)`
     - `j1f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43j1ff>`__

       ▪ 当 :math:`|x| < 8` 时为 9 ULP

       ▪ 否则最大绝对误差 :math:`= 2.2 \cdot 10^{-6}`
     - `j1(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42j1d>`__

       ▪ 当 :math:`|x| < 8` 时为 7 ULP

       ▪ 否则最大绝对误差 :math:`= 5 \cdot 10^{-12}`
   * - :math:`J_n(x)`
     - `jnf(n, x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43jnfif>`__

       当 :math:`n = 128` 时，最大绝对误差 :math:`= 2.2 \cdot 10^{-6}`
     - `jn(n, x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42jnid>`__

       当 :math:`n = 128` 时，最大绝对误差 :math:`= 5 \cdot 10^{-12}`
   * - :math:`Y_0(x)`
     - `y0f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43y0ff>`__

       ▪ 当 :math:`|x| < 8` 时为 9 ULP

       ▪ 否则最大绝对误差 :math:`= 2.2 \cdot 10^{-6}`
     - `y0(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42y0d>`__

       ▪ 当 :math:`|x| < 8` 时为 7 ULP

       ▪ 否则最大绝对误差 :math:`= 5 \cdot 10^{-12}`
   * - :math:`Y_1(x)`
     - `y1f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43y1ff>`__

       ▪ 当 :math:`|x| < 8` 时为 9 ULP

       ▪ 否则最大绝对误差 :math:`= 2.2 \cdot 10^{-6}`
     - `y1(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42y1d>`__

       ▪ 当 :math:`|x| < 8` 时为 7 ULP

       ▪ 否则最大绝对误差 :math:`= 5 \cdot 10^{-12}`
   * - :math:`Y_n(x)`
     - `ynf(n, x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43ynfif>`__

       ▪ 当 :math:`|x| < n` 时为 :math:`\lceil 2 + 2.5n \rceil`

       ▪ 否则最大绝对误差 :math:`= 2.2 \cdot 10^{-6}`
     - `yn(n, x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42ynid>`__

       当 :math:`|x| > 1.5n` 时，最大绝对误差 :math:`= 5 \cdot 10^{-12}`

针对 ``__half`` 、 ``__nv_bfloat16`` 和 ``__float128/_Float128`` 的非标准 CUDA 数学函数仅在设备代码中可用。

.. list-table:: 非标准 CUDA 数学函数 ``__nv_bfloat16`` 、 ``__half`` 、 ``__float128/_Float128`` 的映射与精度（最大 ULP）
   :header-rows: 1
   :widths: 25 25 25 25

   * - 含义
     - ``__nv_bfloat16``
     - ``__half``
     - ``__float128/_Float128``
   * - :math:`\dfrac{1}{x}`
     - `hrcp(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hrcpK13__nv_bfloat16>`__

       0 ULP
     - `hrcp(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hrcpK6__half>`__

       0 ULP
     - N/A
   * - :math:`10^x`
     - `hexp10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hexp10K13__nv_bfloat16>`__

       0 ULP
     - `hexp10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hexp10K6__half>`__

       0 ULP
     - `__nv_fp128_exp10(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_exp10g>`__

       1 ULP
   * - :math:`\dfrac{1}{\sqrt{x}}`
     - `hrsqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hrsqrtK13__nv_bfloat16>`__

       0 ULP
     - `hrsqrt(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hrsqrtK6__half>`__

       0 ULP
     - N/A
   * - :math:`\tanh(x)` （近似）
     - `htanh_approx(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv412htanh_approxK13__nv_bfloat16>`__

       1 ULP
     - `htanh_approx(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv412htanh_approxK6__half>`__

       1 ULP
     - N/A

.. _intrinsic-functions:

5.5.9. 内建函数
---------------

内建数学函数是其对应的 `CUDA C 标准库数学函数 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html>`__ 的更快但精度较低的版本。

- 它们具有相同的名称，前缀为 ``__`` ，例如 ``__sinf(x)`` 。
- 它们仅在设备代码中可用。
- 它们更快，因为映射到的原生指令更少。
- 标志 ``--use_fast_math`` 会自动将相应的 `CUDA Math API 函数 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html>`__ 转换为内建函数。受影响函数的完整列表参见 :ref:`--use_fast_math 的影响 <use-fast-math-effect>` 一节。

.. _basic-intrinsic-functions:

5.5.9.1. 基本内建函数
^^^^^^^^^^^^^^^^^^^^^

一部分数学内建函数允许指定舍入模式：

- 以 ``_rn`` 为后缀的函数使用*向最近偶数舍入*（round to nearest even）模式运算。
- 以 ``_rz`` 为后缀的函数使用*向零舍入*（round towards zero）模式运算。
- 以 ``_ru`` 为后缀的函数使用*向上舍入*（向正无穷方向，round up）模式运算。
- 以 ``_rd`` 为后缀的函数使用*向下舍入*（向负无穷方向，round down）模式运算。

``__fadd_[rn,rz,ru,rd]()`` 、 ``__dadd_[rn,rz,ru,rd]()`` 、 ``__fmul_[rn,rz,ru,rd]()`` 和 ``__dmul_[rn,rz,ru,rd]()`` 函数映射到编译器绝不会合并为 ``FFMA`` 或 ``DFMA`` 指令的加法和乘法运算。相比之下，由 ``*`` 和 ``+`` 运算符生成的加法和乘法通常会被合并为 ``FFMA`` 或 ``DFMA`` 。

下表列出了单精度和双精度浮点内建函数。它们的最大 ULP 误差均为 0，并且符合 IEEE 标准。

.. list-table:: 单精度和双精度浮点内建函数
   :header-rows: 1
   :widths: 20 40 40

   * - 含义
     - ``float``
     - ``double``
   * - :math:`x + y`
     - `__fadd_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fadd_rnff>`__
     - `__dadd_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dadd_rndd>`__
   * - :math:`x - y`
     - `__fsub_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fsub_rnff>`__
     - `__dsub_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dsub_rndd>`__
   * - :math:`x \cdot y`
     - `__fmul_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmul_rnff>`__
     - `__dmul_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dmul_rndd>`__
   * - :math:`x \cdot y + z`
     - `__fmaf_[rn,rz,ru,rd](x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff>`__
     - `__fma_[rn,rz,ru,rd](x, y, z) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv48__fma_rnddd>`__
   * - :math:`\dfrac{x}{y}`
     - `__fdiv_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fdiv_rnff>`__
     - `__ddiv_[rn,rz,ru,rd](x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__ddiv_rndd>`__
   * - :math:`\dfrac{1}{x}`
     - `__frcp_[rn,rz,ru,rd](x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__frcp_rnf>`__
     - `__drcp_[rn,rz,ru,rd](x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__drcp_rnd>`__
   * - :math:`\sqrt{x}`
     - `__fsqrt_[rn,rz,ru,rd](x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fsqrt_rnf>`__
     - `__dsqrt_[rn,rz,ru,rd](x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv410__dsqrt_rnd>`__

.. _single-precision-only-intrinsic-functions:

5.5.9.2. 仅单精度内建函数
^^^^^^^^^^^^^^^^^^^^^^^^^

下表列出了单精度浮点内建函数及其最大 ULP 误差。

- 最大 ULP 误差表述为：函数返回值与按照*向偶数舍入*（round-to-nearest ties-to-even）舍入模式获得的相应精度的正确舍入结果之间，以 ULP 计的差异的最大观测绝对值。

- 误差界限来自广泛但并非详尽的测试。因此，它们无法得到保证。

.. list-table:: 仅单精度浮点内建函数的映射与精度（最大 ULP）
   :header-rows: 1
   :widths: 25 25 50

   * - 函数
     - 含义
     - 最大 ULP 误差
   * - `__fdividef(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff>`__
     - :math:`\dfrac{x}{y}`
     - 当 :math:`|y| \in [2^{-126}, 2^{126}]` 时为 :math:`2`
   * - `__frsqrt_rn(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv411__frsqrt_rnf>`__
     - :math:`\dfrac{1}{\sqrt{x}}`
     - 0 ULP
   * - `__expf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff>`__
     - :math:`e^x`
     - :math:`2 + \lfloor |1.173 \cdot x| \rfloor`
   * - `__exp10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff>`__
     - :math:`10^x`
     - :math:`2 + \lfloor |2.97 \cdot x| \rfloor`
   * - `__powf(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff>`__
     - :math:`x^y`
     - 由 ``exp2f(y * __log2f(x))`` 导出
   * - `__logf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff>`__
     - :math:`\ln(x)`
     - ▪ 当 :math:`x \in [0.5, 2]` 时绝对误差为 :math:`2^{-21.41}`

       ▪ 否则为 3 ULP
   * - `__log2f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff>`__
     - :math:`\log_2(x)`
     - ▪ 当 :math:`x \in [0.5, 2]` 时绝对误差为 :math:`2^{-22}`

       ▪ 否则为 2 ULP
   * - `__log10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff>`__
     - :math:`\log_{10}(x)`
     - ▪ 当 :math:`x \in [0.5, 2]` 时绝对误差为 :math:`2^{-24}`

       ▪ 否则为 3 ULP
   * - `__sinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff>`__
     - :math:`\sin(x)`
     - ▪ 当 :math:`x \in [-\pi, \pi]` 时绝对误差为 :math:`2^{-21.41}`

       ▪ 否则更大
   * - `__cosf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff>`__
     - :math:`\cos(x)`
     - ▪ 当 :math:`x \in [-\pi, \pi]` 时绝对误差为 :math:`2^{-21.41}`

       ▪ 否则更大
   * - `__sincosf(x, sptr, cptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf>`__
     - :math:`\sin(x), \cos(x)`
     - 逐分量与 ``__sinf(x)`` 和 ``__cosf(x)`` 相同
   * - `__tanf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff>`__
     - :math:`\tan(x)`
     - 由 ``__sinf(x) * (1 / __cosf(x))`` 导出
   * - `__tanhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff>`__
     - :math:`\tanh(x)`
     - ▪ 最大相对误差： :math:`2^{-11}`

       ▪ 即使在 ``-ftz=true`` 编译标志下，次正规结果也不会被刷新为零。

.. _use-fast-math-effect:

5.5.9.3. ``--use_fast_math`` 的影响
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nvcc`` 编译标志 ``--use_fast_math`` 会将在设备代码中调用的一部分 `CUDA Math API 函数 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html>`__ 转换为其对应的内建函数。请注意， :ref:`CUDA C++ 标准库函数 <cuda-c-mathematical-standard-library-functions>` 也会受到此标志的影响。有关使用内建函数代替 CUDA Math API 函数的影响的更多细节，参见 :ref:`内建函数 <intrinsic-functions>` 一节。

   更稳健的方法是，仅在性能收益足以证明其合理、且精度降低和特殊情形处理不同等变化后的属性可以接受的地方，选择性地用内建版本替换数学函数调用。

.. list-table:: 受 ``--use_fast_math`` 直接影响的函数
   :header-rows: 1
   :widths: 50 50

   * - 设备函数
     - 内建函数
   * - `x/y, fdividef(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48fdividefff>`__
     - `__fdividef(x, y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff>`__
   * - `sinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44sinff>`__
     - `__sinf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff>`__
   * - `cosf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44cosff>`__
     - `__cosf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff>`__
   * - `tanf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44tanff>`__
     - `__tanf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff>`__
   * - `sincosf(x, sptr, cptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47sincosffPfPf>`__
     - `__sincosf(x, sptr, cptr) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf>`__
   * - `logf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff>`__
     - `__logf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff>`__
   * - `log2f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45log2ff>`__
     - `__log2f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff>`__
   * - `log10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log10ff>`__
     - `__log10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff>`__
   * - `expf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff>`__
     - `__expf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff>`__
   * - `exp10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46exp10ff>`__
     - `__exp10f(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff>`__
   * - `powf(x,y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff>`__
     - `__powf(x,y) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff>`__
   * - `tanhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45tanhff>`__
     - `__tanhf(x) <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff>`__

.. _references:

5.5.10. 参考文献
----------------

#. `IEEE 754-2019 Standard for Floating-Point Arithmetic <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__.

#. Jean-Michel Muller. `On the definition of ulp(x) <https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf>`__. INRIA/LIP research report, 2005.

#. Nathan Whitehead, Alex Fit-Florea. `Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs <https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus>`__. Nvidia Report, 2011.

#. David Goldberg. `What every computer scientist should know about floating-point arithmetic <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>`__. ACM Computing Surveys, March 1991.

#. David Monniaux. `The pitfalls of verifying floating-point computations <https://dl.acm.org/doi/pdf/10.1145/1353445.1353446>`__. ACM Transactions on Programming Languages and Systems, May 2008.

#. Peter Dinda, Conor Hetland. `Do Developers Understand IEEE Floating Point? <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8425212>`__. IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2018.