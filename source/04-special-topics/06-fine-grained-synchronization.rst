.. _fine-grained-synchronization:

4.6. 细粒度同步
================

细粒度同步让消费者通过检查数据本身来判断其输入的每个 tile 何时就绪，而不是等待某个标志或等待生产者 grid 完成。
在本章中，tile 指消费者线程块作为一个整体来加载和使用的输入数据子集。
因此，每个消费者线程块一旦其 tile 就绪就可以使用它，而生产者可以继续处理其他 tile。
这可以降低短的、相互依赖的核函数链中的交接延迟。

该协议用一个保留的位模式（称为哨兵值（sentinel））来标记尚未写入的数据。缓冲区首先用哨兵值初始化。
生产者将其替换为位模式永远不与哨兵值匹配的最终输出值，而消费者在加载其输入 tile 时检查哨兵值。
生产者还必须遵循 :ref:`第 4.6.4.3 节 <fgs-producer-rules>` 的存储规则，使得检查未发现哨兵值的 tile 可以被视为完整。

本章介绍通过 TMA 数据有效性报告（TMA data validity reporting）实现的该协议，它在计算能力 10.7 的设备上可用。
批量异步拷贝可以在将数据从全局内存传输到共享内存的同时检查哨兵值。
它通过一个以 ``layout::v1`` 格式初始化的共享内存屏障来报告结果，该格式区分拷贝完成与数据就绪。
如果拷贝报告了哨兵值，消费者必须再次发出该拷贝；硬件不会自动重试。

该协议可以协调独立的核函数，也可以协调融合或持久化核函数内部的线程块，前提是生产者能够在消费者等待时继续推进。
当生产者使用作用域覆盖两个 GPU 的内存操作向消费者 GPU 内存中的缓冲区写入时，它也适用于 GPU 之间（参见 :ref:`第 3.2.3 节 <thread-scopes>`）。
本章讨论同一 GPU 上的一个生产者和一个消费者。
本章的示例在一个 GPU 上运行一个初始化核函数、一个生产者核函数和一个消费者核函数，并使用程序化依赖启动让后两者重叠。
性能收益取决于这种重叠，但正确性并非如此（参见 :ref:`第 4.6.5 节 <fgs-producer-and-consumer-concurrency>`）。

.. _fgs-definitions:

4.6.1. 定义
------------

以下术语在全章中使用。

**Tile**：tile 是输入数据的子集（例如矩阵的一个块），由消费者线程块作为一个整体来加载和使用。
这一一般含义与 CUDA Tile 编程模型中 tile 的含义不同（参见 :ref:`第 1.2.2.3.1 节 <tile-arrays-and-tiles>`）。

**哨兵值（Sentinel）**：哨兵值是一个保留的位模式，用于标记生产者尚未写入的数据。
缓冲区用哨兵值初始化，且生产者永远不会把哨兵值作为结果存储。
报告机制决定哨兵值（参见 :ref:`第 4.6.4.1 节 <fgs-sentinel-value>`）。

**报告机制（Report Mechanism）**：报告机制决定从全局内存到共享内存的批量异步拷贝是否检查以及检查所拷贝的数据的方式。
拷贝通过 PTX 中的限定符或 ``cuda::ptx`` 中的参数来指定它。
有效性报告机制是除 ``.mbarrier::report::disabled`` 之外的任何报告机制。
它指定哨兵值以及拷贝检查源数据的粒度（参见表 15）。

**Layout v1**：Layout v1 是 PTX ``mbarrier`` 对象的 ``layout::v1`` 格式。
处于该格式的屏障存储一个有效载荷报告（payload report），并允许其主阶段和条件阶段分别推进。
带有效性报告机制的拷贝必须由 layout v1 屏障跟踪。

**report-on 操作（Report-On Operation）**：report-on 操作在共享内存屏障的有效载荷报告中记录信息。
当拷贝检查的某个元素与哨兵值匹配时，带有效性报告机制的拷贝会在跟踪它的屏障上执行一次 report-on 操作。

**有效载荷报告（Payload Report）**：有效载荷报告是 layout v1 屏障为每个主阶段存储的信息，由一个报告谓词和一个报告值组成。
阶段开始时它为零，当该阶段跟踪的某个操作执行了 report-on 操作时它变为非零。

**主阶段与条件阶段（Primary Phase and Conditional Phase）**：每个 PTX ``mbarrier`` 对象，无论其布局如何，
都有一个主阶段和一个条件阶段。当待到达计数和事务计数都归零时，主阶段推进。
在 layout v1 屏障中，只有当有效载荷报告为零时，条件阶段才随之推进（参见 :ref:`第 4.6.3.1 节 <fgs-barrier-layout-and-phases>`）。

**报告谓词（Report Predicate）**：报告谓词是有效载荷报告的一部分，由对主阶段的等待返回。
只有当等待返回 true 时它才有意义。
此时，若已完成阶段所跟踪的某个操作执行了 report-on 操作（例如发现了哨兵值的拷贝），它为 true。

.. _fgs-background:

4.6.2. 背景
------------

生产者核函数将数据（例如张量）写入全局内存，消费者核函数读取它。表 14 总结了为这两个核函数排序的方式。

使用普通的 :ref:`流顺序（stream ordering） <cuda-streams>` 时，消费者要等到生产者完成并且其所有写入都变得可见之后才会启动。
这是最简单的机制，也是最粗粒度的机制：消费者要等待最后一个生产者线程块完成，而且消费者的启动延迟和准备工作完全暴露出来。

:ref:`程序化依赖启动 <programmatic-dependent-launch-details>` 让消费者提前启动，并在生产者仍在执行时运行其准备工作。
然而依赖仍然以整个 grid 的粒度来解决：消费者中的 ``cudaGridDependencySynchronize()`` 只有在生产者 grid 完成
且其所有内存操作都变得可见之后才返回。

标志协议在两个核函数同时运行的期间解决各个线程块之间的依赖。生产者写入一个 tile，执行设备作用域的 release 栅栏，
使该 tile 在标志之前变得可见（参见 :ref:`第 3.2.3 节 <thread-scopes>`），然后写入标志。
消费者以 acquire 语义轮询标志，之后才加载该 tile。因此关键路径包含栅栏加上两次有依赖的内存往返。

细粒度同步保留了标志协议的每 tile 粒度，但去掉了栅栏和标志两者。消费者未发现哨兵值的那次加载也正是交付数据的加载，
因此交接只需一次内存往返而不是两次。作为交换，缓冲区必须在每次使用前填入哨兵值，且生产者必须遵循一小套规则
（参见 :ref:`第 4.6.4.3 节 <fgs-producer-rules>`）。

.. list-table:: 表 14 为生产者核函数与消费者核函数排序的方式
   :header-rows: 1
   :widths: 22 33 10 35

   * - 机制
     - 消费者等待什么
     - 粒度
     - 要求
   * - 流顺序
     - 生产者 grid 的完成与内存刷新，随后是消费者 grid 的启动
     - Grid
     - 无
   * - 程序化依赖启动
     - 整个生产者 grid 的完成及其写入的可见性，由消费者中的 ``cudaGridDependencySynchronize()`` 观察
     - Grid
     - 消费者以 programmatic stream serialization 启动
   * - 标志协议
     - 生产者在 release 栅栏之后写入的标志
     - Tile
     - 生产者能在消费者等待时继续推进；每个 tile 一个标志
   * - 细粒度同步
     - 数据本身不再保持哨兵值
     - Tile
     - 生产者能在消费者等待时继续推进；缓冲区用哨兵值初始化；生产者遵循 :ref:`第 4.6.4.3 节 <fgs-producer-rules>` 的规则

.. _fgs-tma-data-validity-reporting:

4.6.3. TMA 数据有效性报告
--------------------------

要使用细粒度同步，消费者用指定了有效性报告机制的从全局内存到共享内存的 :ref:`批量异步拷贝 <async-copies-tma>`
来加载每个 tile。该拷贝或者是一维的 `cp.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`_，
或者是使用张量映射的 `cp.async.bulk.tensor <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_
（参见 :ref:`第 4.13.2.2 节 <async-copies-tma-multi-dim>`）。
如果拷贝检查的某个元素与哨兵位模式匹配，就在跟踪该拷贝的共享内存屏障上执行一次 report-on 操作。
屏障把这一点记录在有效载荷报告中，等待的线程可以与拷贝的完成一起观察到它。
数据仍会被写入共享内存；报告只是告诉核函数这些数据还不可使用。

``.report_mechanism`` 限定符仅支持从全局内存到共享内存的拷贝。它需要 PTX ISA 9.4 或更高版本，
以及计算能力 10.7 的系列特定编译目标 ``sm_107f`` 或架构特定编译目标 ``sm_107a``，或同系列的更新目标
（参见 :ref:`第 5.1.2.2 节 <compute-capabilities-family-specific-features>`）。
libcu++ 的 ``cuda::ptx`` 命名空间通过 `cuda::ptx::cp_async_bulk <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html>`_
和 `cuda::ptx::cp_async_bulk_tensor <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html>`_
的额外重载（接受一个报告机制参数）将其暴露出来。表 15 列出了可用的机制。

.. list-table:: 表 15 批量异步拷贝的报告机制
   :header-rows: 1
   :widths: 36 30 18 16

   * - PTX 限定符
     - ``cuda::ptx`` 常量
     - 哨兵值
     - 检查方式
   * - ``.mbarrier::report::disabled`` （默认）
     - ``mbarrier_report_disabled`` （仅 ``cp_async_bulk_tensor`` ）
     - 无
     - 无
   * - ``.mbarrier::report::validity::per_16bytes::80000000``
     - ``mbarrier_report_valid_per_16bytes_80000000``
     - ``0x80000000`` （32 位 -0.0）
     - 每个对齐 16 字节中抽样的一个元素
   * - ``.mbarrier::report::validity::per_16bytes::8000``
     - ``mbarrier_report_valid_per_16bytes_8000``
     - ``0x8000`` （16 位 -0.0）
     - 每个对齐 16 字节中抽样的一个元素
   * - ``.mbarrier::report::validity::per_16bytes::80``
     - ``mbarrier_report_valid_per_16bytes_80``
     - ``0x80`` （8 位 -0.0）
     - 每个对齐 16 字节中抽样的一个元素
   * - ``.mbarrier::report::validity::per_16bytes::8``
     - ``mbarrier_report_valid_per_16bytes_8``
     - ``0x8`` （4 位 -0.0）
     - 每个对齐 16 字节中抽样的一个元素
   * - ``.mbarrier::report::validity::per_element::ff``
     - ``mbarrier_report_valid_per_element_ff``
     - ``0xFF`` （8 位 NaN）或 ``0xF`` （4 位全 1）
     - 每个元素

使用按 16 字节（per-16-byte）的机制时，拷贝将源数据每个对齐 16 字节块中抽样的一个元素与该模式进行比较。
因此生产者必须把每个块作为整体写入（参见 :ref:`第 4.6.4.3 节 <fgs-producer-rules>`）。
``per_element::ff`` 机制比较每个元素，因此不依赖生产者写入完整的 16 字节块。

.. _fgs-barrier-layout-and-phases:

4.6.3.1. 屏障布局与阶段
^^^^^^^^^^^^^^^^^^^^^^^^

使用有效性报告机制的拷贝必须由一个以 PTX ``mbarrier`` 对象的 ``layout::v1`` 格式初始化的共享内存屏障来跟踪。
每个屏障都有一个主阶段和一个条件阶段。默认格式 ``layout::v0`` 跟踪 :ref:`第 4.11.6 节 <async-barriers-tracking-async-mem-ops>`
所描述的待到达计数和事务计数，其两个阶段总是一起推进。
Layout v1 额外为每个主阶段存储一个有效载荷报告，并允许两个阶段分别推进：

- 当待到达计数和事务计数都归零时，主阶段推进。对于由该屏障跟踪的拷贝，主阶段完成意味着所有字节都已写入共享内存。
- 只有当正在完成的阶段的有效载荷报告为零时——即该阶段跟踪的没有任何操作执行过 report-on 操作时——
  条件阶段才与主阶段一起推进。如果某次拷贝报告了哨兵值，主阶段会推进而条件阶段不会。
  每个主阶段都以零有效载荷报告开始。

`mbarrier.test_wait 和 mbarrier.try_wait <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>`_
指令通过 ``.phase_type::primary`` 或 ``.phase_type::conditional`` 限定符选择要观察的阶段。
等待主阶段还会返回一个报告谓词，当有效载荷报告非零时它为 true。
该谓词只表明该阶段跟踪的至少一次拷贝匹配了其模式；它不能指明是哪一次拷贝。
因此，需要区分结果的各次拷贝应当由不同的屏障跟踪。

这两个阶段直接对应消费者中的两种角色：

- 负责重新发出拷贝的线程等待主阶段。如果报告谓词被置位，数据尚未就绪。
  随后该线程用期望的字节数重新武装屏障，重新发出拷贝，并以翻转后的 parity 等待下一个主阶段。
- 对数据进行计算的线程等待条件阶段。当某次拷贝报告哨兵值时，条件阶段的 parity 不会改变，
  因此无论拷贝被重新发出多少次，这些线程等待的都是同一个 parity。它们只被释放一次，即当某次拷贝携带有效数据完成时。

在 libcu++ 中，屏障通过 ``cuda::ptx::mbarrier_init(cuda::ptx::layout_v1, ...)`` 创建
（参见 `libcu++ 关于 mbarrier_init 的文档 <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/mbarrier_init.html>`_）。
包装函数 `cuda::ptx::mbarrier_try_wait_parity 和 cuda::ptx::mbarrier_test_wait_parity <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/mbarrier_wait.html>`_
接受 ``cuda::ptx::mbarrier_phase_primary`` 并附带一个用于接收报告谓词的 ``bool&``，
或者接受 ``cuda::ptx::mbarrier_phase_conditional`` 而不附带它。``cuda::barrier`` 对象使用 layout v0，
不能与有效性报告机制一起使用。

.. note::

   Layout v1 屏障每个阶段最多接受 511（``2^9 - 1`` ）个期望到达，而 layout v0 为 ``2^20 - 1``。
   计算传输（compute fabric transport）操作正是使用同样的布局和同样的有效载荷报告来报告错误。
   如果程序将 fabric 操作与使用有效性报告机制的拷贝组合在一起，程序的行为是未定义的。

.. _fgs-protocol:

4.6.4. 协议
------------

本节描述该协议在计算能力 10.7 设备上以 TMA 数据有效性报告方式实现的样子。协议由五个步骤组成，
由管理缓冲区的代码、生产者和消费者分别执行：

- 选择一个有效数据永远不会包含的哨兵值。
- 在消费者开始加载缓冲区之前，用哨兵值初始化缓冲区。
- 产生数据时绝不写入哨兵值，并且每个位置只写一次。使用按 16 字节的报告机制时，每个对齐 16 字节块作为整体写入。
- 在消费者中，用带有效性报告的批量异步拷贝加载每个 tile，并且只要哨兵值被报告就重新发出该拷贝。
- 在缓冲区承载下一轮数据之前重新初始化缓冲区。

.. _fgs-sentinel-value:

4.6.4.1. 哨兵值
^^^^^^^^^^^^^^^^

报告机制决定了哨兵值（参见表 15）。对于该表中带符号位的浮点格式，按 16 字节的机制使用负零。
负零与正零比较相等，因此生产者可以用正零替换它，只有检查零符号的操作才能观察到这种替换。
``per_element::ff`` 机制使用全 1 模式，即 8 位比例因子格式 ue8m0 和 ue5m3 的 NaN 编码。
这些格式没有符号位，因此也没有负零。

检查比较的是原始位模式，与元素类型无关，因此 ``0x80000000`` 模式适用于任何 32 位类型，``0xFF`` 模式适用于任何 8 位类型。
对于整数数据，哨兵值必须是能够证明生产者永远不会发出的位模式。
使用 ``per_element::ff`` 机制时，4 位元素的模式为 ``0xF``，并且每个元素都与之比较。
因此 4 位元数据缓冲区要用完整的 ``0xFF`` 字节初始化，使每个字节的两个 4 位元素都持有该模式。表 16 列出了每种元素类型的哨兵值。

.. list-table:: 表 16 按元素类型划分的哨兵值
   :header-rows: 1
   :widths: 40 18 24 14

   * - 元素类型
     - 哨兵值
     - 位模式
     - 元素宽度
   * - 32 位浮点（``float`` ）
     - -0.0
     - ``0x80000000``
     - 32 位
   * - 16 位浮点（``half``、``bfloat16`` ）
     - -0.0
     - ``0x8000``
     - 16 位
   * - 8 位浮点（``e4m3``、``e5m2`` ）
     - -0.0
     - ``0x80``
     - 8 位
   * - 4 位浮点（``e2m1`` ）
     - -0.0
     - ``0x8``
     - 4 位
   * - 无符号位的 8 位比例因子（``ue8m0``、``ue5m3`` ）以及其他从不持有 ``0xFF`` 的 8 位数据
     - NaN / 全 1
     - ``0xFF``
     - 8 位
   * - 从不持有 ``0xF`` 的 4 位元数据（例如稀疏性元数据）
     - 全 1
     - ``0xF`` （缓冲区用 ``0xFF`` 字节初始化）
     - 4 位
   * - 整数数据
     - 生产者从不发出的值
     - ``0x80000000``、``0x8000``、``0x80`` 或 ``0xFF``
     - 32、16 或 8 位

.. _fgs-buffer-initialization:

4.6.4.2. 缓冲区初始化
^^^^^^^^^^^^^^^^^^^^^^

消费者所等待数据的每个位置，都必须在生产者写入它之前用哨兵值初始化。
初始化还必须在消费者加载该位置之前对消费者可见。初始化就是一组普通的存储，
后跟初始化代码与消费者之间的排序。第一次初始化通常放在单独的核函数中或放在生产者中。

**在单独的核函数中**。同一流中较早的核函数，或图中较早的节点写入哨兵值，由流顺序或图依赖保证可见性。
这将初始化代码从生产者和消费者中分离出来，并允许一个核函数清零多个缓冲区，
但它增加了一次启动，并使框架负责调度清零工作。

**在生产者中**。生产者在其触发依赖启动之前，把哨兵值写入自己的输出。
生产者线程块的各个线程将哨兵值写入该块的输出 tile 并同步。然后一个线程执行 release 栅栏，
随后执行 ``cudaTriggerProgrammaticLaunchCompletion()``：

.. code-block:: cpp

   // Every thread of the block writes the sentinel to the block's output tile.
   write_sentinel_to_output_tile();
   __syncthreads();   // all sentinel stores of the block precede the fence and the trigger
   if (threadIdx.x == 0) {
       cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
       cudaTriggerProgrammaticLaunchCompletion();
   }
   // Compute and store the results; they replace the sentinel.

在程序顺序上先于 `griddepcontrol.launch_dependents <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol>`_
的 release 栅栏，与依赖 grid 的启动同步。这一保证要求两个核函数运行在同一个 :ref:`内存同步域 <memory-sync-domains>` 中
且栅栏具有设备作用域，或者栅栏具有系统作用域。由于触发在线程块中首次执行时即对整个线程块生效，
写入哨兵值的线程会在触发之前同步，从而每个哨兵存储都被排在栅栏之前。

哨兵存储、栅栏和触发会延迟消费者的启动。它们也会延迟生产者自身的工作，除非由一个本来空闲的 warp 来执行这些操作。
当一个 warp 写入哨兵值而其他 warp 计算替换它的输出值时，各 warp 必须同步，使每个输出存储排在同一位置的哨兵存储之后。
使用单独的初始化核函数则允许生产者在其开始时就触发，如示例所示。

如果改用批量异步存储写入哨兵值，发出这些存储的线程必须用 ``cp.async.bulk.commit_group`` 提交它们，
并在该线程块同步之前用 `cp.async.bulk.wait_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group>`_
等待它们的完成。该指令的 ``.read`` 变体只等待源数据被读取，这并不足够。

如果用普通存储写入哨兵值、用批量异步存储写入结果，那么两类存储使用不同的代理。
此时每个线程还要在其哨兵存储之后、在该线程块同步之前执行一次代理栅栏
（``cuda::ptx::fence_proxy_async``，参见 :ref:`异步代理 <async-thread-and-async-proxy>`）。

持久化生产者的每个线程块可以承担多个线程块索引的工作，例如在 grid-stride 循环中。
它首先将哨兵值写入所有这些索引的输出 tile。然后它同步、执行 release 栅栏并只触发一次，而不是每个索引触发一次。
如果线程块在触发之后又获得新的索引，例如通过 :ref:`Cluster Launch Control <cluster-launch-control>`，
那么这些索引的输出 tile 必须已经持有哨兵值，例如来自一个单独的初始化核函数。

.. _fgs-producer-rules:

4.6.4.3. 生产者规则
^^^^^^^^^^^^^^^^^^^^

生产者必须遵循三条规则：绝不写入哨兵值，每个位置只写一次，并且在使用按 16 字节的报告机制时，
每个对齐 16 字节块作为整体写入。第一条规则防止消费者无限等待；另外两条规则使检查未发现哨兵值的 tile 可以被视为完整。

**规范化**。把哨兵值作为合法值发出的生产者可能使消费者无限等待，因此结果在存储之前必须经过规范化。
以负零为哨兵值时，负零结果被替换为正零。使用 ``__fadd_rn()`` 加上正零即可完成替换，如示例所示，
因为在就近舍入模式下负零加正零等于正零。与 ``0.0f`` 的相等比较后跟一次 select 也会把负零替换为正零，
因为该比较对两个零都为真。对于 ``sm_107f`` 和 ``sm_107a`` 编译目标，PTX `cvt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt>`_
指令接受 ``.pzo`` （positive zero only，仅正零）限定符。该限定符在窄化浮点转换过程中执行同样的替换，
例如从 ``float`` 转换到 ``half`` 或转换到 8 位格式。

使用 NaN 模式作为哨兵值时，生产者同样绝不能存储 NaN。用 ``fmaxf`` 和 ``fminf`` 做钳位可以消除 NaN，
因为当一个操作数为 NaN 时这些函数返回另一个数值操作数。使用整数哨兵值时，生产者绝不能存储该值，
例如通过对结果做饱和钳制使其避开该值。

**只写一次**。每个位置必须只写一次，写入其最终值。
先写入部分结果、之后再更新的位置——例如带原子加的 split-K 归约——不适用于该协议。消费者可能接受部分值。

**整块写入**。生产者用单条 128 位存储写入每个对齐的 16 字节块，
例如示例中用一个小型辅助函数包装的 PTX `st.relaxed.gpu.global.b128 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st>`_ 指令。
带 ``.relaxed.gpu`` 限定符和 ``.b128`` 类型的批量异步存储同样以单条 128 位操作写入每个对齐的块。
在使用按 16 字节的报告机制时，把每个块作为整体写入是一项要求。
检查每块可能只查看一个元素，因此由若干较窄存储拼成的块可能在某些元素仍持有哨兵值时被报告为有效。

.. _fgs-consumer-check:

4.6.4.4. 消费者检查
^^^^^^^^^^^^^^^^^^^^

消费者使用带有效性报告机制的批量异步拷贝加载每个 tile，并在报告谓词被置位时重新发出该拷贝
（参见 :ref:`第 4.6.3.1 节 <fgs-barrier-layout-and-phases>`）。
如示例那样以 acquire 语义等待条件阶段的线程，一旦该阶段完成即可使用该 tile，无需任何额外的栅栏。
这只覆盖拷贝所加载的数据。如果消费者还依赖生产者写入的其他数据，那些数据仍需要常规同步，
例如 ``cudaGridDependencySynchronize()`` 或一对 release 与 acquire 操作。

.. _fgs-buffer-reuse:

4.6.4.5. 缓冲区复用
^^^^^^^^^^^^^^^^^^^^

在缓冲区能够承载下一轮数据之前必须重新初始化它，并且重新初始化不得与仍在读取上一轮数据的消费者发生竞争。
在两个或三个缓冲区之间轮换可以把清零移出关键路径，因为可以在一个缓冲区被写入和被读取的同时清零另一个缓冲区。

消费者也可以在消费完某个 tile 之后重新初始化该 tile，例如在持久化或迭代核函数中。
这只有在没有其他消费者线程块仍可能读取该 tile 时才是安全的，例如每个 tile 只被单个消费者线程块读取。
重新初始化还必须排在下一个生产者的写入之前变得有序，并在下一个消费者加载该 tile 之前变得可见，
这可能需要一个代理栅栏（参见 :ref:`异步代理 <async-thread-and-async-proxy>`）。

.. _fgs-partial-tiles-and-padding:

4.6.4.6. 部分 tile 与填充
^^^^^^^^^^^^^^^^^^^^^^^^^^

带有效性报告的拷贝所读取的每个元素，最终都必须持有非哨兵值的某个值，否则消费者可能无限等待。
生产者从不写入的元素——例如最后一个 tile 超出有效数据的部分，或为对齐插入的填充——可以用三种方式处理：

- 这些元素用非哨兵值的某个值初始化。
- 生产者用规范化值（例如正零）写入它们。
- 将拷贝限制在有效数据之内，例如通过张量映射的边界。

当有效数据在对齐的 16 字节块内部结束时——例如张量某一行不是 16 字节的整数倍长——该块的其余部分需要谨慎处理。
即使张量映射边界将其排除在拷贝之外，该块超出有效数据的字节最终也必须持有非哨兵值的某个值。
最简单的方式是让生产者把整个块作为整体写入，有效数据之外的部分写入规范化值。
使用按 16 字节的机制时，只把这些字节初始化为非哨兵值可能不够，因为块可能在其有效数据写入之前就被接受。

带 ``.ignore_oob`` 限定符的一维拷贝不能使用有效性报告机制
（参见 `cp.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`_）。

.. _fgs-producer-and-consumer-concurrency:

4.6.5. 生产者与消费者并发
--------------------------

等待某个 tile 的消费者线程块在生产者写入该 tile 之前无法推进。因此生产者必须能够在消费者线程块等待时继续推进，
而且只有当两个核函数重叠时性能收益才会出现。本章将细粒度同步与
:ref:`程序化依赖启动 <programmatic-dependent-launch-details>` 搭配使用，后者保证生产者的进度并允许两个核函数重叠：

- 生产者在其初始化策略允许的范围内尽早调用 ``cudaTriggerProgrammaticLaunchCompletion()``
  （参见 :ref:`第 4.6.4.2 节 <fgs-buffer-initialization>`），使消费者可以在生产者仍在计算时被调度。
- 消费者以 ``cudaLaunchAttributeProgrammaticStreamSerialization`` 属性启动，
  或者在 CUDA 图中使用从 ``cudaGraphKernelNodePortProgrammatic`` 端口引出的程序化边启动
  （参见 :ref:`第 4.5.3 节 <pdl-use-in-cuda-graphs>`）。
- 消费者在调用 ``cudaGridDependencySynchronize()`` 之前加载产生的数据。该调用等待整个生产者 grid，
  而这正是细粒度同步为已加载数据所替代的粗粒度依赖。消费者仍必须进行该调用，
  例如在其消费完已加载数据之后调用，那里该调用不在交接的关键路径上。

**进度保证**。在每个生产者线程块要么执行了触发要么退出之前，消费者 grid 不会被启动。
因此在第一个消费者线程块开始加载数据时，每个生产者线程块要么已经常驻，要么已经完成。
消费者线程块无法阻止任何生产者线程块被调度。

**单独的流**。当生产者与消费者位于不同的流中时，程序化事件提供同样的进度保证。
生产者以 ``cudaLaunchAttributeProgrammaticEvent`` 属性启动，记录一个用 ``cudaEventDisableTiming`` 创建的事件，
消费者的流等待该事件。当两个流被捕获进 CUDA 图时，这一依赖成为两个核函数节点之间的程序化边
（参见 :ref:`第 4.5.3 节 <pdl-use-in-cuda-graphs>`）。如果该属性的 ``triggerAtBlockStart`` 字段非零，
事件可以在每个生产者线程块一旦启动后立即触发。此时哨兵值必须在生产者启动之前写入，例如由一个单独的核函数。
消费者仍在消费完已加载数据之后调用 ``cudaGridDependencySynchronize()``。
如果没有这样的依赖，两次启动是无序的，先占用 GPU 的消费者可能使生产者饿死
（参见 :ref:`第 5.9.3.1 节 <cuda-cpp-execution-model-dependencies>`）。

**重叠**。两个核函数是否实际重叠仍是机会性的，就像程序化依赖启动通常那样。
即使没有重叠，协议也是正确的，因为消费者会不断重新发出其拷贝直到数据出现。
重叠可能发生在生产者最后一波留下的空闲 SM 上，也可能发生在对两个核函数的线程块都有空间的 SM 上。
示例采取了进一步的两个步骤使重叠更可能发生：

- 示例在序列启动之前用 ``cudaFuncGetAttributes()`` 加载全部三个核函数，
  如 :ref:`第 4.9.5 节 <lazy-loading-potential-hazards>` 所建议。对于默认启用的 :ref:`延迟加载 <lazy-loading-details>`，
  否则核函数会在其首次启动时才被加载，这可能使核函数执行串行化。
- 示例为生产者和消费者将 ``cudaFuncAttributePreferredSharedMemoryCarveout`` 设置为相同的值
  （参见 :ref:`第 3.2.6 节 <configuring-l1-shared-memory-balance>`）。SM 的 L1 缓存和共享内存共享一个统一数据缓存，
  其划分按核函数配置，共享内存配置不同的核函数可能在等待重新配置时串行化。驱动将 carveout 视为一个提示。

.. _fgs-example:

4.6.6. 示例
------------

以下示例针对 ``float`` 数据实现该协议，在同一流中使用三个核函数。
初始化核函数将中间缓冲区填满哨兵值，生产者核函数把输入数组的元素翻倍写入该缓冲区，
消费者核函数将结果中每个 1024 元素 tile 的元素反转。
每个生产者线程写入一个 16 字节块，每个消费者线程块用一个一维的、报告有效性的批量异步拷贝将一个 tile
加载到共享内存。

哨兵值、结果的规范化以及初始化核函数定义如下：

.. code-block:: cpp

   // The sentinel is negative zero. Its bit pattern is 0x80000000 for a 32-bit
   // float; valid producer output never contains this pattern (see canonicalize()).
   constexpr uint32_t SENTINEL_BITS = 0x80000000u;

   // The producer must never emit the sentinel. __fadd_rn() adds positive zero in
   // round-to-nearest mode, which turns a negative-zero result into positive zero.
   // No other value becomes the sentinel, and a NaN stays a NaN.
   __device__ __forceinline__ float canonicalize(float v)
   {
       return __fadd_rn(v, 0.0f);
   }

   // Fill a buffer with the sentinel, using 16-byte stores.
   __global__ void init_sentinel_kernel(float4 *buf, int num_chunks)
   {
       const float  s        = __uint_as_float(SENTINEL_BITS);
       const float4 sentinel = make_float4(s, s, s, s);
       for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_chunks; i += gridDim.x * blockDim.x)
       {
           buf[i] = sentinel;
       }
   }

由于由单独的核函数写入哨兵值（参见 :ref:`第 4.6.4.2 节 <fgs-buffer-initialization>`），
触发之前不需要有任何操作，生产者在开始时即触发依赖启动。
随后它对结果做规范化，并用一条位于设备作用域的 relaxed 128 位存储把每个块写一次：

.. code-block:: cpp

   // Relaxed, device-scope 128-bit store of one aligned 16-byte chunk.
   __device__ __forceinline__ void store_chunk_relaxed_gpu(float4 *p, const float4 &v)
   {
       __int128 bits;
       memcpy(&bits, &v, sizeof(bits));
       asm volatile("st.relaxed.gpu.global.b128 [%0], %1;" : : "l"(p), "q"(bits) : "memory");
   }

   __global__ void producer_kernel(const float4 *__restrict__ in, float4 *__restrict__ out)
   {
       // Allow the consumer to launch as early as possible. The sentinel was
       // written by a separate kernel, so nothing has to precede the trigger here.
       cudaTriggerProgrammaticLaunchCompletion();

       const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
       float4 v = in[chunk];
       v.x = canonicalize(2.0f * v.x);
       v.y = canonicalize(2.0f * v.y);
       v.z = canonicalize(2.0f * v.z);
       v.w = canonicalize(2.0f * v.w);

       // Each aligned 16-byte chunk is written once, with a single 128-bit store,
       // as the per-16-byte validity check requires.
       store_chunk_relaxed_gpu(&out[chunk], v);
   }

消费者的拷贝以内联 PTX 编写，使用 ``.relaxed.gpu`` 限定符和 ``.b128`` 类型，屏障操作使用
`libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html>`_ 的 ``cuda::ptx`` 包装函数。
因此与生产者的存储一样，该拷贝以设备作用域下强序、自然对齐的 128 位操作访问全局内存。
在两者都处于设备作用域时，通过通用代理的强序写入最终会被对同一位置、通过异步代理的强序访问观察到。
relaxed 操作是既无 release 也无 acquire 语义的强序操作。

在消费者中，线程 0 初始化一个 layout v1 屏障，使用 32 位负零报告机制发出拷贝，
并在报告谓词指示 tile 未就绪时重新发出拷贝。所有线程在读取 tile 之前等待 parity 为 0 的条件阶段，
并在写完它们的结果之后调用 ``cudaGridDependencySynchronize()``：

.. code-block:: cpp

   // Relaxed, device-scope bulk-asynchronous copy from global to shared memory
   // that compares a sampled element of each aligned 16-byte chunk with the
   // 32-bit negative-zero sentinel and reports whether any matched. The copy
   // accesses global memory with 128-bit operations and completes on the
   // layout v1 barrier `bar`.
   __device__ __forceinline__ void copy_tile_report_validity(void *smem_dst, const void *gmem_src, uint32_t bytes,
                                                             uint64_t *bar)
   {
       asm volatile("cp.async.bulk.relaxed.gpu.shared::cta.global.mbarrier::complete_tx::bytes"
                    ".mbarrier::report::validity::per_16bytes::80000000.b128 [%0], [%1], %2, [%3];"
                    :
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))), "l"(gmem_src), "r"(bytes),
                      "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar)))
                    : "memory");
   }

   __global__ void consumer_kernel(const float *produced, float *__restrict__ result)
   {
       // Source and destination of a one-dimensional bulk copy must be 16-byte
       // aligned and the size a multiple of 16 bytes.
       __shared__ alignas(128) float tile[TILE_ELEMS];
       // A copy with a validity report mechanism must be tracked by a layout v1
       // barrier, which stores the payload report.
       __shared__ alignas(8) uint64_t bar;

       if (threadIdx.x == 0)
       {
           ptx::mbarrier_init(ptx::layout_v1, &bar, 1);   // one arrival per phase: the issuing thread
       }
       __syncthreads();

       const float *src = produced + blockIdx.x * TILE_ELEMS;

       if (threadIdx.x == 0)
       {
           uint32_t primary_parity = 0;
           for (;;)
           {
               // Arm the current primary phase with the expected byte count, then
               // issue the copy with validity reporting enabled.
               ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(tile));
               copy_tile_report_validity(tile, src, sizeof(tile), &bar);

               // Wait for the *primary* phase, which completes once all bytes have
               // landed. The report predicate tells whether a sampled element still
               // held the sentinel.
               bool sentinel_seen = false;
               while (!ptx::mbarrier_try_wait_parity(ptx::mbarrier_phase_primary, ptx::sem_acquire, ptx::scope_cta,
                                                     sentinel_seen, &bar, primary_parity))
               {
               }
               if (!sentinel_seen)
               {
                   break;   // valid data: the conditional phase has advanced as well
               }
               // Not ready: some of the producer's stores to this tile were not yet
               // visible to the copy. The primary phase completed anyway, so the next
               // copy is tracked by the following primary phase.
               primary_parity ^= 1;
           }
       }

       // Every thread waits for the *conditional* phase. It advances only when a
       // copy completes with no sentinel reported, so the threads wait for parity 0
       // however many times the copy is re-issued.
       while (!ptx::mbarrier_try_wait_parity(ptx::mbarrier_phase_conditional, ptx::sem_acquire, ptx::scope_cta, &bar, 0))
       {
       }

       // The tile in shared memory is complete and valid. Consume it: each thread
       // writes four elements of the reversed tile.
       float *out = result + blockIdx.x * TILE_ELEMS;
       for (int i = 4 * threadIdx.x; i < 4 * threadIdx.x + 4; ++i)
       {
           out[i] = tile[TILE_ELEMS - 1 - i];
       }

       // A dependent grid must still call cudaGridDependencySynchronize(). Calling
       // it here, after the loaded data has been consumed, keeps the wait off the
       // critical path of the handoff.
       cudaGridDependencySynchronize();
   }

**到达计数**。屏障的到达计数为一：只有发出拷贝的线程通过
`mbarrier_arrive_expect_tx <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/mbarrier_arrive.html>`_
到达，并且它在每个主阶段到达一次，包括每次重新发出的拷贝。

在主机端，三个核函数被预加载，并且为生产者和消费者设置了相同的共享内存 carveout
（参见 :ref:`第 4.6.5 节 <fgs-producer-and-consumer-concurrency>`）。随后三个核函数被启动到同一个流中。
流顺序保证哨兵值在生产者开始之前已经可见，并且只有消费者以程序化流序列化启动，
从而它可以在生产者运行时就开始：

.. code-block:: cpp

       // Load all three kernels before the sequence is launched. With lazy module
       // loading (the default), a kernel is otherwise loaded at its first launch,
       // which can serialize kernel execution and keep the producer and the
       // consumer from overlapping.
       cudaFuncAttributes func_attr;
       CUDA_CHECK(cudaFuncGetAttributes(&func_attr, init_sentinel_kernel));
       CUDA_CHECK(cudaFuncGetAttributes(&func_attr, producer_kernel));
       CUDA_CHECK(cudaFuncGetAttributes(&func_attr, consumer_kernel));

       // The L1 cache and the shared memory of an SM share one unified data cache,
       // and the driver configures the split per kernel. Kernels with different
       // shared memory configurations can serialize behind the reconfiguration.
       // The producer uses no shared memory and the consumer does, so both request
       // the same carveout.
       CUDA_CHECK(cudaFuncSetAttribute(producer_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                       cudaSharedmemCarveoutMaxShared));
       CUDA_CHECK(cudaFuncSetAttribute(consumer_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                       cudaSharedmemCarveoutMaxShared));

       // 1. Initialize the intermediate buffer with the sentinel. Stream order
       //    guarantees this completes, and is visible, before the producer starts.
       init_sentinel_kernel<<<NUM_TILES, THREADS, 0, stream>>>(d_mid, NUM_ELEMS / 4);

       // 2. Launch the producer normally.
       producer_kernel<<<NUM_TILES, THREADS, 0, stream>>>(d_in, d_mid);

       // 3. Launch the consumer with programmatic dependent launch so that it can
       //    start while the producer is still running.
       cudaLaunchAttribute attribute[1];
       attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
       attribute[0].val.programmaticStreamSerializationAllowed = 1;

       cudaLaunchConfig_t config = {};
       config.gridDim  = NUM_TILES;
       config.blockDim = THREADS;
       config.stream   = stream;
       config.attrs    = attribute;
       config.numAttrs = 1;
       CUDA_CHECK(cudaLaunchKernelEx(&config, consumer_kernel, (const float *)d_mid, d_out));

.. _fgs-warp-specialized-kernels:

4.6.7. Warp 专用核函数
-----------------------

在示例中，一个线程发出拷贝、等待主阶段，并在哨兵值被报告时重新发出拷贝。
带有多个流水级的 warp 专用核函数（参见 :ref:`第 4.11.7 节 <async-barriers-producer-consumer>`）则可以改为增加一个重试 warp：

- 加载 warp 发出每个流水级的拷贝，就像没有细粒度同步时一样。
- 重试 warp 等待每个流水级的主阶段，为每个流水级单独跟踪 parity。当某个流水级的报告谓词被置位时，
  重试 warp 重新发出该流水级的所有拷贝。
- 计算 warp 只等待条件阶段。只有发出或重新发出拷贝的那个 warp 才在跟踪该拷贝的屏障上到达。
- 当某个流水级的报告谓词未被置位时，重试 warp 还在 signaling 该流水级已就绪可被填充的屏障上到达
  （``consumer_release`` 操作，参见 :ref:`第 4.12 节 <pipelines-details>`）。
  没有这次到达，加载 warp 可能在重试 warp 检查完之前的拷贝之前就重新填充该流水级，而重试 warp 可能漏掉一次报告。

使用 ``cp.async.bulk.tensor`` 的 ``.cta_group::2`` 限定符时，一个屏障可以跟踪向
:ref:`集群 <thread-block-clusters>` 中 CTA 对两个线程块的拷贝，但只有持有该屏障的线程块能等待它。
如果另一个线程块重新发出其拷贝或使用数据，则必须被告知拷贝何时完成以及是否报告了哨兵值，
例如通过 :ref:`分布式共享内存 <writing-cuda-kernels-distributed-shared-memory>`。

.. _fgs-guidelines:

4.6.8. 指南
------------

以下几点涉及该技术何时能带来收益以及常见的错误来源。

**适用性**。细粒度同步可以缩短核函数之间的交接。
它有利于这样的场景：消费者线程块可以在生产者最后的线程块仍在运行时启动的短核函数链。
长的、受吞吐限制的核函数收益很小，而且它们仍需付出初始化的代价。

**残留的哨兵值**。如果拷贝所依赖的某个位置一直保持哨兵值，消费者可能挂起，
例如未写入的填充（参见 :ref:`第 4.6.4.6 节 <fgs-partial-tiles-and-padding>`），
或者当 NaN 为哨兵值时出现了 NaN 结果（参见 :ref:`第 4.6.4.3 节 <fgs-producer-rules>`）。
调试时，一个在重新发出足够多次之后陷入陷阱（trap）的重试循环可以把这种挂起转换为主机可观察到的错误。

**缺少初始化**。在每次使用前没有用哨兵值初始化的缓冲区可能导致错误结果而不是挂起。
拷贝对陈旧数据不报告任何内容，消费者会把它当作就绪而接受。
