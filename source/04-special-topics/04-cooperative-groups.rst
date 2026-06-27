.. _cooperative-groups:

4.4. Cooperative Groups
========================

.. _cg-introduction:

4.4.1. 简介
-----------

协作组（Cooperative Groups）是对 CUDA 编程模型的一种扩展，用于组织相互协作的线程组。
协作组允许开发者控制线程协作的粒度，从而帮助他们实现更丰富、更高效的并行分解。
此外，协作组还提供了常见并行原语（如 `scan` 和 `parallel reduce` ）的实现。

曾经，CUDA 编程模型为同步协作线程只提供了一种单一且简单的机制：即通过 ``__syncthreads()`` 内置函数实现单个线程块内所有线程的屏障。
为了表达更广泛的并行交互模式，许多追求极致性能的程序员不得不自己编写一些临时的且不安全的原语，用来同步单个线程束（warp）内的线程，或者同步运行在单个 GPU 上的多个线程块集合。
虽然这些做法能带来可观的性能提升，但也导致了一系列脆弱的代码不断积累，这些代码编写、调优的成本高昂，且随着时间的推移和 GPU 架构的迭代，维护起来愈发困难。
协作组提供了一种安全且面向未来的机制，使开发者能够编写出兼具高性能与可靠性的代码。

.. _cg-handle-member-functions:

4.4.2. Cooperative Group 句柄与成员函数
----------------------------------------

协作组通过协作组句柄（Cooperative Group Handle）进行管理。
借助协作组句柄，组内参与线程可获取自身在组内的位置、组大小以及其他组相关信息。
下表展示了部分常用的成员函数。

.. _tbl:cg-member-functions:

.. list-table:: 常用成员函数
   :header-rows: 1
   :widths: 30 70
   :align: center

   * - 访问器
     - 返回值
   * - ``thread_rank()``
     - 调用线程的排名。
   * - ``num_threads()``
     - 组中的线程总数。
   * - ``thread_index()``
     - 线程在启动块内的三维索引。
   * - ``dim_threads()``
     - 启动块的三维尺寸（以线程为单位）。

完整的协作组接口请参考 :ref:`Cooperative Groups API<cooperative-groups-partition-h>` 。

.. _cg-default-behavior:

4.4.3. 默认行为 / 无组执行
---------------------------

表示网格与线程块的线程组会根据 kernel 的启动配置自动隐式创建。
这类 `隐式组` 为开发者提供了一个起点，开发者可在此基础上显式拆分出粒度更细的线程组。
可通过以下接口访问隐式线程组：


.. _tbl:cg-implicit-groups:

.. list-table:: CUDA 运行时隐式创建的 Cooperative Groups
   :header-rows: 1
   :widths: 40 60
   :align: center

   * - 访问器
     - 组作用域
   * - ``this_thread_block()``
     - 返回包含当前线程块中所有线程的组句柄。
   * - ``this_grid()``
     - 返回包含 grid 中所有线程的组句柄。
   * - ``coalesced_threads()`` [1]_
     - 返回 warp 中当前活动线程组的句柄。
   * - ``this_cluster()`` [2]_
     - 返回当前 cluster 中线程组的句柄。

.. [1] ``coalesced_threads()`` 返回当前时刻处于活跃状态的线程集合。不保证具体返回哪些线程（只要是活跃的即可），也不保证这些线程在整个执行过程中会始终保持合并（coalesced）状态。

.. [2] ``this_cluster()`` 在非集群网格，默认采用 ``1×1×1`` 的线程集群配置。该接口要求计算能力不低于 9.0。

更多信息请参考 :ref:`Cooperative Groups API<cooperative-groups-partition-h>` 。

.. _cg-create-early:

4.4.3.1. 尽早创建隐式组句柄
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了获得最佳性能，建议尽早（在任何分支发生之前）为隐式组创建一个句柄，并在整个 kernel 中复用该句柄。


.. _cg-pass-by-reference:

4.4.3.2. 仅通过引用传递组句柄
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将线程组句柄传入函数时，建议以引用方式传递。
线程组句柄必须在声明时完成初始化，因其不存在默认构造函数。不推荐通过拷贝构造的方式创建线程组句柄。

.. _cg-creating-groups:

4.4.4. 创建 Cooperative Groups
-------------------------------

线程组通过将父组划分成若干子组来创建。对线程组执行划分操作时，系统会生成对应的句柄用于管理划分后的子组。开发者可使用以下划分接口：

.. _tbl:cg-partition-operations:

.. list-table:: Cooperative Group 划分操作
   :header-rows: 1
   :widths: 25 75
   :align: center

   * - 划分类型
     - 描述
   * - tiled_partition
     - 将父组划分为一系列固定大小的子组，按一维行优先格式排列。
   * - labeled_partition
     - 基于条件标签将父组划分为一维子组，标签可以是任何整数类型。
   * - binary_partition
     - labeled partition 的特殊形式，标签只能是 `0` 或 `1` 。

以下示例展示了如何创建 tiled partition：

.. code-block:: cuda

   namespace cg = cooperative_groups;
   // Obtain the current thread's cooperative group
   cg::thread_block my_group = cg::this_thread_block();

   // Partition the cooperative group into tiles of size 8
   cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(my_group);

   // do work as my_subgroup

最佳划分策略取决于上下文。更多信息请参考 :ref:`Cooperative Groups API<cooperative-groups-partition-h>` 。

.. _cg-creation-hazards:

4.4.4.1. 线程组创建避坑
~~~~~~~~~~~~~~~~~~~~~~~~

划分组是一项集体操作，组内的所有线程都必须参与。如果组是在并非所有线程都能执行到的条件分支中创建的，这可能会导致死锁或数据损坏。

.. _cg-synchronization:

4.4.5. 同步
------------

在引入协作组之前，CUDA 编程模型仅允许在核函数执行完毕的边界处进行线程块同步。而协作组允许开发者在不同的粒度上对协作线程组进行同步。”

.. _cg-sync:

4.4.5.1. Sync
~~~~~~~~~~~~~

开发者可以通过调用 ``sync()`` 函数来同步协作组。与 ``__syncthreads()`` 类似， ``sync()`` 函数提供以下保证：

- 同步点之前，组内各线程完成的所有内存访问操作（包括读、写），在同步点之后的对组内全部线程可见。
- 组内所有线程都抵达同步点之后，任意线程才允许执行同步点之后的代码。

以下示例展示了与 ``__syncthreads()`` 等效的 ``cooperative_groups::sync()`` ：

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = cg::this_thread_block();

   // Synchronize threads in the block
   cg::sync(my_group);


协作组可用于对整个线程网格执行同步。自 CUDA 13 版本起，协作组不再支持多设备同步功能。详情可参阅 :ref:`cg-large-scale-groups` 。


.. _cg-barriers:

4.4.5.2. Barriers
~~~~~~~~~~~~~~~~~

协作组提供了一套与 ``cuda::barrier`` 类似的屏障接口，可用于实现更复杂的同步逻辑。协作组屏障接口与 ``cuda::barrier`` 存在几处关键区别：

- 协作组屏障会自动初始化
- 每个阶段内，组内所有线程都必须抵达屏障并在此等待一次。
- ``barrier_arrive`` 会返回一个 ``arrival_token`` 对象，该对象必须传递给相应的 ``barrier_wait`` 函数，并被消耗掉，无法再次使用。

开发者在使用协作组屏障时，必须留意规避各类程序风险：

- 在调用 ``barrier_arrive`` 之后、调用 ``barrier_wait`` 之前，该线程组不得执行任何集合操作。
- ``barrier_wait`` 仅保证组内所有线程均已调用 ``barrier_arrive`` ，并不保证所有线程都执行到了 ``barrier_wait`` 。

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = this_block();
   cg::cluster_group cluster = this_cluster();

   auto token = cluster.barrier_arrive();

   // Optional: Do some local processing to hide the synchronization latency
      local_processing(my_group);

   // Make sure all other blocks in the cluster are running and
   // initialized shared data before accessing dsmem
   cluster.barrier_wait(std::move(token));

.. _cg-collective-operations:

4.4.6. 集合操作
----------------

协作组内置了一组可供线程组执行的集合操作。这类操作必须由指定组内的全部线程共同参与，操作才能正常完成。

对于每一次集合操作调用，组内所有线程传入对应参数的值必须保持一致，除非 :ref:`协作组接口<cg-large-scale-groups>` 明确允许传入不同数值。否则该调用将产生未定义行为。

.. _cg-reduce:

4.4.6.1. Reduce
~~~~~~~~~~~~~~~

``reduce`` （归约）函数用于对指定组内每个线程提供的数据执行并行归约操作。归约的类型由下表所示的运算符指定。

.. _tbl:cg-reduction-operators:

.. list-table:: Cooperative Groups 归约操作符
   :header-rows: 1
   :widths: 25 75
   :align: center

   * - 操作符
     - 返回值
   * - plus
     - 组中所有值的总和
   * - less
     - 最小值
   * - greater
     - 最大值
   * - bit_and
     - 按位与归约
   * - bit_or
     - 按位或归约
   * - bit_xor
     - 按位异或归约

当硬件支持时，归约操作会使用硬件加速（要求计算能力达到 8.0+）。
对于不支持硬件加速的旧硬件，系统提供了软件回退机制。
此外，只有 4 字节（4B）的数据类型才能被硬件加速。

更多信息请参考 :ref:`reduce-operators` 。

下方示例演示如何使用 ``cooperative_groups::reduce()`` 完成线程块范围内的求和归约运算。

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = cg::this_thread_block();

   int val = data[threadIdx.x];

   int sum = cg::reduce(my_group, val, cg::plus<int>());

   // Store the result from the reduction
   if (my_group.thread_rank() == 0) {
     result[blockIdx.x] = sum;
   }


.. _cg-scans:

4.4.6.2. Scans
~~~~~~~~~~~~~~~

协作组提供了 ``inclusive_scan`` 与 ``exclusive_scan`` 的实现，可适用于任意大小的线程组。
这两个函数会针对指定线程组内每个线程传入的数据执行 ``scan`` 运算。

程序员可以选择指定归约操作符，如上 :ref:`tbl:cg-reduction-operators` 表中所列。

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = cg::this_thread_block();

   int val = data[my_group.thread_rank()];

   int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());

   result[my_group.thread_rank()] = exclusive_sum;


更多信息请参考 :ref:`inclusive-scan-and-exclusive-scan` 。

.. _cg-invoke-one:

4.4.6.3. Invoke One
~~~~~~~~~~~~~~~~~~~~

协作组提供了 ``invoke_one`` 函数，适用于需要由单个线程代表整个线程组串行执行一段任务的场景。

- ``invoke_one`` 会从调用该函数的线程组中任选一个线程，并使用该线程，结合传入的参数执行用户提供的可调用函数。
- ``invoke_one_broadcast`` 与 ``invoke_one`` 类似，唯一的区别在于：该函数调用的结果会被广播给组内的所有线程。

线程的选择机制不保证是确定性的。

以下示例展示了基本的 ``invoke_one`` 用法：

.. code-block:: cuda

   namespace cg = cooperative_groups;
   cg::thread_block my_group = cg::this_thread_block();

   // Ensure only one thread in the thread block prints the message
   cg::invoke_one(my_group, []() {
     printf("Hello from one thread in the block!");
   });

   // Synchronize to make sure all threads wait until the message is printed
   cg::sync(my_group);

在用户提供的可调用函数内部不允许与调用组内部进行通信或同步。但是允许与调用组之外的线程进行通信。

.. _cg-async-data-movement:

4.4.7. 异步数据移动
--------------------

CUDA 中协作组的 ``memcpy_async`` 功能可实现在全局内存与共享内存之间执行异步内存拷贝。
这对优化数据传输、让计算与数据传输并行执行以提升性能方面很有用。

``memcpy_async`` 函数用于启动一个从全局内存到共享内存的异步加载操作。
其设计初衷是提供类似 `预取（prefetch）` 的机制，使在数据在真正使用之前，提前加载好。

``wait`` 函数会强制线程组内所有线程等待，直至异步内存传输完成。
在线程访问共享内存中的数据前，组内所有线程都必须调用 ``wait`` 。

以下示例展示了如何使用 ``memcpy_async`` 和 ``wait`` 来预取数据：

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_group my_group = cg::this_thread_block();

   __shared__ int shared_data[];

   // Perform an asynchronous copy from global memory to shared memory
   cg::memcpy_async(my_group,
                    shared_data + my_group.rank(),
                    input + my_group.rank(),
                    sizeof(int));

   // Hide latency by doing work here. Cannot use shared_data

   // Wait for the asynchronous copy to complete
   cg::wait(my_group);

   // Prefetched data is now available

有关更多信息，请参阅 :ref:`memcpy-async` 。

.. _cg-memcpy-async-alignment:

4.4.7.1. Memcpy Async 对齐要求
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

仅当源地址为全局内存、目标地址为共享内存，且二者均至少是 4 字节对齐时， ``memcpy_async`` 才是异步的。
若要达到最优性能，建议全局内存与共享内存地址均采用 16 字节对齐

.. _cg-large-scale-groups:

4.4.8. Large Scale Groups
--------------------------

协作组支持创建覆盖整个网格的超大范围线程组（Large Scale Groups）。
前文介绍的所有协作组功能均可在这类大线程组上使用，但有一处需要重要注意：若要对整个网格执行同步，必须使用接口 ``cudaLaunchCooperativeKernel`` 来启动核函数。

自 CUDA 13 版本起，协作组相关的多设备启动接口及其配套参考内容已被移除。

.. _cg-cooperative-kernel:

4.4.8.1. 什么时候使用 ``cudaLaunchCooperativeKernel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaLaunchCooperativeKernel`` 是一个 CUDA Runtime 接口，用于启动使用了协作组特性的单设备核函数（single-device kernel），尤其是需要跨线程块同步的核函数。
该函数能够确保核函数中的所有线程在整个网格范围内进行同步与协作，而这在传统 CUDA 核函数中是无法实现的（传统核函数仅允许在单个线程块内部进行同步）。
此外， ``cudaLaunchCooperativeKernel`` 保证了核函数启动的原子性：即如果该 API 调用成功，指定数量的线程块将全部在目标设备上启动。

良好的编程实践是：首先通过查询设备属性 ``cudaDevAttrCooperativeLaunch`` ，来确认当前设备是否支持该特性。

.. code-block:: cuda

   int dev = 0;
   int supportsCoopLaunch = 0;
   cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

如果设备 0 支持该属性，该查询会将 ``supportsCoopLaunch`` 赋值为 1。
仅计算能力 6.0 及以上的显卡支持协作启动。除此之外，程序还需在以下环境中运行：

- 不带 MPS 的 Linux 平台
- 带 MPS 的 Linux 平台，且设备计算能力为 7.0 或更高版本
- 最新的 Windows 平台
