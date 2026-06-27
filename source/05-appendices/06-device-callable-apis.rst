.. _device-callable-apis:

5.6. 设备可调用 API 和内建函数
==============================

本章节包含可从 CUDA 内核和设备代码调用的 API 和内建函数的参考材料和 API 文档。

.. _memory-barrier-primitives-interface:

5.6.1. 内存屏障原语接口
-----------------------

原语 API 是 ``cuda::barrier`` 功能的 C 语言接口。通过包含 ``<cuda_awbarrier_primitives.h>`` 头文件可以使用这些原语。

.. _data-types:

5.6.1.1. 数据类型
^^^^^^^^^^^^^^^^^

.. code-block:: c++

   typedef /* 实现定义 */ __mbarrier_t;
   typedef /* 实现定义 */ __mbarrier_token_t;

.. _memory-barrier-primitives-api:

5.6.1.2. 内存屏障原语 API
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   uint32_t __mbarrier_maximum_count();
   void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);

- ``bar`` 必须是指向 ``__shared__`` 内存的指针。
- ``expected_count <= __mbarrier_maximum_count()`` 。
- 初始化 ``*bar`` 当前和下一阶段的预期到达计数为 ``expected_count`` 。

.. code-block:: c++

   void __mbarrier_inval(__mbarrier_t* bar);

- ``bar`` 必须是指向驻留在共享内存中的屏障对象的指针。
- 在相应的共享内存可以被重新分配用途之前，需要使 ``*bar`` 失效。

.. code-block:: c++

   __mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);

- 在此调用之前必须发生 ``*bar`` 的初始化。
- 待处理计数必须不为零。
- 原子递减屏障当前阶段的待处理计数。
- 返回与递减之前立即屏障状态关联的到达令牌。

.. code-block:: c++

   __mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);

- 在此调用之前必须发生 ``*bar`` 的初始化。
- 待处理计数必须不为零。
- 原子递减屏障当前阶段的待处理计数和下一阶段的预期计数。
- 返回与递减之前立即屏障状态关联的到达令牌。

.. code-block:: c++

   bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);

- ``token`` 必须与 ``*bar`` 的紧接前一阶段或当前阶段关联。
- 如果 ``token`` 与 ``*bar`` 的紧接前一阶段关联，则返回 ``true`` ，否则返回 ``false`` 。

.. code-block:: c++

   bool __mbarrier_test_wait_parity(__mbarrier_t* bar, bool phase_parity);

- ``phase_parity`` 必须指示 ``*bar`` 的当前阶段或紧接前一阶段的奇偶性。值 ``true`` 对应于奇数编号的阶段，值 ``false`` 对应于偶数编号的阶段。
- 如果 ``phase_parity`` 指示 ``*bar`` 的紧接前一阶段的整数奇偶性，则返回 ``true`` ，否则返回 ``false`` 。

.. code-block:: c++

   bool __mbarrier_try_wait(__mbarrier_t* bar, __mbarrier_token_t token, uint32_t max_sleep_nanosec);

- ``token`` 必须与 ``*bar`` 的紧接前一阶段或当前阶段关联。
- 如果 ``token`` 与 ``*bar`` 的紧接前一阶段关联，则返回 ``true`` 。否则，执行的线程可能会被挂起。挂起的线程在指定阶段完成时恢复执行（返回 ``true`` ），或者在阶段完成之前达到系统相关的时间限制（返回 ``false`` ）。
- ``max_sleep_nanosec`` 指定时间限制（以纳秒为单位），可用于替代系统相关的时间限制。

.. code-block:: c++

   bool __mbarrier_try_wait_parity(__mbarrier_t* bar, bool phase_parity, uint32_t max_sleep_nanosec);

- ``phase_parity`` 必须指示 ``*bar`` 的当前阶段或紧接前一阶段的奇偶性。值 ``true`` 对应于奇数编号的阶段，值 ``false`` 对应于偶数编号的阶段。
- 如果 ``phase_parity`` 指示 ``*bar`` 的紧接前一阶段的整数奇偶性，则返回 ``true`` 。否则，执行的线程可能会被挂起。挂起的线程在指定阶段完成时恢复执行（返回 ``true`` ），或者在阶段完成之前达到系统相关的时间限制（返回 ``false`` ）。
- ``max_sleep_nanosec`` 指定时间限制（以纳秒为单位），可用于替代系统相关的时间限制。

.. _pipeline-primitives-interface:

5.6.2. 流水线原语接口
---------------------

流水线原语为 ``<cuda/pipeline>`` 中可用的功能提供 C 语言接口。通过包含 ``<cuda_pipeline.h>`` 头文件可以使用流水线原语接口。在不使用 ISO C++ 2011 兼容性编译时，包含 ``<cuda_pipeline_primitives.h>`` 头文件。

.. note::

   流水线原语 API 仅支持跟踪从全局内存到共享内存的异步复制，具有特定的大小和对齐要求。它提供与 ``cuda::thread_scope_thread`` 的 ``cuda::pipeline`` 对象等效的功能。

.. _memcpy-async-primitive:

5.6.2.1. ``memcpy_async`` 原语
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                                const void* __restrict__ src_global,
                                size_t size_and_align,
                                size_t zfill=0);

- 请求提交以下操作以进行异步求值：

  .. code-block:: c++

     size_t i = 0;
     for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* 复制 */
     for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* 零填充 */

- 要求：
  
  - ``dst_shared`` 必须是指向 ``memcpy_async`` 共享内存目的地的指针。
  - ``src_global`` 必须是指向 ``memcpy_async`` 全局内存源的指针。
  - ``size_and_align`` 必须是 4、8 或 16。
  - ``zfill <= size_and_align`` 。
  - ``size_and_align`` 必须是 ``dst_shared`` 和 ``src_global`` 的对齐。

- 在等待 ``memcpy_async`` 操作完成之前，任何线程修改源内存或观察目标内存都是竞争条件。在提交 ``memcpy_async`` 操作和等待其完成之间，以下任何操作都会引入竞争条件：
  
  - 从 ``dst_shared`` 加载。
  - 存储到 ``dst_shared`` 或 ``src_global`` 。
  - 对 ``dst_shared`` 或 ``src_global`` 应用原子更新。

.. _commit-primitive:

5.6.2.2. 提交原语
^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __pipeline_commit();

- 将提交的 ``memcpy_async`` 作为当前批次提交到流水线。

.. _wait-primitive:

5.6.2.3. 等待原语
^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __pipeline_wait_prior(size_t N);

- 设 ``{0, 1, 2, ..., L}`` 是与给定线程调用 ``__pipeline_commit()`` 关联的索引序列。
- 等待*至少*到并包括 ``L-N`` 的批次完成。

.. _arrive-on-barrier-primitive:

5.6.2.4. 到达屏障原语
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __pipeline_arrive_on(__mbarrier_t* bar);

- ``bar`` 指向共享内存中的屏障。
- 当此调用之前排序的所有 memcpy_async 操作完成时，屏障到达计数递增一，到达计数递减一，因此对到达计数的净效果为零。用户有责任确保到达计数的递增不超过 ``__mbarrier_maximum_count()`` 。

.. _cooperative-groups-api:

5.6.3. Cooperative Groups API
-----------------------------

.. _cooperative-groups-h:

5.6.3.1. cooperative_groups.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _class-thread-block:

5.6.3.1.1. thread_block 类
^^^^^^^^^^^^^^^^^^^^^^^^^^

任何 CUDA 程序员都已经熟悉一组线程：线程块。Cooperative Groups 扩展引入了一种新数据类型 ``thread_block`` ，以在内核中显式表示此概念。

.. code-block:: c++

   class thread_block;

通过以下方式构造：

.. code-block:: c++

   thread_block g = this_thread_block();

**公共成员函数：**

``static void sync()`` ：同步组中命名的线程，等效于 ``g.barrier_wait(g.barrier_arrive())`` 。

``thread_block::arrival_token barrier_arrive()`` ：到达 thread_block 屏障，返回需要传递给 ``barrier_wait()`` 的令牌。

``void barrier_wait(thread_block::arrival_token&& t)`` ：等待 thread_block 屏障，将 ``barrier_arrive()`` 返回的到达令牌作为右值引用。

``static unsigned int thread_rank()`` ：调用线程在 [0, num_threads) 范围内的秩。

``static dim3 group_index()`` ：块在启动 grid 中的 3 维索引。

``static dim3 thread_index()`` ：线程在启动块中的 3 维索引。

``static dim3 dim_threads()`` ：启动块以线程为单位的维度。

``static unsigned int num_threads()`` ：组中线程的总数。

**遗留成员函数（别名）：**

``static unsigned int size()`` ：组中线程的总数（ ``num_threads()`` 的别名）。

``static dim3 group_dim()`` ：启动块的维度（ ``dim_threads()`` 的别名）。

**示例：**

.. code-block:: c++

   /// 从全局内存加载整数到共享内存
   __global__ void kernel(int *globalInput) {
       __shared__ int x;
       thread_block g = this_thread_block();
       // 在线程块中选择一个领导者
       if (g.thread_rank() == 0) {
           // 从全局加载到共享，供所有线程使用
           x = (*globalInput);
       }
       // 在将数据加载到共享内存后，如果要同步，
       // 如果线程块中的所有线程都需要看到它
       g.sync(); // 等效于 __syncthreads();
   }

.. note::

   组中的所有线程都必须参与集体操作，否则行为未定义。

.. _class-cluster-group:

5.6.3.1.2. cluster_group 类
^^^^^^^^^^^^^^^^^^^^^^^^^^^

此组对象表示在单个集群中启动的所有线程。这些 API 在计算能力 9.0+ 的所有硬件上可用。在这种情况下，当启动非集群 grid 时，API 假设为 1x1x1 集群。

.. code-block:: c++

   class cluster_group;

通过以下方式构造：

.. code-block:: c++

   cluster_group g = this_cluster();

**公共成员函数：**

``static void sync()`` ：同步组中命名的线程，等效于 ``g.barrier_wait(g.barrier_arrive())`` 。

``static cluster_group::arrival_token barrier_arrive()`` ：到达集群屏障，返回需要传递给 ``barrier_wait()`` 的令牌。

``static void barrier_wait(cluster_group::arrival_token&& t)`` ：等待集群屏障，将 ``barrier_arrive()`` 返回的到达令牌作为右值引用。

``static unsigned int thread_rank()`` ：调用线程在 [0, num_threads) 范围内的秩。

``static unsigned int block_rank()`` ：调用块在 [0, num_blocks) 范围内的秩。

``static unsigned int num_threads()`` ：组中线程的总数。

``static unsigned int num_blocks()`` ：组中块的总数。

``static dim3 dim_threads()`` ：启动集群以线程为单位的维度。

``static dim3 dim_blocks()`` ：启动集群以块为单位的维度。

``static dim3 block_index()`` ：调用块在启动集群中的 3 维索引。

``static unsigned int query_shared_rank(const void *addr)`` ：获取共享内存地址所属的块秩。

``static T* map_shared_rank(T *addr, int rank)`` ：获取集群中另一个块的共享内存变量地址。

.. _class-grid-group:

5.6.3.1.3. grid_group 类
^^^^^^^^^^^^^^^^^^^^^^^^

此组对象表示在单个 grid 中启动的所有线程。除 ``sync()`` 外的 API 始终可用，但要在 grid 范围内同步，需要使用协作启动 API。

.. code-block:: c++

   class grid_group;

通过以下方式构造：

.. code-block:: c++

   grid_group g = this_grid();

**公共成员函数：**

``bool is_valid() const`` ：返回 grid_group 是否可以同步。

``void sync() const`` ：同步组中命名的线程，等效于 ``g.barrier_wait(g.barrier_arrive())`` 。

``grid_group::arrival_token barrier_arrive()`` ：到达 grid 屏障，返回需要传递给 ``barrier_wait()`` 的令牌。

``void barrier_wait(grid_group::arrival_token&& t)`` ：等待 grid 屏障，将 ``barrier_arrive()`` 返回的到达令牌作为右值引用。

``static unsigned long long thread_rank()`` ：调用线程在 [0, num_threads) 范围内的秩。

``static unsigned long long block_rank()`` ：调用块在 [0, num_blocks) 范围内的秩。

``static unsigned long long cluster_rank()`` ：调用集群在 [0, num_clusters) 范围内的秩。

``static unsigned long long num_threads()`` ：组中线程的总数。

``static unsigned long long num_blocks()`` ：组中块的总数。

``static unsigned long long num_clusters()`` ：组中集群的总数。

``static dim3 dim_blocks()`` ：启动 grid 以块为单位的维度。

``static dim3 dim_clusters()`` ：启动 grid 以集群为单位的维度。

``static dim3 block_index()`` ：块在启动 grid 中的 3 维索引。

``static dim3 cluster_index()`` ：集群在启动 grid 中的 3 维索引。

.. _class-thread-block-tile:

5.6.3.1.4. thread_block_tile 类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

平铺组的模板化版本，其中使用模板参数指定 tile 的大小——在编译时已知这一点，有可能实现更优的执行。

.. code-block:: c++

   template <unsigned int Size, typename ParentT = void>
   class thread_block_tile;

通过以下方式构造：

.. code-block:: c++

   template <unsigned int Size, typename ParentT>
   _CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)

``Size`` 必须是 2 的幂，并且小于或等于 1024。注释部分描述了在计算能力 7.5 或更低版本的硬件上创建大于 32 的 tile 所需的额外步骤。

``ParentT`` 是从中分区此组的父类型。它是自动推断的，但 ``void`` 值将此信息存储在组句柄中而不是类型中。

**公共成员函数：**

``void sync() const`` ：同步组中命名的线程。

``unsigned long long num_threads() const`` ：组中线程的总数。

``unsigned long long thread_rank() const`` ：调用线程在 [0, num_threads) 范围内的秩。

``unsigned long long meta_group_size() const`` ：返回父组分区时创建的组数。

``unsigned long long meta_group_rank() const`` ：组在从父组分区的 tile 集合中的线性秩（由 meta_group_size 限定）。

``T shfl(T var, unsigned int src_rank) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__，**注意：对于大于 32 的大小，组中的所有线程必须指定相同的 src_rank，否则行为未定义。**

``T shfl_up(T var, int delta) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__，仅适用于小于或等于 32 的大小。

``T shfl_down(T var, int delta) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__，仅适用于小于或等于 32 的大小。

``T shfl_xor(T var, int delta) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__，仅适用于小于或等于 32 的大小。

``int any(int predicate) const`` ：参见 `Warp Vote 函数 <index.html#warp-vote-functions>`__。

``int all(int predicate) const`` ：参见 `Warp Vote 函数 <index.html#warp-vote-functions>`__。

``unsigned int ballot(int predicate) const`` ：参见 `Warp Vote 函数 <index.html#warp-vote-functions>`__，仅适用于小于或等于 32 的大小。

``unsigned int match_any(T val) const`` ：参见 `Warp Match 函数 <cpp-language-extensions.html#warp-match-functions>`__，仅适用于小于或等于 32 的大小。

``unsigned int match_all(T val, int &pred) const`` ：参见 `Warp Match 函数 <cpp-language-extensions.html#warp-match-functions>`__，仅适用于小于或等于 32 的大小。

**示例：**

.. code-block:: c++

   /// 以下代码将创建两组平铺组，大小分别为 32 和 4：
   /// 后者将来源编码在类型中，而前者将其存储在句柄中
   thread_block block = this_thread_block();
   thread_block_tile<32> tile32 = tiled_partition<32>(block);
   thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);

   /// 以下代码将在所有计算能力上创建大小为 128 的 tile。
   /// block_tile_memory 可以在计算能力 8.0 或更高版本上省略。
   __global__ void kernel(...) {
       // 为 thread_block_tile 使用保留共享内存，
       //   指定块大小最多为 256 个线程。
       __shared__ block_tile_memory<256> shared;
       thread_block thb = this_thread_block(shared);

       // 创建具有 128 个线程的 tile。
       auto tile = tiled_partition<128>(thb);

       // ...
   }

.. _class-coalesced-group:

5.6.3.1.5. coalesced_group 类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 CUDA 的 SIMT 架构中，在硬件级别，多处理器以称为 warp 的 32 个线程组执行线程。如果应用程序代码中存在数据相关的条件分支，使得 warp 内的线程分叉，则 warp 串行执行每个分支，禁用不在该路径上的线程。在该路径上保持活动的线程称为合并的。Cooperative Groups 具有发现和创建包含所有合并线程的组的功能。

通过 ``coalesced_threads()`` 构造组句柄是机会主义的。它返回当时处于活动状态的线程集，并且不保证返回哪些线程（只要它们是活动的）或它们将在整个执行过程中保持合并（它们将被重新组合在一起以执行集体操作，但之后可能再次分叉）。

.. code-block:: c++

   class coalesced_group;

通过以下方式构造：

.. code-block:: c++

   coalesced_group active = coalesced_threads();

**公共成员函数：**

``void sync() const`` ：同步组中命名的线程。

``unsigned long long num_threads() const`` ：组中线程的总数。

``unsigned long long thread_rank() const`` ：调用线程在 [0, num_threads) 范围内的秩。

``unsigned long long meta_group_size() const`` ：返回父组分区时创建的组数。如果此组是通过查询活动线程集创建的，例如 ``coalesced_threads()`` ，则 ``meta_group_size()`` 的值将为 1。

``unsigned long long meta_group_rank() const`` ：组在从父组分区的 tile 集合中的线性秩（由 meta_group_size 限定）。如果此组是通过查询活动线程集创建的，例如 ``coalesced_threads()`` ，则 ``meta_group_rank()`` 的值将始终为 0。

``T shfl(T var, unsigned int src_rank) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__。

``T shfl_up(T var, int delta) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__。

``T shfl_down(T var, int delta) const`` ：参见 `Warp Shuffle 函数 <cpp-language-extensions.html#warp-shuffle-functions>`__。

``int any(int predicate) const`` ：参见 `Warp Vote 函数 <index.html#warp-vote-functions>`__。

``int all(int predicate) const`` ：参见 `Warp Vote 函数 <index.html#warp-vote-functions>`__。

``unsigned int ballot(int predicate) const`` ：参见 `Warp Vote 函数 <index.html#warp-vote-functions>`__。

``unsigned int match_any(T val) const`` ：参见 `Warp Match 函数 <cpp-language-extensions.html#warp-match-functions>`__。

``unsigned int match_all(T val, int &pred) const`` ：参见 `Warp Match 函数 <cpp-language-extensions.html#warp-match-functions>`__。

**示例：**

.. code-block:: c++

   /// 考虑这样一种情况：代码中有一个分支，
   /// 其中每个 warp 中只有第 2、4 和 8 个线程是活动的。
   /// 放置在该分支中的 coalesced_threads() 调用将为每个
   /// warp 创建一个组 active，它有三个线程（秩为 0-2）。
   __global__ void kernel(int *globalInput) {
       // 假设 globalInput 表示线程 2、4、8 应该处理数据
       if (threadIdx.x == *globalInput) {
           coalesced_group active = coalesced_threads();
           // active 包含 0-2（含）
           active.sync();
       }
   }

.. _cooperative-groups-async-h:

5.6.3.2. cooperative_groups/async.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _memcpy-async:

5.6.3.2.1. ``memcpy_async``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``memcpy_async`` 是组范围的集体 memcpy，利用硬件加速支持从全局内存到共享内存的非阻塞内存事务。给定组中命名的一组线程， ``memcpy_async`` 将通过单个流水线阶段移动指定数量的字节或输入类型的元素。此外，为了在使用 ``memcpy_async`` API 时实现最佳性能，共享内存和全局内存都需要 16 字节的对齐。重要的是要注意，虽然这通常是一个 memcpy，但只有当源是全局内存且目标是共享内存并且两者都可以用 16、8 或 4 字节对齐寻址时，它才是异步的。异步复制的数据只能在调用 wait 或 wait_prior 后读取，这表示相应阶段已完成将数据移动到共享内存。

必须等待所有未完成的请求可能会失去一些灵活性（但获得简单性）。为了有效地重叠数据传输和执行，重要的是能够在等待和操作请求 **N** 的同时启动 **N+1** 个 ``memcpy_async`` 请求。为此，使用 ``memcpy_async`` 并使用基于阶段的集体 ``wait_prior`` API 等待它。有关更多详细信息，请参阅 `wait 和 wait_prior <#cg-api-async-wait>`__。

**用法 1**

.. code-block:: c++

   template <typename TyGroup, typename TyElem, typename TyShape>
   void memcpy_async(
     const TyGroup &group,
     TyElem *__restrict__ _dst,
     const TyElem *__restrict__ _src,
     const TyShape &shape
   );

复制 ``shape`` 字节。

**用法 2**

.. code-block:: c++

   template <typename TyGroup, typename TyElem, typename TyDstLayout, typename TySrcLayout>
   void memcpy_async(
     const TyGroup &group,
     TyElem *__restrict__ dst,
     const TyDstLayout &dstLayout,
     const TyElem *__restrict__ src,
     const TySrcLayout &srcLayout
   );

复制 ``min(dstLayout, srcLayout)`` 元素。如果布局类型为 ``cuda::aligned_size_t<N>`` ，两者必须指定相同的对齐。

**勘误** CUDA 11.1 中引入的带有 src 和 dst 输入布局的 ``memcpy_async`` API 期望布局以元素而不是字节提供。元素类型从 ``TyElem`` 推断，大小为 ``sizeof(TyElem)`` 。如果使用 ``cuda::aligned_size_t<N>`` 类型作为布局，则指定的元素数量乘以 ``sizeof(TyElem)`` 必须是 N 的倍数，建议使用 ``std::byte`` 或 ``char`` 作为元素类型。

如果指定的复制形状或布局类型为 ``cuda::aligned_size_t<N>`` ，则对齐将保证至少为 ``min(16, N)`` 。在这种情况下， ``dst`` 和 ``src`` 指针都需要对齐到 N 字节，复制的字节数需要是 N 的倍数。

**代码生成要求：** 最低计算能力 5.0，计算能力 8.0 以实现异步，C++11

需要包含 ``cooperative_groups/memcpy_async.h`` 头文件。

**示例：**

.. code-block:: c++

   /// 此示例从全局内存流式传输每个线程块的数据
   /// 到有限大小的共享内存块（elementsInShared）中进行操作。
   #include <cooperative_groups.h>
   #include <cooperative_groups/memcpy_async.h>

   namespace cg = cooperative_groups;

   __global__ void kernel(int* global_data) {
       cg::thread_block tb = cg::this_thread_block();
       const size_t elementsPerThreadBlock = 16 * 1024;
       const size_t elementsInShared = 128;
       __shared__ int local_smem[elementsInShared];

       size_t copy_count;
       size_t index = 0;
       while (index < elementsPerThreadBlock) {
           cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
           copy_count = min(elementsInShared, elementsPerThreadBlock - index);
           cg::wait(tb);
           // 使用 local_smem 工作
           index += copy_count;
       }
   }

.. _wait-and-wait-prior:

5.6.3.2.2. ``wait`` 和 ``wait_prior``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   template <typename TyGroup>
   void wait(TyGroup & group);

   template <unsigned int NumStages, typename TyGroup>
   void wait_prior(TyGroup & group);

``wait`` 和 ``wait_prior`` 集体操作允许等待 memcpy_async 复制完成。 ``wait`` 阻塞调用线程直到所有先前的复制完成。 ``wait_prior`` 允许最新的 NumStages 可能仍未完成，并等待所有先前的请求。因此，对于总共请求的 N 个复制，它等待直到前 N-NumStages 个完成，最后 NumStages 可能仍在进行中。 ``wait`` 和 ``wait_prior`` 都将同步命名的组。

**代码生成要求：** 最低计算能力 5.0，计算能力 8.0 以实现异步，C++11

需要包含 ``cooperative_groups/memcpy_async.h`` 头文件。

.. _cooperative-groups-partition-h:

5.6.3.3. cooperative_groups/partition.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _tiled-partition:

5.6.3.3.1. ``tiled_partition``
""""""""""""""""""""""""""""""

.. code-block:: c++

   template <unsigned int Size, typename ParentT>
   thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);

.. code-block:: c++

   thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);

``tiled_partition`` 方法是一个集体操作，它将父组分区为一维的、行主序的 subgroups tile。总共将创建 ((size(parent)/tilesz) 个 subgroups，因此父组的大小必须能被 ``Size`` 整除。允许的父组类型为 ``thread_block`` 或 ``thread_block_tile`` 。

实现可能会导致调用线程等待，直到父组的所有成员都调用了该操作后才恢复执行。功能仅限于原生硬件大小，即 1/2/4/8/16/32，并且 ``cg::size(parent)`` 必须大于 ``Size`` 参数。 ``tiled_partition`` 的模板版本还支持 64/128/256/512 大小，但在计算能力 7.5 或更低版本上需要一些额外步骤，详情请参阅 :ref:`class-thread-block-tile` 。

**代码生成要求：** 最低计算能力 5.0，大于 32 的大小需要 C++11

.. _labeled-partition:

5.6.3.3.2. ``labeled_partition``
""""""""""""""""""""""""""""""""

.. code-block:: c++

   template <typename Label>
   coalesced_group labeled_partition(const coalesced_group& g, Label label);

.. code-block:: c++

   template <unsigned int Size, typename Label>
   coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);

``labeled_partition`` 方法是一个集体操作，它将父组分区为一维的 subgroups，其中线程是合并的。实现将评估条件标签，并将具有相同标签值的线程分配到同一个组中。

``Label`` 可以是任何整数类型。

实现可能会导致调用线程等待，直到父组的所有成员都调用了该操作后才恢复执行。

**注意：** 此功能仍在评估中，未来可能会有所变化。

**代码生成要求：** 最低计算能力 7.0，C++11

.. _binary-partition:

5.6.3.3.3. ``binary_partition``
"""""""""""""""""""""""""""""""

.. code-block:: c++

   coalesced_group binary_partition(const coalesced_group& g, bool pred);

.. code-block:: c++

   template <unsigned int Size>
   coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);

``binary_partition()`` 方法是一个集体操作，它将父组分区为一维的 subgroups，其中线程是合并的。实现将评估谓词，并将具有相同值的线程分配到同一个组中。这是 ``labeled_partition()`` 的一种特殊形式，其中标签只能是 0 或 1。

实现可能会导致调用线程等待，直到父组的所有成员都调用了该操作后才恢复执行。

**示例：**

.. code-block:: c++

   /// This example divides a 32-sized tile into a group with odd
   /// numbers and a group with even numbers
   _global__ void oddEven(int *inputArr) {
       auto block = cg::this_thread_block();
       auto tile32 = cg::tiled_partition<32>(block);

       // inputArr contains random integers
       int elem = inputArr[block.thread_rank()];
       // after this, tile32 is split into 2 groups,
       // a subtile where elem&1 is true and one where its false
       auto subtile = cg::binary_partition(tile32, (elem & 1));
   }

.. _cooperative-groups-reduce-h:

5.6.3.4. cooperative_groups/reduce.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _reduce-operators:

5.6.3.4.1. ``Reduce`` 算子
""""""""""""""""""""""""""

以下是可以使用 ``reduce`` 进行的一些基本操作的函数对象原型。

.. code-block:: c++

   namespace cooperative_groups {
     template <typename Ty>
     struct cg::plus;

     template <typename Ty>
     struct cg::less;

     template <typename Ty>
     struct cg::greater;

     template <typename Ty>
     struct cg::bit_and;

     template <typename Ty>
     struct cg::bit_xor;

     template <typename Ty>
     struct cg::bit_or;
   }

Reduce 受限于编译时实现可用的信息。因此，为了利用计算能力 8.0 引入的内建函数， ``cg::`` 命名空间暴露了多个与硬件对应的函数对象。这些对象看起来与 C++ STL 中呈现的类似，但 ``less/greater`` 除外。与 STL 不同的原因在于，这些函数对象旨在实际反映硬件内建函数的操作。

**功能描述：**

- ``cg::plus:`` 接受两个值并使用 operator+ 返回两者之和。
- ``cg::less:`` 接受两个值并使用 operator< 返回较小值。不同之处在于**返回较小值**而非布尔值。
- ``cg::greater:`` 接受两个值并使用 operator< 返回较大值。不同之处在于**返回较大值**而非布尔值。
- ``cg::bit_and:`` 接受两个值并返回 operator& 的结果。
- ``cg::bit_xor:`` 接受两个值并返回 operator^ 的结果。
- ``cg::bit_or:`` 接受两个值并返回 operator| 的结果。

**示例：**

.. code-block:: c++

   {
       // cg::plus<int> is specialized within cg::reduce and calls __reduce_add_sync(...) on CC 8.0+
       cg::reduce(tile, (int)val, cg::plus<int>());

       // cg::plus<float> fails to match with an accelerator and instead performs a standard shuffle based reduction
       cg::reduce(tile, (float)val, cg::plus<float>());

       // While individual components of a vector are supported, reduce will not use hardware intrinsics for the following
       // It will also be necessary to define a corresponding operator for vector and any custom types that may be used
       int4 vec = {...};
       cg::reduce(tile, vec, cg::plus<int4>())

       // Finally lambdas and other function objects cannot be inspected for dispatch
       // and will instead perform shuffle based reductions using the provided function object.
       cg::reduce(tile, (int)val, [](int l, int r) -> int {return l + r;});
   }

.. _reduce:

5.6.3.4.2. ``reduce``
"""""""""""""""""""""

.. code-block:: c++

   template <typename TyGroup, typename TyArg, typename TyOp>
   auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));

``reduce`` 对传入组中每个线程提供的数据执行归约操作。它利用硬件加速（在计算能力 8.0 及更高设备上）进行算术加法、最小值或最大值操作以及逻辑 AND、OR 或 XOR 操作，同时在旧一代硬件上提供软件回退。只有 4B 类型可以通过硬件加速。

``group`` ：有效的组类型为 ``coalesced_group`` 和 ``thread_block_tile`` 。

``val`` ：满足以下要求的任何类型：

- 符合简单可复制条件，即 ``is_trivially_copyable<TyArg>::value == true``
- 对于 ``coalesced_group`` 和大小不超过 32 的 tile， ``sizeof(T) <= 32`` ；对于更大的 tile， ``sizeof(T) <= 8``
- 具有给定函数对象所需的适当算术或比较运算符。

**注意：** 组中的不同线程可以为此参数传递不同的值。

``op`` ：对于整数类型将提供硬件加速的有效函数对象为 ``plus(), less(), greater(), bit_and(), bit_xor(), bit_or()`` 。这些必须被构造，因此需要 TyVal 模板参数，即 ``plus<int>()`` 。Reduce 还支持 lambda 表达式和其他可以使用 ``operator()`` 调用的函数对象。

异步 reduce

.. code-block:: c++

   template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
   void reduce_update_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

   template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
   void reduce_store_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

   template <typename TyGroup, typename TyArg, typename TyOp>
   void reduce_store_async(const TyGroup& group, TyArg* ptr, TyArg&& val, TyOp&& op);

``*_async`` 变体的 API 异步计算结果，由参与线程之一存储到或更新指定目标，而不是由每个线程返回。要观察这些异步调用的效果，需要同步调用线程组或包含它们的更大组。

- 对于原子存储或更新变体， ``atomic`` 参数可以是 `CUDA C++ Standard Library <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html>`__ 中的 ``cuda::atomic`` 或 ``cuda::atomic_ref`` 。此 API 变体仅在 CUDA C++ Standard Library 支持这些类型的平台和设备上可用。归约的结果用于根据指定的 ``op`` 原子更新 atomic，例如在 ``cg::plus()`` 的情况下，结果被原子地加到 atomic 上。 ``atomic`` 持有的类型必须与 ``TyArg`` 的类型匹配。原子的作用域必须包含组中的所有线程，如果多个组同时使用同一个原子，则作用域必须包含所有使用它的组中的所有线程。原子更新使用 relaxed 内存序执行。
- 对于指针存储变体，归约的结果将被弱存储到 ``dst`` 指针中。

.. _cooperative-groups-scan-h:

5.6.3.5. cooperative_groups/scan.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _inclusive-scan-and-exclusive-scan:

5.6.3.5.1. ``inclusive_scan`` 和 ``exclusive_scan``
"""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: c++

   template <typename TyGroup, typename TyVal, typename TyFn>
   auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

   template <typename TyGroup, typename TyVal>
   TyVal inclusive_scan(const TyGroup& group, TyVal&& val);

   template <typename TyGroup, typename TyVal, typename TyFn>
   auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

   template <typename TyGroup, typename TyVal>
   TyVal exclusive_scan(const TyGroup& group, TyVal&& val);

``inclusive_scan`` 和 ``exclusive_scan`` 对传入组中每个线程提供的数据执行扫描操作。对于 ``exclusive_scan`` ，每个线程的结果是对具有比该线程更低 ``thread_rank`` 的线程数据的归约。 ``inclusive_scan`` 的结果还在归约中包含调用线程的数据。

``group`` ：有效的组类型为 ``coalesced_group`` 和 ``thread_block_tile`` 。

``val`` ：满足以下要求的任何类型：

- 符合简单可复制条件，即 ``is_trivially_copyable<TyArg>::value == true``
- 对于 ``coalesced_group`` 和大小不超过 32 的 tile， ``sizeof(T) <= 32`` ；对于更大的 tile， ``sizeof(T) <= 8``
- 具有给定函数对象所需的适当算术或比较运算符。

**注意：** 组中的不同线程可以为此参数传递不同的值。

``op`` ：为方便起见定义的函数对象为 ``plus(), less(), greater(), bit_and(), bit_xor(), bit_or()`` ，在 :ref:`cooperative-groups-reduce-h` 中描述。这些必须被构造，因此需要 TyVal 模板参数，即 ``plus<int>()`` 。 ``inclusive_scan`` 和 ``exclusive_scan`` 还支持 lambda 表达式和其他可以使用 ``operator()`` 调用的函数对象。不带此参数的重载使用 ``cg::plus<TyVal>()`` 。

**Scan update**

.. code-block:: c++

   template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
   auto inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

   template <typename TyGroup, typename TyAtomic, typename TyVal>
   TyVal inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

   template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
   auto exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

   template <typename TyGroup, typename TyAtomic, typename TyVal>
   TyVal exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

``*_scan_update`` 集体操作接受一个额外的参数 ``atomic`` ，它可以是 `CUDA C++ Standard Library <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html>`__ 中的 ``cuda::atomic`` 或 ``cuda::atomic_ref`` 。这些 API 变体仅在 CUDA C++ Standard Library 支持这些类型的平台和设备上可用。这些变体将根据 ``op`` 对 ``atomic`` 执行更新，更新值为组中所有线程输入值的总和。 ``atomic`` 的前一个值将与每个线程的扫描结果合并并返回。 ``atomic`` 持有的类型必须与 ``TyVal`` 的类型匹配。原子的作用域必须包含组中的所有线程，如果多个组同时使用同一个原子，则作用域必须包含所有使用它的组中的所有线程。原子更新使用 relaxed 内存序执行。

以下伪代码说明了 scan update 变体的工作方式：

.. code-block:: c++

   /*
    inclusive_scan_update behaves as the following block,
    except both reduce and inclusive_scan is calculated simultaneously.
   auto total = reduce(group, val, op);
   TyVal old;
   if (group.thread_rank() == selected_thread) {
       atomically {
           old = atomic.load();
           atomic.store(op(old, total));
       }
   }
   old = group.shfl(old, selected_thread);
   return op(inclusive_scan(group, val, op), old);
   */

需要包含 ``cooperative_groups/scan.h`` 头文件。

**使用 exclusive_scan 进行流压缩的示例：**

.. code-block:: c++

   #include <cooperative_groups.h>
   #include <cooperative_groups/scan.h>
   namespace cg = cooperative_groups;

   // put data from input into output only if it passes test_fn predicate
   template<typename Group, typename Data, typename TyFn>
   __device__ int stream_compaction(Group &g, Data *input, int count, TyFn&& test_fn, Data *output) {
       int per_thread = count / g.num_threads();
       int thread_start = min(g.thread_rank() * per_thread, count);
       int my_count = min(per_thread, count - thread_start);

       // get all passing items from my part of the input
       //  into a contagious part of the array and count them.
       int i = thread_start;
       while (i < my_count + thread_start) {
           if (test_fn(input[i])) {
               i++;
           }
           else {
               my_count--;
               input[i] = input[my_count + thread_start];
           }
       }

       // scan over counts from each thread to calculate my starting
       //  index in the output
       int my_idx = cg::exclusive_scan(g, my_count);

       for (i = 0; i < my_count; ++i) {
           output[my_idx + i] = input[thread_start + i];
       }
       // return the total number of items in the output
       return g.shfl(my_idx + my_count, g.num_threads() - 1);
   }

**使用 exclusive_scan_update 进行动态缓冲区空间分配的示例：**

.. code-block:: c++

   #include <cooperative_groups.h>
   #include <cooperative_groups/scan.h>
   namespace cg = cooperative_groups;

   // Buffer partitioning is static to make the example easier to follow,
   // but any arbitrary dynamic allocation scheme can be implemented by replacing this function.
   __device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
       return tile.thread_rank() % 2 + 1;
   }

   __device__ int my_thread_data(int i) {
       return i;
   }

   __global__ void kernel() {
       __shared__ extern int buffer[];
       __shared__ cuda::atomic<int, cuda::thread_scope_block> buffer_used;

       auto block = cg::this_thread_block();
       auto tile = cg::tiled_partition<32>(block);
       buffer_used = 0;
       block.sync();

       // each thread calculates buffer size it needs
       int buf_needed = calculate_buffer_space_needed(tile);

       // scan over the needs of each thread, result for each thread is an offset
       // of that thread's part of the buffer. buffer_used is atomically updated with
       // the sum of all thread's inputs, to correctly offset other tile's allocations
       int buf_offset =
           cg::exclusive_scan_update(tile, buffer_used, buf_needed);

       // each thread fills its own part of the buffer with thread specific data
       for (int i = 0 ; i < buf_needed ; ++i) {
           buffer[buf_offset + i] = my_thread_data(i);
       }

       block.sync();
       // buffer_used now holds total amount of memory allocated
       // buffer is {0, 0, 1, 0, 0, 1 ...};
   }

.. _cooperative-groups-sync-h:

5.6.3.6. cooperative_groups/sync.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _barrier-arrive-and-barrier-wait:

5.6.3.6.1. ``barrier_arrive`` 和 ``barrier_wait``
"""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: c++

   T::arrival_token T::barrier_arrive();
   void T::barrier_wait(T::arrival_token&&);

``barrier_arrive`` 和 ``barrier_wait`` 成员函数提供类似于 ``cuda::barrier`` :ref:`asynchronous-barriers` 的同步 API。Cooperative Groups 自动初始化组屏障，但到达和等待操作有一个额外的限制，这源于这些操作的集体性质：组中的所有线程在每个阶段必须到达并等待屏障一次。当对组调用 ``barrier_arrive`` 时，在用 ``barrier_wait`` 调用观察到屏障阶段完成之前，使用该组调用任何集体操作或另一个屏障到达的结果是未定义的。阻塞在 ``barrier_wait`` 上的线程可能在其他线程调用 ``barrier_wait`` 之前就被释放同步，但仅在组中所有线程都调用了 ``barrier_arrive`` 之后。组类型 ``T`` 可以是任何 :ref:`tbl:cg-implicit-groups` 。这允许线程在到达之后和等待同步解决之前执行独立工作，从而隐藏部分同步延迟。 ``barrier_arrive`` 返回一个 ``arrival_token`` 对象，该对象必须传递给相应的 ``barrier_wait`` 。令牌以这种方式被消耗，不能用于另一个 ``barrier_wait`` 调用。

**使用 barrier_arrive 和 barrier_wait 同步跨集群共享内存初始化的示例：**

.. code-block:: c++

   #include <cooperative_groups.h>

   using namespace cooperative_groups;

   void __device__ init_shared_data(const thread_block& block, int *data);
   void __device__ local_processing(const thread_block& block);
   void __device__ process_shared_data(const thread_block& block, int *data);

   __global__ void cluster_kernel() {
       extern __shared__ int array[];
       auto cluster = this_cluster();
       auto block   = this_thread_block();

       // Use this thread block to initialize some shared state
       init_shared_data(block, &array[0]);

       auto token = cluster.barrier_arrive(); // Let other blocks know this block is running and data was initialized

       // Do some local processing to hide the synchronization latency
       local_processing(block);

       // Map data in shared memory from the next block in the cluster
       int *dsmem = cluster.map_shared_rank(&array[0], (cluster.block_rank() + 1) % cluster.num_blocks());

       // Make sure all other blocks in the cluster are running and initialized shared data before accessing dsmem
       cluster.barrier_wait(std::move(token));

       // Consume data in distributed shared memory
       process_shared_data(block, dsmem);
       cluster.sync();
   }

.. _sync:

5.6.3.6.2. ``sync``
"""""""""""""""""""

.. code-block:: c++

   static void T::sync();

   template <typename T>
   void sync(T& group);

``sync`` 同步组中命名的线程。组类型 ``T`` 可以是任何现有的组类型，因为它们都支持同步。它可以作为每种组类型的成员函数使用，也可以作为接受组参数的自由函数使用。

.. _grid-synchronization:

5.6.3.6.2.1. Grid 同步
""""""""""""""""""""""

在引入 Cooperative Groups 之前，CUDA 编程模型只允许在内核完成边界处进行线程块之间的同步。内核边界带有隐式的状态失效，并随之带来潜在的性能影响。

例如，在某些用例中，应用程序有大量小内核，每个内核代表处理流水线中的一个阶段。当前 CUDA 编程模型需要这些内核的存在，以确保在一个流水线阶段上运行的线程块已产生数据，然后操作下一个流水线阶段的线程块才准备消费它。在这种情况下，提供全局线程块间同步的能力将允许应用程序重构为持久线程块，这些线程块能够在给定阶段完成时在设备上进行同步。

要在内核内部跨 grid 同步，只需使用 ``grid.sync()`` 函数：

.. code-block:: c++

   grid_group grid = this_grid();
   grid.sync();

在启动内核时，需要使用 ``cudaLaunchCooperativeKernel`` CUDA 运行时启动 API 或 ``CUDA driver 等效函数`` 来代替 ``<<<...>>>`` 执行配置语法。

**示例：**

为了保证线程块在 GPU 上的共同驻留，需要仔细考虑启动的块数。例如，可以按照以下方式启动与 SM 数量相同的块：

.. code-block:: c++

   int dev = 0;
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   // initialize, then launch
   cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);

或者，可以通过使用占用率计算器计算每个 SM 可以同时容纳多少块来最大化暴露的并行度：

.. code-block:: c++

   /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
   int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
   int numThreads = 128;
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
   // launch
   void *kernelArgs[] = { /* add kernel args */ };
   dim3 dimBlock(numThreads, 1, 1);
   dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
   cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);

良好的做法是先通过查询设备属性 ``cudaDevAttrCooperativeLaunch`` 来确保设备支持协作启动：

.. code-block:: c++

   int dev = 0;
   int supportsCoopLaunch = 0;
   cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

如果设备 0 支持该属性，则会将 ``supportsCoopLaunch`` 设置为 1。仅支持计算能力 6.0 及更高的设备。此外，需要在以下任一环境上运行：

- 不使用 MPS 的 Linux 平台
- 使用 MPS 且计算能力 7.0 或更高的 Linux 平台
- 最新的 Windows 平台

.. _cuda-device-runtime:

5.6.4. CUDA 设备运行时
----------------------

CUDA 设备运行时是内核代码中可用的 API，它提供了与主机上 CUDA Runtime API 许多相同的功能。这些 API 最常用于 :ref:`cuda-dynamic-parallelism` 或 :ref:`cuda-graphs-device-graph-launch` 的上下文中。

.. _including-device-runtime-api-in-cuda-code:

5.6.4.1. 在 CUDA 代码中包含设备运行时 API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与主机端运行时 API 类似，CUDA 设备运行时 API 的原型在程序编译期间自动包含。无需显式包含 ``cuda_device_runtime_api.h`` 。

.. _memory-in-the-cuda-device-runtime:

5.6.4.2. CUDA 设备运行时中的内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _configuration-options:

5.6.4.2.1. 配置选项
"""""""""""""""""""

设备运行时系统软件的资源分配通过主机程序的 ``cudaDeviceSetLimit()`` API 控制。必须在启动任何内核之前设置限制，并且在 GPU  actively 运行程序时不得更改。

可以设置以下命名限制：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 限制
     - 行为
   * - ``cudaLimitDevRuntimePendingLaunchCount``
     - 控制为缓冲尚未开始执行的内核启动和事件而预留的内存量，这些启动和事件可能由于未解决的依赖关系或缺乏执行资源而尚未开始。当缓冲区已满时，在设备端内核启动期间尝试分配启动槽将失败并返回 ``cudaErrorLaunchOutOfResources`` ，而尝试分配事件槽将失败并返回 ``cudaErrorMemoryAllocation`` 。默认启动槽数量为 2048。应用程序可以通过设置 ``cudaLimitDevRuntimePendingLaunchCount`` 来增加启动和/或事件槽的数量。分配的事件槽数量是该限制值的两倍。
   * - ``cudaLimitStackSize``
     - 控制每个 GPU 线程的栈大小（以字节为单位）。CUDA 驱动程序会根据需要自动增加每次内核启动的每线程栈大小。此大小不会在每次启动后重置为原始值。要将每线程栈大小设置为不同的值，可以调用 ``cudaDeviceSetLimit()`` 来设置此限制。栈将立即调整大小，如有必要，设备将阻塞直到所有先前请求的任务完成。可以调用 ``cudaDeviceGetLimit()`` 来获取当前的每线程栈大小。

.. _allocation-and-lifetime:

5.6.4.2.2. 分配和生命周期
"""""""""""""""""""""""""

``cudaMalloc()`` 和 ``cudaFree()`` 在主机和设备环境之间具有不同的语义。从主机调用时， ``cudaMalloc()`` 从未使用的设备内存中分配一个新区域。从设备运行时调用时，这些函数映射到设备端的 ``malloc()`` 和 ``free()`` 。这意味着在设备环境中，总可分配内存受限于设备 ``malloc()`` 堆大小，该大小可能小于可用的未使用设备内存。此外，在主机程序上对在设备上通过 ``cudaMalloc()`` 分配的指针调用 ``cudaFree()`` 是错误的，反之亦然。

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * -
     - 主机上的 ``cudaMalloc()``
     - 设备上的 ``cudaMalloc()``
   * - 主机上的 ``cudaFree()``
     - 支持
     - 不支持
   * - 设备上的 ``cudaFree()``
     - 不支持
     - 支持
   * - 分配限制
     - 可用设备内存
     - ``cudaLimitMallocHeapSize``

.. _memory-declarations:

5.6.4.2.2.1. 内存声明
"""""""""""""""""""""

.. _device-and-constant-memory:

5.6.4.2.2.1.1. 设备和常量内存
"""""""""""""""""""""""""""""

在文件作用域中使用 ``__device__`` 或 ``__constant__`` 内存空间说明符声明的内存，在使用设备运行时行为相同。所有内核都可以读写设备变量，无论内核最初是由主机还是设备运行时启动的。同样，所有内核都将具有与在模块作用域中声明的 ``__constant__`` 相同的视图。

.. _textures-and-surfaces:

5.6.4.2.2.1.2. 纹理和表面
"""""""""""""""""""""""""

  设备运行时不允许从设备代码内部创建或销毁纹理或表面对象。从主机创建的纹理和表面对象可以在设备上自由使用和传递。无论在何处创建，动态创建的纹理对象始终有效，并且可以从父内核传递给子内核。

.. note::

   设备运行时不支持从设备启动的内核中的遗留模块作用域（即计算能力 2.0 或 Fermi 风格）纹理和表面。模块作用域（遗留）纹理可以从主机创建并在设备代码中使用，与任何内核一样，但只能由顶级内核（即从主机启动的内核）使用。

.. _shared-memory-variable-declarations:

5.6.4.2.2.1.3. 共享内存变量声明
"""""""""""""""""""""""""""""""

在 CUDA C++ 中，共享内存可以声明为静态大小的文件作用域或函数作用域变量，也可以声明为 ``extern`` 变量，其大小在内核调用者通过启动配置参数在运行时确定。两种类型的声明在设备运行时下都有效。

.. code-block:: c++

   __global__ void permute(int n, int *data) {
      extern __shared__ int smem[];
      if (n <= 1)
          return;

      smem[threadIdx.x] = data[threadIdx.x];
      __syncthreads();

      permute_data(smem, n);
      __syncthreads();

      // Write back to GMEM since we can't pass SMEM to children.
      data[threadIdx.x] = smem[threadIdx.x];
      __syncthreads();

      if (threadIdx.x == 0) {
          permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
          permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
      }
   }

   void host_launch(int *data) {
       permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
   }

.. _constant-memory:

5.6.4.2.2.1.4. 常量内存
"""""""""""""""""""""""

常量不能从设备修改。它们只能从主机修改，但在主机修改常量的同时，如果存在并发 grid 在其生命周期的任何时刻访问该常量，则行为是未定义的。

.. _symbol-addresses:

5.6.4.2.2.1.5. 符号地址
"""""""""""""""""""""""

设备端符号（即标记为 ``__device__`` 的符号）可以通过 ``&`` 运算符从内核中引用，因为所有全局作用域设备变量都在内核的可见地址空间中。这也适用于 ``__constant__`` 符号，尽管在这种情况下指针将引用只读数据。

由于设备端符号可以直接引用，因此引用符号的 CUDA 运行时 API（例如 ``cudaMemcpyToSymbol()`` 或 ``cudaGetSymbolAddress()`` ）是不必要的，设备运行时不支持这些 API。这意味着常量数据不能在运行中的内核内部更改，即使在子内核启动之前也不行，因为对 ``__constant__`` 空间的引用是只读的。

.. _sm-id-and-warp-id:

5.6.4.3. SM Id 和 Warp Id
^^^^^^^^^^^^^^^^^^^^^^^^^

请注意，在 PTX 中 ``%smid`` 和 ``%warpid`` 被定义为 volatile 值。设备运行时可能会将线程块重新调度到不同的 SM 上，以更有效地管理资源。因此，依赖 ``%smid`` 或 ``%warpid`` 在线程或线程块的整个生命周期内保持不变是不安全的。

.. _launch-setup-apis:

5.6.4.4. 启动设置 API
^^^^^^^^^^^^^^^^^^^^^

:ref:`dynamic-parallelism-device-runtime-kernel-launch` 描述了使用与主机 CUDA Runtime API 相同的三重尖括号启动符号从设备代码启动内核的语法。

内核启动是通过设备运行时库公开的系统级机制。它也可以通过 ``cudaGetParameterBuffer()`` 和 ``cudaLaunchDevice()`` API 直接从 PTX 中获得。允许 CUDA 应用程序自行调用这些 API，要求与 PTX 相同。在这两种情况下，用户负责按照规范以正确的格式正确填充所有必要的数据结构。这些数据结构的向后兼容性得到保证。

与主机端启动一样，设备端操作符 ``<<<>>>`` 映射到底层内核启动 API。这允许针对 PTX 的用户执行启动。NVCC 编译器前端将 ``<<<>>>`` 转换为这些调用。

.. list-table:: 新的仅设备启动实现函数
   :header-rows: 1
   :widths: 30 70

   * - 运行时 API 启动函数
     - 与主机运行时行为的差异描述（如无描述则行为相同）
   * - ``cudaGetParameterBuffer``
     - 从 ``<<<>>>`` 自动生成。注意与主机等效函数的 API 不同。
   * - ``cudaLaunchDevice``
     - 从 ``<<<>>>`` 自动生成。注意与主机等效函数的 API 不同。

这些启动函数的 API 与 CUDA Runtime API 的 API 不同，定义如下：

.. code-block:: c++

   extern   device   cudaError_t cudaGetParameterBuffer(void **params);
   extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                           void *params, dim3 gridDim,
                                           dim3 blockDim,
                                           unsigned int sharedMemSize = 0,
                                           cudaStream_t stream = 0);

.. _device-management:

5.6.4.5. 设备管理
^^^^^^^^^^^^^^^^^

设备运行时不支持多 GPU；设备运行时只能在其当前执行的设备上进行操作。但是，允许查询系统中任何具有 CUDA 能力的设备的属性。

.. _api-reference:

5.6.4.6. API 参考
^^^^^^^^^^^^^^^^^

设备运行时支持的 CUDA Runtime API 部分在此详述。主机和设备运行时 API 具有相同的语法；除另有说明外，语义相同。下表提供了相对于主机可用版本的 API 概述。

.. list-table:: 支持的 API 函数
   :header-rows: 1
   :widths: 40 60

   * - 运行时 API 函数
     - 详情
   * - ``cudaDeviceGetCacheConfig``
     -
   * - ``cudaDeviceGetLimit``
     -
   * - ``cudaGetLastError``
     - 最后错误是每线程状态，不是每块状态
   * - ``cudaPeekAtLastError``
     -
   * - ``cudaGetErrorString``
     -
   * - ``cudaGetDeviceCount``
     -
   * - ``cudaDeviceGetAttribute``
     - 将返回任何设备的属性
   * - ``cudaGetDevice``
     - 始终返回从主机看到的当前设备 ID
   * - ``cudaStreamCreateWithFlags``
     - 必须传递 ``cudaStreamNonBlocking`` 标志
   * - ``cudaStreamDestroy``
     -
   * - ``cudaStreamWaitEvent``
     -
   * - ``cudaEventCreateWithFlags``
     - 必须传递 ``cudaEventDisableTiming`` 标志
   * - ``cudaEventRecord``
     -
   * - ``cudaEventDestroy``
     -
   * - ``cudaFuncGetAttributes``
     -
   * - ``cudaMemcpyAsync``
     - 关于所有 ``memcpy/set`` 函数的说明：仅支持异步 ``memcpy/set`` 函数；仅允许设备到设备的 ``memcpy`` ；不能传入本地或共享内存指针
   * - ``cudaMemcpy2DAsync``
     - 同上
   * - ``cudaMemcpy3DAsync``
     - 同上
   * - ``cudaMemsetAsync``
     - 同上
   * - ``cudaMemset2DAsync``
     -
   * - ``cudaMemset3DAsync``
     -
   * - ``cudaRuntimeGetVersion``
     -
   * - ``cudaMalloc``
     - 不能在设备上对主机创建的指针调用 ``cudaFree`` ，反之亦然
   * - ``cudaFree``
     - 同上
   * - ``cudaOccupancyMaxActiveBlocksPerMultiprocessor``
     -
   * - ``cudaOccupancyMaxPotentialBlockSize``
     -
   * - ``cudaOccupancyMaxPotentialBlockSizeVariableSMem``
     -

.. _api-errors-and-launch-failures:

5.6.4.7. API 错误和启动失败
^^^^^^^^^^^^^^^^^^^^^^^^^^^

与 CUDA 运行时一样，任何函数都可能返回错误代码。返回的最后一个错误代码被记录，可以通过 ``cudaGetLastError()`` 调用检索。错误按线程记录，因此每个线程都可以识别它生成的最近错误。错误代码的类型为 ``cudaError_t`` 。

与主机端启动类似，设备端启动可能因多种原因失败（无效参数等）。用户必须调用 ``cudaGetLastError()`` 来确定启动是否生成了错误，但启动后没有错误并不意味着子内核成功完成。

对于设备端异常，例如访问无效地址，子 grid 中的错误将返回给主机。

.. _device-runtime-streams:

5.6.4.8. 设备运行时流
^^^^^^^^^^^^^^^^^^^^^

CUDA 设备运行时公开了特殊的命名流，为从设备启动的内核和图提供特定行为。与设备图启动相关的命名流在 :ref:`cuda-graphs-device-graph-launch` 中有文档说明。另外两个可用于 CUDA 设备运行时中内核和 memcpy 操作的命名流是 ``cudaStreamTailLaunch`` 和 ``cudaStreamTailLaunch`` 。这些命名流的具体行为在本节中有文档说明。

命名和未命名（NULL）流都可以从设备运行时获得。流句柄不能传递给父 grid 或子 grid。换句话说，流应被视为其创建的 grid 的私有对象。

主机端 NULL 流的跨流屏障语义在设备上不受支持（详见下文）。为了保持与主机运行时的语义兼容性，所有设备流必须使用 ``cudaStreamCreateWithFlags()`` API 创建，并传递 ``cudaStreamNonBlocking`` 标志。 ``cudaStreamCreate()`` API 在 CUDA 设备运行时中不可用。

由于 ``cudaStreamSynchronize()`` 和 ``cudaStreamQuery()`` 不受设备运行时支持，当应用程序需要知道流启动的子内核已完成时，应使用启动到 ``cudaStreamTailLaunch`` 流中的内核。

.. _the-implicit-null-stream:

5.6.4.8.1. 隐式（NULL）流
"""""""""""""""""""""""""

在主机程序中，未命名（NULL）流与其他流之间具有额外的屏障同步语义（详情请参阅 :ref:`async-execution-blocking-non-blocking-default-stream` ）。设备运行时提供一个在线程块中所有线程之间共享的单一隐式未命名流，但由于所有命名流必须使用 ``cudaStreamNonBlocking`` 标志创建，启动到 NULL 流中的工作不会在任何其他流（包括其他线程块的 NULL 流）中的待处理工作上插入隐式依赖。

.. _fire-and-forget-stream:

5.6.4.8.2. Fire-and-Forget 流
"""""""""""""""""""""""""""""

Fire-and-forget 命名流（ ``cudaStreamFireAndForget`` ）允许用户以更少的样板代码和无需流跟踪开销的方式启动 fire-and-forget 工作。它在功能上等同于但比每次启动创建新流并启动到该流更快。

Fire-and-forget 启动会立即被调度启动，而不依赖于先前启动 grid 的完成。没有其他 grid 启动可以依赖于 fire-and-forget 启动的完成，除非通过父 grid 末尾的隐式同步。因此，尾部启动或父 grid 流中的下一个 grid 不会在父 grid 的 fire-and-forget 工作完成之前启动。

.. code-block:: c++

   // In this example, C2's launch will not wait for C1's completion
   __global__ void P( ... ) {
      C1<<< ... , cudaStreamFireAndForget >>>( ... );
      C2<<< ... , cudaStreamFireAndForget >>>( ... );
   }

Fire-and-forget 流不能用于记录或等待事件。尝试这样做会导致 ``cudaErrorInvalidValue`` 。在定义了 ``CUDA_FORCE_CDP1_IF_SUPPORTED`` 编译时不支持 fire-and-forget 流。Fire-and-forget 流的使用需要以 64 位模式编译。

.. _tail-launch-stream:

5.6.4.8.3. Tail Launch 流
"""""""""""""""""""""""""

Tail launch 命名流（ ``cudaStreamTailLaunch`` ）允许 grid 在其完成后调度新 grid 启动。在大多数情况下，应该可以使用 tail launch 实现与 ``cudaDeviceSynchronize()`` 相同的功能。

每个 grid 都有自己的 tail launch 流。grid 启动的所有非 tail launch 工作在 tail 流启动之前被隐式同步。即，父 grid 的 tail launch 在父 grid 和父 grid 启动到普通流或每线程或 fire-and-forget 流中的所有工作完成之前不会启动。如果两个 grid 被启动到同一个 grid 的 tail launch 流中，后一个 grid 在前一个 grid 及其所有后代工作完成之前不会启动。

.. code-block:: c++

   // In this example, C2 will only launch after C1 completes.
   __global__ void P( ... ) {
      C1<<< ... , cudaStreamTailLaunch >>>( ... );
      C2<<< ... , cudaStreamTailLaunch >>>( ... );
   }

启动到 tail launch 流中的 grid 在父 grid 的所有工作完成之前不会启动，包括父 grid 在所有非 tail launch 流中启动的所有其他 grid（及其后代），包括在 tail launch 之后执行或启动的工作。

.. code-block:: c++

   // In this example, C will only launch after all X, F and P complete.
   __global__ void P( ... ) {
      C<<< ... , cudaStreamTailLaunch >>>( ... );
      X<<< ... , cudaStreamPerThread >>>( ... );
      F<<< ... , cudaStreamFireAndForget >>>( ... )
   }

父 grid 流中的下一个 grid 不会在父 grid 的 tail launch 工作完成之前启动。换句话说，tail launch 流的行为就像它被插入到其父 grid 和父 grid 流中的下一个 grid 之间。

.. code-block:: c++

   // In this example, P2 will only launch after C completes.
   __global__ void P1( ... ) {
      C<<< ... , cudaStreamTailLaunch >>>( ... );
   }

   __global__ void P2( ... ) {
   }

   int main ( ... ) {
      ...
      P1<<< ... >>>( ... );
      P2<<< ... >>>( ... );
      ...
   }

每个 grid 只能获得一个 tail launch 流。要 tail launch 并发 grid，可以像下面的示例这样做。

.. code-block:: c++

   // In this example,  C1 and C2 will launch concurrently after P's completion
   __global__ void T( ... ) {
      C1<<< ... , cudaStreamFireAndForget >>>( ... );
      C2<<< ... , cudaStreamFireAndForget >>>( ... );
   }

   __global__ void P( ... ) {
      ...
      T<<< ... , cudaStreamTailLaunch >>>( ... );
   }

Tail launch 流不能用于记录或等待事件。尝试这样做会导致 ``cudaErrorInvalidValue`` 。在定义了 ``CUDA_FORCE_CDP1_IF_SUPPORTED`` 编译时不支持 tail launch 流。Tail launch 流的使用需要以 64 位模式编译。

.. _ecc-errors:

5.6.4.9. ECC 错误
^^^^^^^^^^^^^^^^^

CUDA 内核中的代码无法获得 ECC 错误的通知。ECC 错误在整个启动树完成后在主机端报告。在嵌套程序执行期间出现的任何 ECC 错误将生成异常或继续执行（取决于错误和配置）。
