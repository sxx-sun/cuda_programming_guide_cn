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

执行 **``shape`` 字节**的复制。

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

执行 **``min(dstLayout, srcLayout)`` 元素**的复制。如果布局类型为 ``cuda::aligned_size_t<N>`` ，两者必须指定相同的对齐。

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

.. note::

   有关设备可调用 API 和内建函数的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html>`_。