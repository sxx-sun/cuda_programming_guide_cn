.. _async-copies-details:

4.12. 异步数据拷贝
===================

基于 :ref:`asynchronous-data-copies` ，本节为 GPU 内存层次结构内的异步数据搬运提供详细指导和示例。
它涵盖了用于元素级拷贝的 LDGSTS、用于批量（一维和多维）传输的张量内存加速器 (TMA)、用于寄存器到分布式共享内存拷贝的 STAS，
并展示了这些机制如何与 :ref:`async-barriers-details` 和 :ref:`pipelines-details` 集成。

.. _using-ldgsts:

4.12.1. 使用 LDGSTS
-------------------

许多 CUDA 应用程序需要在 Global 内存和共享内存之间频繁移动数据。通常，这涉及复制较小的数据元素或执行不规则的内存访问模式。
`LDGSTS <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy>`_ （CC 8.0+）
的主要目标是为较小的元素级数据传输提供从 Global 内存到共享内存的有效异步数据传输机制，同时通过重叠执行更好地利用计算资源。

**维度**。 LDGSTS 支持复制 4、8 或 16 字节。复制 4 或 8 字节始终以所谓的 L1 ACCESS 模式进行，此时数据也缓存在 L1 中，而复制 16 字节启用 L1 BYPASS 模式，此时 L1 不会被污染。

**源和目标**。 LDGSTS 异步拷贝操作支持的唯一方向是从 Global 内存到共享内存。指针需要根据复制的数据大小对齐到 4、8 或 16 字节。
当共享内存和 Global 内存的对齐都是 128 字节时，可获得最佳性能。

**异步性**。使用 LDGSTS 的数据传输是 :ref:`异步的 <asynchronous-execution-features>` ，并建模为 :ref:`异步线程操作 <async-thread-and-async-proxy>` 。
这允许发起线程继续计算，而硬件异步复制数据。*数据传输是否实际异步执行取决于硬件实现，未来可能会发生变化*。

LDGSTS 必须在操作完成时提供信号。LDGSTS 可以使用 :ref:`共享内存屏障 <asynchronous-barriers>` 或 :ref:`管道 <pipelines>` 作为提供完成信号的机制。
默认情况下，每个线程只等待自己的 LDGSTS 拷贝。因此，如果您使用 LDGSTS 预取一些将与其他线程共享的数据，则在同步 LDGSTS 完成机制后需要 ``__syncthreads()`` 。

.. list-table:: 使用 LDGSTS 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持。
   :widths: 20 20 40 20
   :header-rows: 2

   * - 方向
     -
     - 异步拷贝 (LDGSTS)
     -
   * - 源
     - 目标
     - 完成机制
     - API
   * - Global
     - Global
     -
     -
   * - shared::cta
     - Global
     -
     -
   * - Global
     - shared::cta
     - 共享内存屏障、管道
     - | ``cuda::memcpy_async``
       | ``cooperative_groups::memcpy_async``
       | ``__pipeline_memcpy_async``
   * - Global
     - Cluster Shared
     -
     -
   * - shared::cluster
     - shared::cta
     -
     -
   * - shared::cta
     - shared::cta
     -
     -

在接下来的章节中，我们将通过示例演示如何使用 LDGSTS，并解释不同 API 之间的差异。

.. _async-copies-batching-loads:

4.12.1.1. 条件代码中的批量加载
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在这个模板示例中，线程块的第一个 warp 负责集体加载中心和左右 halo 所需的所有数据。
使用同步拷贝时，由于代码的条件性质，编译器可能会选择生成一系列从 Global 加载 (LDG) 到共享存储 (STS) 的指令，而不是 3 个 LDG 后跟 3 个 STS，这将是加载数据以隐藏 Global 内存延迟的最佳方式。

.. code-block:: cuda

   __global__ void stencil_kernel(const float *left, const float *center, const float *right)
   {
       // Left halo (8 elements) - center (32 elements) - right halo (8 elements)
       __shared__ float buffer[8 + 32 + 8];
       const int tid = threadIdx.x;

       if (tid < 8) {
           buffer[tid] = left[tid]; // Left halo
       } else if (tid >= 32 - 8) {
           buffer[tid + 16] = right[tid]; // Right halo
       }
       if (tid < 32) {
         buffer[tid + 8] = center[tid]; // Center
       }
       __syncthreads();

       // Compute stencil
   }

为了确保以最佳方式加载数据，我们可以用异步内存拷贝替换同步内存拷贝，直接从 Global 内存加载数据到共享内存。
这不仅通过直接将数据复制到共享内存来减少寄存器使用，还确保所有来自 Global 内存的加载都在进行中。


.. tab-set::

   .. tab-item:: cuda::memcpy_async

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cuda/barrier>

        __global__ void stencil_kernel(const float *left, const float *center, const float *right)
        {
            auto block = cooperative_groups::this_thread_block();
            auto thread = cooperative_groups::this_thread();
            using barrier_t = cuda::barrier<cuda::thread_scope_block>;
            const int tid = block.thread_rank();

            __shared__ barrier_t barrier;
            __shared__ float buffer[8 + 32 + 8];

            // Initialize synchronization object.
            if (block.thread_rank() == 0) {
                init(&barrier, block.size());
            }
            __syncthreads();

            // Version 1: Issue the copies in individual threads.
            if (tid < 8) {
                cuda::memcpy_async(buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Left halo
                // or cuda::memcpy_async(thread, buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
            } else if (tid >= 32 - 8) {
                cuda::memcpy_async(buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Right halo
                // or cuda::memcpy_async(thread, buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
            }
            if (tid < 32) {
                cuda::memcpy_async(buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Center
                // or cuda::memcpy_async(thread, buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
            }

            // Version 2: Cooperatively issue the copies across all threads.
            cuda::memcpy_async(block, buffer, left, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // Left halo
            cuda::memcpy_async(block, buffer + 8, center, cuda::aligned_size_t<4>(32 * sizeof(float)), barrier); // Center
            cuda::memcpy_async(block, buffer + 40, right, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // Right halo

            // Wait for all copies to complete.
            barrier.arrive_and_wait();
            __syncthreads();

            // Compute stencil
        }

   .. tab-item:: cooperative_groups::memcpy_async

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cooperative_groups/memcpy_async.h>

        namespace cg = cooperative_groups;

        __global__ void
        stencil_kernel (const float *left, const float *center, const float *right)
        {
          cg::thread_block block = cg::this_thread_block ();
          // Left halo (8 elements) - center (32 elements) - right halo (8 elements).
          __shared__ float buffer[8 + 32 + 8];

          // Cooperatively issue the copies across all threads.
          cg::memcpy_async (block, buffer, left, 8 * sizeof (float)); // Left halo
          cg::memcpy_async (block, buffer + 8, center, 32 * sizeof (float)); // Center
          cg::memcpy_async (block, buffer + 40, right, 8 * sizeof (float)); // Right halo
          cg::wait (block);                      // Waits for all copies to complete.
          __syncthreads ();

          // Compute stencil.
        }

   .. tab-item:: CUDA C primitives

      .. code-block:: cuda

        #include <cuda_pipeline.h>

        __global__ void stencil_kernel(const float *left, const float *center, const float *right)
        {
            // Left halo (8 elements) - center (32 elements) - right halo (8 elements).
            __shared__ float buffer[8 + 32 + 8];
            const int tid = threadIdx.x;

            if (tid < 8) {
                __pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float)); // Left halo
            } else if (tid >= 32 - 8) {
                __pipeline_memcpy_async(buffer + tid + 16, right + tid, sizeof(float)); // Right halo
            }
            if (tid < 32) {
                __pipeline_memcpy_async(buffer + tid + 8, center + tid, sizeof(float)); // Center
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();

            // Compute stencil.
        }

``cuda::memcpy_async`` 用于 ``cuda::barrier`` 的重载允许使用 :ref:`异步屏障 <asynchronous-barriers>` 同步异步数据传输。
此重载执行拷贝操作，就像由绑定到屏障的另一个线程执行一样，通过增加当前相位的预期计数，并在拷贝操作完成时减少它，
使得 ``barrier`` 的相位只有在屏障参与的所有线程都已到达且所有绑定到屏障当前相位的 ``memcpy_async`` 都完成后才会推进。
我们使用块级 ``barrier`` ，块中的所有线程都参与，并使用 ``arrive_and_wait`` 合并屏障的到达和等待，因为在相位之间我们不执行任何工作。

请注意，我们可以使用线程级拷贝（版本 1）或集体拷贝（版本 2）来达到相同的结果。
在版本 2 中，API 将自动处理底层拷贝的完成方式。在这两个版本中，我们使用 ``cuda::aligned_size_t<4>()`` 告知编译器数据按 4 字节对齐且拷贝的数据大小是 4 的倍数，以启用 LDGSTS。
请注意，为了与 ``cuda::barrier`` 互操作，这里使用来自 ``cuda/barrier`` 头文件的 ``cuda::memcpy_async`` 。

:ref:`cooperative_groups::memcpy_async <memcpy-async>` 实现在块的所有线程中集体协调内存传输，但使用 ``cg::wait(block)`` 而不是显式屏障操作来同步完成。

基于低级原始 API 的实现使用 ``__pipeline_memcpy_async()`` 启动元素级内存传输， ``__pipeline_commit()`` 提交一批拷贝， ``__pipeline_wait_prior(0)`` 等待管道中的所有操作完成。
与高级 API 相比，这提供了最直接的控制，但代码更冗长。它还确保底层将使用 LDGSTS，而高级 API 不保证这一点。

.. note::

   ``cooperative_groups::memcpy_async`` API 在此示例中效率较低，因为它会在启动时自动立即提交每个拷贝操作，从而阻止了其他 API 能够实现的在单个提交操作之前批处理多个拷贝的优化。

.. _async-copies-prefetching:

4.12.1.2. 预取数据
^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何使用异步数据拷贝从 Global 内存预取数据到共享内存。在迭代拷贝和计算模式中，这允许用当前迭代的计算隐藏未来迭代的数据传输延迟，可能增加飞行中的字节数。

.. tab-set::

   .. tab-item:: cuda::memcpy_async

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cuda/pipeline>

        template <size_t num_stages = 2 /* Pipeline with num_stages stages */>
        __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
            auto grid = cooperative_groups::this_grid();
            auto block = cooperative_groups::this_thread_block();
            auto thread = cooperative_groups::this_thread();
            assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size

            extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes
            size_t shared_offset[num_stages];
            for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

            cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

            auto block_batch = [&](size_t batch) -> int {
                return block.group_index().x * block.size() + grid.size() * batch;
            };

            // Fill the pipeline with the first ``num_stages`` batches.
            for (int s = 0; s < num_stages; ++s) {
                pipeline.producer_acquire();
                cuda::memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s) + tid,
                                    cuda::aligned_size_t<4>(sizeof(int)), pipeline);
                pipeline.producer_commit();
            }

            int stage = 0;

            // compute_batch: next batch to process
            // fetch_batch:   next batch to fetch from global memory
            for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size;
                  ++compute_batch, ++fetch_batch) {
                // Wait for the first requested stage to complete.
                constexpr size_t pending_batches = num_stages - 1;
                cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
                __syncthreads(); // Not required if each thread works on the data it copied.

                // Compute on the current batch
                compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);

                // Release the current stage.
                pipeline.consumer_release();
                __syncthreads(); // Not required if each thread works on the data it copied.

                // Load future stage ``num_stages`` ahead of current compute batch.
                pipeline.producer_acquire();
                if (fetch_batch < batch_size) {
                    cuda::memcpy_async(shared + shared_offset[stage] + tid,
                                        global_in + block_batch(fetch_batch) + tid,
                                        cuda::aligned_size_t<4>(sizeof(int)), pipeline);
                }
                pipeline.producer_commit();
                stage = (stage + 1) % num_stages;
            }
        }

   .. tab-item:: cooperative_groups::memcpy_async

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cooperative_groups/memcpy_async.h>

        namespace cg = cooperative_groups;

        template <size_t num_stages = 2 /* Pipeline with num_stages stages */>
        __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
            auto grid = cooperative_groups::this_grid();
            auto block = cooperative_groups::this_thread_block();
            assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size

            extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes
            size_t shared_offset[num_stages];
            for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

            auto block_batch = [&](size_t batch) -> int {
                return block.group_index().x * block.size() + grid.size() * batch;
            };

            // Fill the pipeline with the first ``num_stages`` batches.
            for (int s = 0; s < num_stages; ++s) {
                size_t block_batch_idx = block_batch(s);
                cg::memcpy_async(block, shared + shared_offset[s], global_in + block_batch_idx,
                                  cuda::aligned_size_t<4>(sizeof(int)));
            }

            int stage = 0;

            // compute_batch: next batch to process
            // fetch_batch:   next batch to fetch from global memory
            for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size;
                  ++compute_batch, ++fetch_batch) {
                // Wait for the first requested stage to complete.
                size_t pending_batches = (fetch_batch < batch_size - num_stages) ? num_stages - 1 : batch_size - fetch_batch - 1;
                cg::wait_prior(pending_batches);
                __syncthreads(); // Not required if each thread works on the data it copied.

                // Compute on the current batch.
                compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);

                __syncthreads(); // Not required if each thread works on the data it copied.

                // Load future stage ``num_stages`` ahead of current compute batch.
                size_t fetch_batch_idx = block_batch(fetch_batch);
                if (fetch_batch < batch_size) {
                    cg::memcpy_async(block, shared + shared_offset[stage], global_in + block_batch(fetch_batch),
                                      cuda::aligned_size_t<4>(sizeof(int)) * block.size());
                }
                stage = (stage + 1) % num_stages;
            }
        }

   .. tab-item:: CUDA C primitives

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cuda_awbarrier_primitives.h>

        template <size_t num_stages = 2 /* Pipeline with num_stages stages */>
        __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
            auto grid = cooperative_groups::this_grid();
            auto block = cooperative_groups::this_thread_block();
            assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size

            extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes
            size_t shared_offset[num_stages];
            for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

            auto block_batch = [&](size_t batch) -> int {
                return block.group_index().x * block.size() + grid.size() * batch;
            };

            // Fill the pipeline with the first ``num_stages`` batches.
            for (int s = 0; s < num_stages; ++s) {
                __pipeline_memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s) + tid,
                                        cuda::aligned_size_t<4>(sizeof(int)));
                __pipeline_commit();
            }

            // compute_batch: next batch to process
            // fetch_batch:   next batch to fetch from global memory
            for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size;
                  ++compute_batch, ++fetch_batch) {
                // Wait for the first requested stage to complete.
                constexpr size_t pending_batches = num_stages - 1;
                __pipeline_wait_prior<pending_batches>();
                __syncthreads(); // Not required if each thread works on the data it copied.

                // Compute on the current batch.
                compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);

                __syncthreads(); // Not required if each thread works on the data it copied.

                // Load future stage ``num_stages`` ahead of current compute batch.
                if (fetch_batch < batch_size) {
                    __pipeline_memcpy_async(shared + shared_offset[stage] + tid,
                                            global_in + block_batch(fetch_batch) + tid,
                                            cuda::aligned_size_t<4>(sizeof(int)));
                }
                __pipeline_commit();
                stage = (stage + 1) % num_stages;
            }
        }

``cuda::memcpy_async`` 实现演示了使用 ``cuda::pipeline`` （参见 :ref:`管道 <pipelines>` ）和 ``cuda::memcpy_async`` 的多阶段数据预取。它：

- 初始化一个线程本地的管道
- 通过调度 ``num_stages`` 个 ``memcpy_async`` 操作启动管道
- 循环所有批次：阻塞所有线程等待当前批次完成，然后对当前批次执行计算，最后调度下一个 ``memcpy_async`` （如果有的话）

``cooperative_groups::memcpy_async`` 实现演示了使用 ``cooperative_groups::memcpy_async`` 的多阶段数据预取。
与前一个实现的主要区别是，我们不使用管道对象，而是依赖 ``cooperative_groups::memcpy_async`` 在底层分阶段调度内存传输。

CUDA C 原始 API 实现以与第一个非常相似的方式演示了使用低级原始 API 的多阶段数据预取。

此示例中实现高效代码生成的一个重要细节是保持 ``num_stages`` 个批次在管道中，即使没有更多批次要获取。
这是通过提交到管道来完成的，即使没有更多批次要获取（ ``pipeline.producer_commit()`` 或 ``__pipeline_commit()`` ）。
请注意，这对于 cooperative groups API 是不可能的，因为我们无法访问内部管道。

.. _async-copies-producer-consumer:

4.12.1.3. 通过 Warp 特化的生产者 - 消费者模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何实现生产者 - 消费者模式，其中单个 warp 专门化作为生产者，执行从 Global 到共享内存的异步数据拷贝，而剩余的 warp 从共享内存消费数据并执行计算。
为了启用生产者和消费者线程之间的并发性，我们在共享内存中使用双缓冲。
当消费者 warp 处理一个缓冲区中的数据时，生产者 warp 异步获取下一批数据到另一个缓冲区。

.. tab-set::

   .. tab-item:: cuda::memcpy_async

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cuda/pipeline>

        #pragma nv_diag_suppress static_var_with_dynamic_init

        using pipeline = cuda::pipeline<cuda::thread_scope_block>;

        __device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch,
                                float *buffer, int buffer_len, float *in, int N)
        {
          if (batch < num_batches)
          {
            pipe.producer_acquire();
            /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */
            cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x, in + batch * buffer_len + threadIdx.x,
                                cuda::aligned_size_t<4>(sizeof(float)), pipe);
            pipe.producer_commit();
          }
        }

        __device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch,
                                float *buffer, int buffer_len, float *out, int N)
        {
          pipe.consumer_wait();
          /* consume buffer(stage) and update out(batch) */
          pipe.consumer_release();
        }

        __global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
        {
          auto block = cooperative_groups::this_thread_block();
          constexpr int warpSize = 32;

          /* Shared memory buffer declared below is of size 2 * buffer_len
              so that we can alternatively work between two buffers.
              buffer_0 = buffer and buffer_1 = buffer + buffer_len */
          __shared__ extern float buffer[];

          const int num_batches = N / buffer_len;

          // Create a partitioned pipeline with 2 stages where the first warp is the producer and the other warps are consumers.
          constexpr auto scope = cuda::thread_scope_block;
          constexpr int num_stages = 2;
          cuda::std::size_t producer_count = warpSize;
          __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
          pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

          // Producer fills the pipeline
          if (block.thread_rank() < producer_count)
            for (int s = 0; s < num_stages; ++s)
              produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);

          // Process the batches
          int stage = 0;
          for (size_t b = 0; b < num_batches; ++b)
          {
            if (block.thread_rank() < producer_count)
            {
              // Producers prefetch the next batch
              produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
            }
            else
            {
              // Consumers consume the oldest batch
              consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
            }
            stage = (stage + 1) % num_stages;
          }
        }

   .. tab-item:: CUDA C primitives

      .. code-block:: cuda

        #include <cooperative_groups.h>
        #include <cuda_awbarrier_primitives.h>

        __device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N)
        {
          for (int i = 0; i < N / buffer_len; ++i)
          {
            __mbarrier_token_t token = __mbarrier_arrive(&ready[i % 2]); /* wait for buffer_(i%2) to be ready to be filled */
            while(!__mbarrier_try_wait(&ready[i % 2], token, 1000)) {}
            /* produce, i.e., fill in, buffer_(i%2) */
            __pipeline_memcpy_async(buffer + i * buffer_len + threadIdx.x, in + i * buffer_len + threadIdx.x,
                                    cuda::aligned_size_t<4>(sizeof(float)));
            __pipeline_arrive_on(filled[i % 2]);
            __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) is filled */
          }
        }

        __device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N)
        {
          __mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
          __mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
          for (int i = 0; i < N / buffer_len; ++i)
          {
            __mbarrier_token_t token = __mbarrier_arrive(&filled[i % 2]);
            while(!__mbarrier_try_wait(&filled[i % 2], token, 1000)) {}
            /* consume buffer_(i%2) */
            __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
          }
        }

        __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
        {
          /* Shared memory buffer declared below is of size 2 * buffer_len
              so that we can alternatively work between two buffers.
              buffer_0 = buffer and buffer_1 = buffer + buffer_len */
          __shared__ extern float buffer[];

          /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
              while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
          __shared__ __mbarrier_t bar[4];

          // Initialize the barriers
          auto block = cooperative_groups::this_thread_block();
          if (block.thread_rank() < 4)
            __mbarrier_init(bar + block.thread_rank(), block.size());
          __syncthreads();

          if (block.thread_rank() < warpSize)
            produce(bar, bar + 2, buffer, buffer_len, in, N);
          else
            consume(bar, bar + 2, buffer, buffer_len, out, N);
        }

``cuda::memcpy_async`` 实现演示了使用 ``cuda::memcpy_async`` 和具有 2 个阶段的 ``cuda::pipeline`` 的最高抽象级别 API。
它使用分区管道（参见 :ref:`管道 <pipelines>` ），其中第一个 warp 作为生产者，其余 warp 作为消费者。
生产者最初填充两个管道阶段。然后在主处理循环中，当消费者处理当前批次时，生产者为未来批次获取数据，保持稳定的工作流。

基于原始 API 的 CUDA C 实现结合 ``__pipeline_memcpy_async()`` 与 :ref:`共享内存屏障 <asynchronous-barriers>` 作为完成机制来协调异步内存传输。
``__pipeline_arrive_on()`` 函数将内存拷贝与屏障关联。
它将屏障到达计数增加一，当它之前的所有异步操作完成时，到达计数会自动减少一，因此对到达计数的净效应为零。
因此，我们还需要使用 ``__mbarrier_arrive()`` 显式等待屏障。

.. _using-tma:

.. _async-copies-tma:

4.12.2. 使用张量内存加速器 (TMA)
--------------------------------

许多应用程序需要将大量数据传输到 Global 内存和从 Global 内存传输。通常，数据在 Global 内存中布局为具有非顺序数据访问模式的多维数组。
为了减少 Global 内存访问，在用于计算之前，此类数组的子 tile 被复制到共享内存。加载和存储涉及容易出错且重复的地址计算。
为了卸载这些计算，计算能力 9.0（Hopper）及更高版本（参见 `PTX 文档 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions>`_）具有 **张量内存加速器** (Tensor Memory Accelerator, TMA)。
TMA 的主要目标是为多维数组提供从 Global 内存到共享内存的有效数据传输机制。

**命名**。 TMA 是用于指代本节中描述的功能的广泛术语。
为了前向兼容性和减少与 PTX ISA 的差异，本节中的文本将 TMA 操作称为 *批量异步拷贝* 或 *批量张量异步拷贝*，具体取决于使用的拷贝类型。
术语"批量"用于将这些操作与上一节中描述的异步内存操作进行对比。

**维度**。 TMA 支持复制一维和多维数组（最多 5 维）。一维连续数组的批量异步拷贝编程模型与多维数组的批量张量异步拷贝编程模型不同。
要执行多维数组的批量张量异步拷贝，硬件需要 `张量映射 <https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap>`_。
此对象描述多维数组在 Global 和共享内存中的布局。
张量映射通常在主机上使用 `cuTensorMapEncode API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY>`_ 创建，
然后作为用 ``__grid_constant__`` 注解的 ``const`` 内核参数从主机传输到设备（参见 :ref:`__grid_constant__ 参数 <grid-constant-parameters>` ）。
张量映射作为用 ``__grid_constant__`` 注解的 ``const`` 内核参数从主机传输到设备，可在设备上用于在共享和 Global 内存之间拷贝数据 tile。
相比之下，执行连续一维数组的批量异步拷贝不需要张量映射：它可以使用指针和大小参数在设备上执行。

**源和目标**。TMA 操作的源和目标地址可以在共享或 Global 内存中。
操作可以从 Global 读取到共享内存，从共享写入到 Global 内存，也可以从共享内存复制到同一集群中另一个块的 :ref:`分布式共享内存 <writing-cuda-kernels-distributed-shared-memory>` 。
此外，在集群中时，批量异步张量操作可以指定为 *多播*。
在这种情况下，数据可以从 Global 内存传输到集群中多个块的共享内存。
多播功能针对 ``sm_90a`` 目标架构进行了优化，
在其他目标上可能 `性能显著降低 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_。
因此，建议使用 `计算架构 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list>`_ ``sm_90a`` 使用它。

**异步性**。使用 TMA 的数据传输是 :ref:`异步的 <asynchronous-execution-features>` ，并建模为异步代理操作（参见 :ref:`异步线程和异步代理 <async-thread-and-async-proxy>` ）。
这允许发起线程继续计算，而硬件异步复制数据。
*数据传输是否实际异步执行取决于硬件实现，未来可能会发生变化*。
批量异步操作可以使用多种 `完成机制 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms>`_ 来发出完成信号。
当操作从 Global 读取到共享内存时，块中的任何线程都可以通过等待 :ref:`共享内存屏障 <asynchronous-barriers>` 等待数据在共享内存中可读。
当批量异步操作将数据从共享内存写入到 Global 或分布式共享内存时，只有发起线程可以等待操作完成。
这是通过使用 *批量异步组* 完成机制完成的。描述完成机制的表格见下文以及 `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy>`_ 。

.. list-table:: 使用 TMA 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持。
   :widths: 20 20 60
   :header-rows: 2

   * - 方向
     -
     - 异步拷贝 (TMA，CC 9.0+)
   * - 源
     - 目标
     - 完成机制
   * - Global
     - Global
     -
   * - shared::cta
     - global
     - bulk async-group
   * - global
     - shared::cta
     - shared memory barrier
   * - global
     - shared::cluster
     - shared memory barrier (multicast)
   * - shared::cta
     - shared::cluster
     - shared memory barrier
   * - shared::cta
     - shared::cta
     -

.. _async-copies-tma-one-dim:

4.12.2.1. 使用 TMA 传输一维数组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

下表总结了使用批量异步 TMA 的可能源和目标内存空间及完成机制以及 API。

.. list-table:: 使用批量异步 TMA 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持。
   :widths: 15 15 20 50
   :header-rows: 2

   * - 方向
     -
     - 批量异步拷贝 (TMA，CC9.0+)
     -
   * - 源
     - 目标
     - 完成机制
     - API
   * - Global
     - Global
     -
     -
   * - shared::cta
     - global
     - bulk async-group
     - `cuda::ptx::cp_async_bulk <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html>`_
   * - global
     - shared::cta
     - shared memory barrier
     - | `cuda::memcpy_async <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html>`_
       | :ref:`cuda::device::memcpy_async_tx<memcpy-async>`
       | `cuda::ptx::cp_async_bulk <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html>`_
   * - global
     - shared::cluster
     - shared memory barrier
     - `cuda::ptx::cp_async_bulk <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html>`_
   * - shared::cta
     - shared::cluster
     - shared memory barrier
     - `cuda::ptx::cp_async_bulk <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html>`_
   * - shared::cta
     - shared::cta
     -
     -

某些功能需要内联 PTX ，这些功能目前通过 `CUDA 标准 C++ 库 <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html>`_ 中的 ``cuda::ptx`` 命名空间提供。
可以使用以下代码检查这些包装器的可用性：

.. code-block:: cuda

   #if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
   static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
   #endif // __CUDA_MINIMUM_ARCH__

请注意，如果源地址和目标地址按 16 字节对齐且大小是 16 字节的倍数， ``cuda::memcpy_async`` 会使用 TMA ，否则它会回退到同步拷贝。
另一方面， ``cuda::device::memcpy_async_tx`` 和 ``cuda::ptx::cp_async_bulk`` 始终使用 TMA ，如果未满足要求，将导致未定义行为。

下面我们通过示例演示如何使用批量异步拷贝。该示例对一维数组进行读 - 改 - 写操作。
核函数经历以下步骤：

1. 初始化一个共享内存屏障，作为从 global 到共享内存的批量异步拷贝的完成机制。
2. 发起从 global 到共享内存的一块内存的拷贝。
3. 在共享内存屏障上到达并等待拷贝完成。
4. 递增共享内存缓冲区的值。
5. 使用代理栅栏（proxy fence）确保共享内存写入（通用代理）对后续的批量异步拷贝（异步代理）可见。
6. 发起将共享内存中的缓冲区批量异步拷贝到 global 内存。
7. 等待批量异步拷贝完成对共享内存的读取。

.. code-block:: cuda

   #include <cuda/barrier>
   #include <cuda/ptx>

   using barrier = cuda::barrier<cuda::thread_scope_block>;
   namespace ptx = cuda::ptx;

   static constexpr size_t buf_len = 1024;

   __device__ inline bool is_elected()
   {
       unsigned int tid = threadIdx.x;
       unsigned int warp_id = tid / 32;
       unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
       return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
   }

   __global__ void add_one_kernel(int* data, size_t offset)
   {
     // Shared memory buffer. The destination shared memory buffer of
     // a bulk operation should be 16 byte aligned.
     __shared__ alignas(16) int smem_data[buf_len];

     // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__ barrier bar;
     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads();

     // 2. Initiate TMA transfer to copy global to shared memory from a single thread.
     if (is_elected()) {
       // Launch the async copy and communicate how many bytes are expected to come in (the transaction count).

       // Version 1: cuda::memcpy_async
       cuda::memcpy_async(
           smem_data, data + offset,
           cuda::aligned_size_t<16>(sizeof(smem_data)),
           bar);

       // Version 2: cuda::device::memcpy_async_tx
       // cuda::device::memcpy_async_tx(
       //   smem_data, data + offset,
       //   cuda::aligned_size_t<16>(sizeof(smem_data)),
       //   bar);
       // cuda::device::barrier_expect_tx(
       //     cuda::device::barrier_native_handle(bar),
       //     sizeof(smem_data));

       // Version 3: cuda::ptx::cp_async_bulk
       // ptx::cp_async_bulk(
       //     ptx::space_shared, ptx::space_global,
       //     smem_data, data + offset,
       //     sizeof(smem_data),
       //     cuda::device::barrier_native_handle(bar));
       // cuda::device::barrier_expect_tx(
       //     cuda::device::barrier_native_handle(bar),
       //     sizeof(smem_data));
     }

     // 3a. All threads arrive on the barrier.
     barrier::arrival_token token = bar.arrive();

     // 3b. Wait for the data to have arrived.
     bar.wait(std::move(token));

     // 4. Compute saxpy and write back to shared memory.
     for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
       smem_data[i] += 1;
     }

     // 5. Wait for shared memory writes to be visible to TMA engine.
     ptx::fence_proxy_async(ptx::space_shared);
     __syncthreads();
     // After syncthreads, writes by all threads are visible to TMA engine.

     // 6. Initiate TMA transfer to copy shared memory to global memory.
     if (is_elected()) {
       ptx::cp_async_bulk(
           ptx::space_global, ptx::space_shared,
           data + offset, smem_data, sizeof(smem_data));
       // 7. Wait for TMA transfer to have finished reading shared memory.
       // Create a "bulk async-group" out of the previous bulk copy operation.
       ptx::cp_async_bulk_commit_group();
       // Wait for the group to have completed reading from shared memory.
       ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
     }
   }

**屏障初始化**。屏障以参与该块的线程数进行初始化。因此，只有当所有线程都在此屏障上到达后，屏障才会翻转。
共享内存屏障在 :ref:`共享内存屏障 <asynchronous-barriers>` 中有更详细的描述。

**TMA 读取**。批量异步拷贝指令指示硬件将一大块数据拷贝到共享内存，并在完成读取后更新共享内存屏障的事务计数（transaction count）。
通常，以尽可能少的次数、尽可能大的规模发起批量拷贝可获得最佳性能。由于拷贝可以由硬件异步执行，因此不需要将拷贝拆分为更小的块。

发起批量异步拷贝操作的线程还会告诉屏障预计将到达多少事务（tx）。
在本例中，事务以字节为单位计数。 ``cuda::memcpy_async`` 会自动执行这一步，而 ``cuda::device::memcpy_async_tx`` 和 ``cuda::ptx::cp_async_bulk`` 则不会，在这两者之后需要显式调用 ``cuda::ptx::mbarrier_expect_tx`` 。
如果多个线程更新事务计数，预期事务计数将是所有更新的总和。只有当所有线程都已到达并且所有字节都已到达时，屏障才会翻转。
一旦屏障翻转，这些字节就可以安全地从共享内存中读取，线程和后续的批量异步拷贝都可以读取。
有关屏障事务计数的更多信息，请参见 :ref:`跟踪异步内存操作 <async-barriers-tracking-async-mem-ops>` 。

**屏障等待**。使用 token 通过 ``bar.wait()`` 等待屏障翻转。使用屏障的显式阶段跟踪（参见 :ref:`显式阶段跟踪 <async-barriers-explicit-phase-tracking>` ）可能更高效。

**共享内存写入与同步**。缓冲区值的递增会读取和写入共享内存。为了使写入对后续的批量异步拷贝可见，使用 ``cuda::ptx::fence_proxy_async`` 函数。
这将共享内存的写入排在后续通过异步代理（async proxy）读取的批量异步拷贝操作的读取之前。
因此，每个线程首先通过 ``cuda::ptx::fence_proxy_async`` 在异步代理中对共享内存对象的写入进行排序，并且所有线程的这些操作通过 ``__syncthreads()`` 排在线程 0 执行的异步操作之前。

**TMA 写入与同步**。从共享内存到 global 内存的写入同样由单个线程发起。该写入的完成不由共享内存屏障跟踪，而是使用线程本地机制。
多个写入可以批量组成所谓的批量异步组（bulk async-group）。之后，线程可以等待该组中的所有操作完成从共享内存读取（如上面的代码所示），或完成向 global 内存写入，使写入对发起线程可见。
有关详细信息，
请参考 `cp.async.bulk.wait_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group>`_ 的 PTX ISA 文档。
请注意，批量异步和非批量异步拷贝指令具有不同的异步组：同时存在 ``cp.async.wait_group`` 和 ``cp.async.bulk.wait_group`` 指令。

.. note::

   建议由块中的单个线程发起 TMA 操作。
   虽然使用 ``if (threadIdx.x == 0)`` 看起来可能足够，但编译器无法验证确实只有一个线程发起拷贝，并可能为所有活动线程插入剥离循环，这会导致 warp 序列化和性能降低。
   为了防止这种情况，我们定义 ``is_elected()`` 辅助函数，使用 ``cuda::ptx::elect_sync`` 从 warp 0（编译器已知的）选择一个线程执行拷贝，允许编译器生成更高效的代码。
   或者，可以使用 :ref:`cooperative_groups::invoke_one <cg-invoke-one>` 实现相同的效果。

批量异步指令对其源和目标地址有特定的对齐要求。更多信息请见下表。

.. list-table:: 一维批量异步操作的对齐要求
   :widths: 40 60
   :header-rows: 1

   * - 地址/大小
     - 对齐要求
   * - Global 内存地址
     - 必须 16 字节对齐
   * - 共享内存地址
     - 必须 16 字节对齐
   * - 共享内存屏障地址
     - 必须 8 字节对齐（由 ``cuda::barrier`` 保证）
   * - 传输大小
     - 必须是 16 字节的倍数

.. _async-copies-tma-one-dim-staging:

4.12.2.1.1. 预取数据
"""""""""""""""""""""""""""""

在此示例中，我们将演示如何使用 TMA 从 Global 内存预取数据到共享内存。
在迭代拷贝和计算模式中，这允许用当前迭代的计算隐藏未来迭代的数据传输延迟，可能增加飞行中的字节数。

使用 CUDA C++ ``cuda::device::memcpy_async_tx``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/barrier>
   #include <cuda/ptx>

   namespace ptx = cuda::ptx;
   namespace cg = cooperative_groups;

   __device__ inline bool is_elected()
   {
       unsigned int tid = threadIdx.x;
       unsigned int warp_id = tid / 32;
       unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
       return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
   }

   template <int block_size, int num_stages>
   __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
       auto grid = cg::this_grid();
       auto block = cg::this_thread_block();
       const int tid = threadIdx.x;
       assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size

       // 1. Initialization Phase
       __shared__ int shared[num_stages * block_size];
       size_t shared_offset[num_stages];
       for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

       auto block_batch = [&](size_t batch) -> int {
           return block.group_index().x * block.size() + grid.size() * batch;
       };

       // Initialize shared memory barrier with the number of threads participating in the barrier.
       // We will use explicit phase tracking for the barrier, which allows us to have only one
       // thread arrive on the barrier to set the transaction count and other threads wait for
       // a parity-based phase flip.
       #pragma nv_diag_suppress static_var_with_dynamic_init
       __shared__ cuda::barrier<cuda::thread_scope_block> bar[num_stages];
       if (tid == 0) {
           #pragma unroll num_stages
           for (int i = 0; i < num_stages; i++) {
               init(&bar[i], 1);
           }
       }
       __syncthreads();

       // Fill the pipeline with the first ``num_stages`` batches.
       if (is_elected()) {
           size_t num_bytes = block_size * sizeof(int);

           #pragma unroll num_stages
           for (int s = 0; s < num_stages; ++s) {
               cuda::device::memcpy_async_tx(&shared[shared_offset[s]], &global_in[block_batch(s)], cuda::aligned_size_t<16>(num_bytes), bar[s]);
               (void)cuda::device::barrier_arrive_tx(bar[s], 1, num_bytes);
           }
       }

       // 2. Main Processing Loop.
       // compute_batch: next batch to process.
       // fetch_batch:   next batch to fetch from global memory.
       int stage = 0;       // current stage in the shared memory buffer.
       uint32_t parity = 0; // barrierparity
       for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size;
            ++compute_batch, ++fetch_batch) {
           // (a) Wait on current batch.
           while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, cuda::device::barrier_native_handle(bar[stage]), parity)) {}

           // (b) Compute on the current batch.
           compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);
           __syncthreads();

           // (c) Load next stage ``num_stages`` ahead of current compute batch.
           if (is_elected() && fetch_batch < batch_size) {
               size_t num_bytes = block_size * sizeof(int);
               cuda::device::memcpy_async_tx(&shared[shared_offset[stage]], &global_in[block_batch(fetch_batch)], cuda::aligned_size_t<16>(num_bytes), bar[stage]);
               (void)cuda::device::barrier_arrive_tx(bar[stage], 1, num_bytes);
           }

           // (d) Stage management.
           stage++;
           if (stage == num_stages) {
               stage = 0;
               parity ^= 1;
           }
       }
   }

该示例使用 ``cuda::device::memcpy_async_tx`` 实现 TMA 拷贝的多阶段数据预取，并采用具有显式阶段跟踪的共享内存屏障来同步拷贝。

- **初始化阶段**：设置共享内存屏障（每个阶段一个），并将前 ``num_stages`` 个批次预加载到不同的共享内存段中。

- **主处理循环**：

  - **等待**：使用 ``mbarrier_try_wait_parity()`` 等待当前批次完成拷贝。

  - **计算**：处理当前批次的数据。

  - **预取**：为未来的数据调度下一个 ``memcpy_async_tx`` 操作（保持领先 ``num_stages`` 个阶段）。

  - **阶段管理**：使用循环缓冲区方式在各阶段之间循环，并跟踪屏障的奇偶性。

.. _async-copies-tma-multi-dim:

4.12.2.2. 使用 TMA 传输多维数组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在本节中，我们将重点讨论多维 TMA 拷贝。一维与多维情况之间的主要区别在于：
必须在主机上创建张量映射（tensor map），并将其传递给 CUDA kernel 。
下表总结了使用批量张量异步 TMA 的异步拷贝的可能源和目标内存空间、完成机制，
以及在设备代码中暴露该功能的 API。

.. list-table:: 使用批量张量异步 TMA 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持。
   :widths: 15 15 20 50
   :header-rows: 2

   * - 方向
     -
     - 批量张量异步拷贝 (TMA)
     -
   * - 源
     - 目标
     - 完成机制
     - API
   * - global
     - global
     -
     -
   * - shared::cta
     - global
     - bulk async-group
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - global
     - shared::cta
     - shared memory barrier
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - global
     - shared::cluster
     - shared memory barrier
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - shared::cta
     - shared::cluster
     - shared memory barrier
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - shared::cta
     - shared::cta
     -
     -

所有功能都需要内联 PTX ，目前这些内联 PTX 通过 CUDA Standard C++ 库中的
``cuda::ptx`` 命名空间提供。
在下文中，我们将描述如何使用 CUDA Driver API 创建张量映射，如何将其传递给设备，
以及如何在设备上使用它。

**Driver API**。张量映射使用 ``cuTensorMapEncodeTiled`` driver API 创建。
该 API 可以通过直接链接到 Driver 的 ``-lcuda`` 库，或使用
``cudaGetDriverEntryPointByVersion`` API 来访问。
下面，我们展示如何获取 ``cuTensorMapEncodeTiled`` API 的指针。
更多信息，请参阅 :ref:`driver-entry-point-access-details` 。

.. code-block:: cuda

   #include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

   PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
     // Get pointer to cuTensorMapEncodeTiled
     cudaDriverEntryPointQueryResult driver_status;
     void* cuTensorMapEncodeTiled_ptr = nullptr;
     CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                                  cudaEnableDefault, &driver_status));
     assert(driver_status == cudaDriverEntryPointSuccess);

     return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
   }

**创建**：创建张量映射需要很多参数，其中包括 global 内存中数组的基指针、
数组的大小（以元素个数计）、从一行到下一行的跨步（以字节计）、
共享内存缓冲区的大小（以元素个数计）。
下面的代码创建了一个描述大小为 GMEM_HEIGHT x GMEM_WIDTH 的二维行主序数组的张量映射。
注意参数的顺序：变化最快的维度排在最前面。

.. code-block:: cuda

   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
   // The stride is the number of bytes to traverse from the first element of one row to the next.
   // It must be a multiple of 16.
   uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
   // The box_size is the size of the shared memory buffer that is used as the
   // destination of a TMA transfer.
   uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
   // The distance between elements in units of sizeof(element). A stride of 2
   // can be used to load only the real component of a complex-valued tensor, for instance.
   uint32_t elem_stride[rank] = {1, 1};

   // Get a function pointer to the cuTensorMapEncodeTiled driver API.
   auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
     &tensor_map,                // CUtensorMap *tensorMap,
     CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
     rank,                       // cuuint32_t tensorRank,
     tensor_ptr,                 // void *globalAddress,
     size,                       // const cuuint64_t *globalDim,
     stride,                     // const cuuint64_t *globalStrides,
     box_size,                   // const cuuint32_t *boxDim,
     elem_stride,                // const cuuint32_t *elementStrides,
     // Interleave patterns can be used to accelerate loading of values that
     // are less than 4 bytes long.
     CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
     // Swizzling can be used to avoid shared memory bank conflicts.
     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
     // L2 Promotion can be used to widen the effect of a cache-policy to a wider
     // set of L2 cache lines.
     CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
     // Any element that is outside of bounds will be set to zero by the TMA transfer.
     CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

**主机到设备传输**：让设备代码可以访问张量映射有三种方式。
推荐的方法是将张量映射作为 ``const __grid_constant__`` 参数传递给 kernel 。
其他可能的方式是：使用 ``cudaMemcpyToSymbol`` 将张量映射复制到设备的
``__constant__`` 内存中，或者通过 global 内存访问它。
将张量映射作为参数传递时，某些版本的 GCC C++ 编译器会发出警告
"the ABI for passing parameters with 64-byte alignment has changed in GCC 4.6" 。
此警告可以忽略。

.. code-block:: cuda

   #include <cuda.h>

   __global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
   {
      // Use tensor_map here.
   }
   int main() {
     CUtensorMap map;
     // [ ..Initialize map.. ]
     kernel<<<1, 1>>>(map);
   }

作为 ``__grid_constant__`` kernel 参数的替代，可以使用全局 ``__constant__``
变量。下面包含一个示例。

.. code-block:: cuda

   #include <cuda.h>

   __constant__ CUtensorMap global_tensor_map;
   __global__ void kernel()
   {
     // Use global_tensor_map here.
   }
   int main() {
     CUtensorMap local_tensor_map;
     // [ ..Initialize map.. ]
     cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
     kernel<<<1, 1>>>();
   }

最后，也可以将张量映射复制到 global 内存。使用指向 global 设备内存中张量映射的指针时，
在该线程块中的任何线程使用更新后的张量映射之前，每个线程块都需要执行一次栅栏（fence）操作。
除非张量映射再次被修改，否则该线程块后续对张量映射的使用不需要再次加栅栏。
请注意，这种机制可能比上述两种机制更慢。

.. code-block:: cuda

   #include <cuda.h>
   #include <cuda/ptx>
   namespace ptx = cuda::ptx;

   __device__ CUtensorMap global_tensor_map;
   __global__ void kernel(CUtensorMap *tensor_map)
   {
     // Fence acquire tensor map:
     ptx::n32_t<128> size_bytes;
     // Since the tensor map was modified from the host using cudaMemcpy,
     // the scope should be .sys.
     ptx::fence_proxy_tensormap_generic(
        ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
    );
    // Safe to use tensor_map after fence inside this thread.
   }
   int main() {
     CUtensorMap local_tensor_map;
     // [ ..Initialize map.. ]
     cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
     kernel<<<1, 1>>>(global_tensor_map);
   }

**使用**：下面的 kernel 从一个更大的二维数组中加载大小为 SMEM_HEIGHT x SMEM_WIDTH
的二维 tile 。tile 的左上角由索引 ``x`` 和 ``y`` 指出。
该 tile 被加载到共享内存中，经过修改后，再写回 global 内存。

.. code-block:: cuda

   #include <cuda.h>         // CUtensormap
   #include <cuda/barrier>

   using barrier = cuda::barrier<cuda::thread_scope_block>;
   namespace ptx = cuda::ptx;

   __device__ inline bool is_elected()
   {
       unsigned int tid = threadIdx.x;
       unsigned int warp_id = tid / 32;
       unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
       return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
   }

   __global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
     // The destination shared memory buffer of a bulk tensor operation should be
     // 128 byte aligned.
     __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

     // Initialize shared memory barrier with the number of threads participating in the barrier.
     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__ barrier bar;

     if (threadIdx.x == 0) {
       // Initialize barrier. All `blockDim.x` threads in block participate.
       init(&bar, blockDim.x);
     }
     // Syncthreads so initialized barrier is visible to all threads.
     __syncthreads();

     barrier::arrival_token token;
     if (is_elected()) {
       // Initiate bulk tensor copy.
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(
         ptx::space_shared, ptx::space_global,
         &smem_buffer, &tensor_map, tensor_coords,
         cuda::device::barrier_native_handle(bar));
       // Arrive on the barrier and tell how many bytes are expected to come in.
       token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
     } else {
       // Other threads just arrive.
       token = bar.arrive();
     }
     // Wait for the data to have arrived.
     bar.wait(std::move(token));

     // Symbolically modify a value in shared memory.
     smem_buffer[0][threadIdx.x] += threadIdx.x;

     // Wait for shared memory writes to be visible to TMA engine.
     ptx::fence_proxy_async(ptx::space_shared);
     __syncthreads();
     // After syncthreads, writes by all threads are visible to TMA engine.

     // Initiate TMA transfer to copy shared memory to global memory
     if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(
         ptx::space_global, ptx::space_shared,
         &tensor_map, tensor_coords, &smem_buffer);
       // Wait for TMA transfer to have finished reading shared memory.
       // Create a "bulk async-group" out of the previous bulk copy operation.
       ptx::cp_async_bulk_commit_group();
       // Wait for the group to have completed reading from shared memory.
       ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
     }

     // Destroy barrier. This invalidates the memory region of the barrier. If
     // further computations were to take place in the kernel, this allows the
     // memory location of the shared memory barrier to be reused.
     if (threadIdx.x == 0) {
       (&bar)->~barrier();
     }
   }

**负索引和越界**：当从 global 内存读取到共享内存的 tile 的一部分越界时，
与该越界区域对应的共享内存会被零填充。tile 的左上角索引也可以为负。
当从共享内存写入 global 内存时，tile 的某些部分可以越界，但左上角不能有任何负索引。

**大小和跨步**：张量的大小是沿一个维度的元素个数。所有大小都必须大于一。
跨步是同一维度的元素之间相隔的字节数。例如，一个 4 x 4 的整数矩阵的大小为 4 和 4。
由于每个元素占 4 字节，其跨步为 4 和 16 字节。
由于对齐要求，一个 4 x 3 的行主序整数矩阵的跨步同样必须是 4 和 16 字节：
每一行被填充额外的 4 个字节，以确保下一行的起始位置按 16 字节对齐。
有关对齐要求的更多信息可以在下表中找到。

.. list-table:: 多维批量张量异步拷贝操作的对齐要求
   :widths: 40 60
   :header-rows: 1

   * - 地址/大小
     - 对齐要求
   * - Global 内存地址
     - 必须 16 字节对齐
   * - Global 内存大小
     - 必须大于或等于一。不必是 16 字节的倍数。
   * - Global 内存跨步
     - 必须是 16 字节的倍数
   * - 共享内存地址
     - 必须 128 字节对齐
   * - 共享内存屏障地址
     - 必须 8 字节对齐（由 ``cuda::barrier`` 保证）。
   * - 传输大小
     - 必须是 16 字节的倍数

.. _async-copies-tma-encode-on-device:

4.12.2.2.1. 在设备上编码张量映射
""""""""""""""""""""""""""""""""

前面的章节已经描述了如何使用 CUDA Driver API 在主机上创建张量映射。

本节介绍如何在设备上编码平铺类型（tiled-type）的张量映射。这在典型传输方式（使用 ``const __grid_constant__`` kernel 参数）不适用的情况下非常有用，例如在单次 kernel 启动中处理一批不同大小的张量时。

推荐的模式如下：

1. 在主机上使用 Driver API 创建张量映射"模板" ``template_tensor_map`` 。
2. 在设备 kernel 中，复制 ``template_tensor_map`` ，修改副本，存储到全局内存中，并适当地设置内存栅栏。
3. 在 kernel 中使用张量映射，并配合适当的栅栏操作。

高级代码结构如下：

.. code-block:: c++

   // Initialize device context:
   CUDA_CHECK(cudaDeviceSynchronize());

   // Create a tensor map template using the cuTensorMapEncodeTiled driver function
   CUtensorMap template_tensor_map = make_tensormap_template();

   // Allocate tensor map and tensor in global memory
   CUtensorMap* global_tensor_map;
   CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
   char* global_buf;
   CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

   // Fill global buffer with data.
   fill_global_buf<<<1, 1>>>(global_buf);

   // Define the parameters of the tensor map that will be created on device.
   tensormap_params p{};
   p.global_address    = global_buf;
   p.rank              = 2;
   p.box_dim[0]        = 128; // The box in shared memory has half the width of the full buffer
   p.box_dim[1]        = 4;   // The box in shared memory has half the height of the full buffer
   p.global_dim[0]     = 256; //
   p.global_dim[1]     = 8;   //
   p.global_stride[0]  = 256; //
   p.element_stride[0] = 1;   //
   p.element_stride[1] = 1;   //

   // Encode global_tensor_map on device:
   encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

   // Use it from another kernel:
   consume_tensor_map<<<1, 1>>>(global_tensor_map);

   // Check for errors:
   CUDA_CHECK(cudaDeviceSynchronize());

以下章节描述了高级步骤。在示例中，以下 ``tensormap_params`` 结构体包含要更新的字段的新值，供阅读示例时参考。

.. code-block:: c++

   struct tensormap_params {
     void* global_address;
     int rank;
     uint32_t box_dim[5];
     uint64_t global_dim[5];
     size_t global_stride[4];
     uint32_t element_stride[5];
   };

.. _async-copies-tma-device-encode-modify:

4.12.2.2.2. 设备端张量映射的编码与修改
""""""""""""""""""""""""""""""""""""""""

在全局内存中编码张量映射的推荐流程如下：

1. 将已有的张量映射 ``template_tensor_map`` 传递给 kernel 。与在 ``cp.async.bulk.tensor`` 指令中使用张量映射的 kernel 不同，这可以通过任何方式完成：全局内存指针、kernel 参数、 ``__constant__`` 变量等。
2. 使用 ``template_tensor_map`` 值在共享内存中拷贝初始化一个张量映射。
3. 使用 ``cuda::ptx::tensormap_replace`` 函数修改共享内存中的张量映射。这些函数封装了 ``tensormap.replace`` PTX 指令，可用于修改平铺类型张量映射的任何字段，包括基地址、大小、步幅等。
4. 使用 ``cuda::ptx::tensormap_copy_fenceproxy`` 函数将修改后的张量映射从共享内存复制到全局内存，并执行必要的栅栏操作。

以下代码包含遵循这些步骤的 kernel 。为了完整性，它修改了张量映射的所有字段。通常，kernel 只会修改少数字段。

在此 kernel 中， ``template_tensor_map`` 作为 kernel 参数传递。这是将 ``template_tensor_map`` 从主机移动到设备的首选方式。如果 kernel 旨在更新设备内存中已有的张量映射，它可以接受指向已有张量映射的指针以进行修改。

.. note::

   张量映射的格式可能会随时间变化。因此， ``cuda::ptx::tensormap_replace`` 函数及相应的 ``tensormap.replace.tile`` PTX 指令被标记为特定于 sm_90a 。要使用它们，请使用 ``nvcc -arch sm_90a ....`` 编译。

.. tip::

   在 sm_90a 上，共享内存中零初始化的缓冲区也可用作初始张量映射值。这使得可以完全在设备上编码张量映射，而无需使用 Driver API 编码 ``template_tensor_map`` 值。

.. note::

   设备端修改仅支持平铺类型的张量映射；其他类型的张量映射不能在设备上修改。有关张量映射类型的更多信息，请参阅 Driver API 参考。

.. code-block:: c++

   #include <cuda/ptx>

   namespace ptx = cuda::ptx;

   // launch with 1 warp.
   __launch_bounds__(32)
   __global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map,
                                     tensormap_params p, CUtensorMap* out) {
      __shared__ alignas(128) CUtensorMap smem_tmap;
      if (threadIdx.x == 0) {
         // Copy template to shared memory:
         smem_tmap = template_tensor_map;

         const auto space_shared = ptx::space_shared;
         ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
         // For field .rank, the operand new_val must be ones less than the desired
         // tensor rank as this field uses zero-based numbering.
         ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

         // Set box dimensions:
         if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
         if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
         if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
         if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
         if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
         // Set global dimensions:
         if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
         if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
         if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
         if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
         if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
         // Set global stride:
         if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
         if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
         if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
         if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
         // Set element stride:
         if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
         if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
         if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
         if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
         if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

         // These constants are documented in this table:
         // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
         auto u8_elem_type = ptx::n32_t<0>{};
         ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
         auto no_interleave = ptx::n32_t<0>{};
         ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
         auto no_swizzle = ptx::n32_t<0>{};
         ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
         auto zero_fill = ptx::n32_t<0>{};
         ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
      }
      // Synchronize the modifications with other threads in warp
      __syncwarp();
      // Copy the tensor map to global memory collectively with threads in the warp.
      // In addition: make the updated tensor map visible to other threads on device that
      // for use with cp.async.bulk.
      ptx::n32_t<128> bytes_128;
      ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
   }

.. _async-copies-tma-usage-modified:

4.12.2.2.3. 使用修改后的张量映射
""""""""""""""""""""""""""""""""

与使用作为 ``const __grid_constant__`` kernel 参数传递的张量映射不同，使用全局内存中的张量映射需要在修改张量映射的线程和使用张量映射的线程之间，在张量映射代理中显式建立 release-acquire 模式。

release 部分已在前一节中展示，通过 ``cuda::ptx::tensormap.cp_fenceproxy`` 函数完成。

acquire 部分通过 ``cuda::ptx::fence_proxy_tensormap_generic`` 函数完成，该函数封装了 ``fence.proxy.tensormap::generic.acquire`` 指令。如果参与 release-acquire 模式的两个线程在同一设备上， ``.gpu`` 作用域即可。如果线程在不同设备上，则必须使用 ``.sys`` 作用域。一旦一个线程获取了张量映射，在充分同步后（例如使用 ``__syncthreads()`` ），它可以被块中的其他线程使用。使用张量映射的线程和执行栅栏操作的线程必须在同一个块中。也就是说，例如，如果线程位于同一集群的两个不同线程块、同一网格或不同 kernel 中， ``cooperative_groups::cluster`` 或 ``grid_group::sync()`` 等同步 API 或流有序同步不足以建立张量映射更新的排序，即这些其他线程块中的线程在使用更新后的张量映射之前仍需要在正确的作用域获取张量映射代理。如果没有中间修改，则不必在每次 ``cp.async.bulk.tensor`` 指令之前重复栅栏操作。

以下示例展示了 ``fence`` 及后续张量映射的使用。

.. code-block:: c++

   // Consumer of tensor map in global memory:
   __global__ void consume_tensor_map(CUtensorMap* tensor_map) {
     // Fence acquire tensor map:
     ptx::n32_t<128> size_bytes;
     ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
     // Safe to use tensor_map after fence.

     __shared__ uint64_t bar;
     __shared__ alignas(128) char smem_buf[4][128];

     if (threadIdx.x == 0) {
       // Initialize barrier
       ptx::mbarrier_init(&bar, 1);
       // Issue TMA request
       ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);
       // Arrive on barrier. Expect 4 * 128 bytes.
       ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
     }
     const int parity = 0;
     // Wait for load to have completed
     while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

     // print items:
     printf("Got:\n\n");
     for (int j = 0; j < 4; ++j) {
       for (int i = 0; i < 128; ++i) {
         printf("%3d ", smem_buf[j][i]);
         if (i % 32 == 31) { printf("\n"); };
       }
       printf("\n");
     }
   }

.. _async-copies-tma-template-driver-api:

4.12.2.2.4. 使用 Driver API 创建模板张量映射值
""""""""""""""""""""""""""""""""""""""""""""""

以下代码创建一个最小的平铺类型张量映射，后续可在设备上进行修改。

.. code-block:: c++

   CUtensorMap make_tensormap_template() {
     CUtensorMap template_tensor_map{};
     auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

     uint32_t dims_32         = 16;
     uint64_t dims_strides_64 = 16;
     uint32_t elem_strides    = 1;

     // Create the tensor descriptor.
     CUresult res = cuTensorMapEncodeTiled(
       &template_tensor_map, // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
       1,                // cuuint32_t tensorRank,
       nullptr,          // void *globalAddress,
       &dims_strides_64, // const cuuint64_t *globalDim,
       &dims_strides_64, // const cuuint64_t *globalStrides,
       &dims_32,         // const cuuint32_t *boxDim,
       &elem_strides,    // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

     CU_CHECK(res);
     return template_tensor_map;
   }

.. _async-copies-tma-swizzling:

4.12.2.2.5. 共享内存 Bank 交错
""""""""""""""""""""""""""""""""

默认情况下，TMA 引擎按照数据在 global memory 中的布局顺序将其加载到共享内存中。然而，对于某些共享内存访问模式，这种布局可能并不是最优的，因为它可能导致共享内存 bank 冲突。为了提升性能并减少 bank 冲突，我们可以通过应用"swizzle 模式"来改变共享内存布局。共享内存由 32 个 bank 组成，其组织方式使得连续的 32 位字映射到连续的 bank。每个 bank 每时钟周期的带宽为 32 位。在加载和存储共享内存时，如果同一个 bank 在一次事务中被多次使用，就会产生 bank 冲突，从而导致带宽下降。参见 :ref:`writing-cuda-kernels-shared-memory-access-patterns`。

为了确保共享内存中的数据布局能够让用户代码避免共享内存 bank 冲突，可以指示 TMA 引擎在将数据存储到共享内存之前对数据进行"swizzle"（洗牌），并在将数据从共享内存拷贝回 global memory 时进行"unswizzle"（逆洗牌）。tensor map 中编码了"swizzle 模式"，指明所使用的 swizzle 模式。

.. note::

   交错仅在计算能力 9 及更高版本上受支持。交错功能仅适用于使用 ``cuda::ptx::cp_async_bulk_tensor`` 指令的多维数组。

**示例：矩阵转置**

一个示例是矩阵转置，其中数据的访问从按行映射为按列。数据在 global memory 中以行优先（row major）方式存储，但我们也希望在共享内存中按列访问它，这会导致 bank 冲突。然而，通过使用 128 字节的"swizzle"模式以及新的共享内存索引，这些冲突被消除了。

在该示例中，我们加载一个 8x8 的 ``int4`` 类型矩阵，它在 global memory 中以行优先方式存储并载入共享内存。然后，每 8 个线程一组，从共享内存缓冲区中加载一行，并将其存储到另一个转置共享内存缓冲区的一列中。这会在存储时产生八路 bank 冲突。最后，将转置缓冲区写回 global memory。

为了避免 bank 冲突，可以使用 ``CU_TENSOR_MAP_SWIZZLE_128B`` 布局。该布局与 128 字节的行长匹配，并以一种使按列和按行访问都不需要在同一事务中使用相同 bank 的方式改变共享内存布局。

下面的 图 51 和 图 52 分别展示了 ``int4`` 类型 8x8 矩阵及其转置矩阵的常规共享内存布局和 swizzle 后的共享内存布局。颜色指示矩阵元素被映射到八个四 bank 分组中的哪一组，边缘行和边缘列列出了 global memory 的行和列索引。表项展示了 16 字节矩阵元素的共享内存索引。

.. figure:: /_static/images/swizzle-example1.png
   :alt: 无 swizzle 的共享内存数据布局
   :align: center

   图 51 在无 swizzle 的共享内存数据布局中，共享内存索引与 global memory 索引相同。每条加载指令读取一行，并将其存储在转置缓冲区的一列中。由于转置缓冲区中该列的所有矩阵元素都落在同一个 bank，存储必须串行化，产生八次存储事务，即每存储一列就有八路 bank 冲突。

.. figure:: /_static/images/swizzle-example2.png
   :alt: 使用 CU_TENSOR_MAP_SWIZZLE_128B 的共享内存数据布局
   :align: center

   图 52 使用 ``CU_TENSOR_MAP_SWIZZLE_128B`` 交错的共享内存数据布局。一行被存储到一列中，无论按行还是按列，每个矩阵元素都位于不同的 bank，因此没有任何 bank 冲突。

.. code-block:: cuda

   __global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
      // The destination shared memory buffer of a bulk tensor operation
      // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
      __shared__ alignas(1024) int4 smem_buffer[8][8];
      __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

      // Initialize shared memory barrier
      #pragma nv_diag_suppress static_var_with_dynamic_init
      __shared__ barrier bar;

      if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
      }
      __syncthreads();

      barrier::arrival_token token;
      if (is_elected()) {
        // Initiate bulk tensor copy from global to shared memory,
        // in the same way as without swizzle.
        int32_t tensor_coords[2] = { 0, 0 };
        ptx::cp_async_bulk_tensor(
          ptx::space_shared, ptx::space_global,
          &smem_buffer, &tensor_map, tensor_coords,
          cuda::device::barrier_native_handle(bar));
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
      } else {
        token = bar.arrive();
      }

      bar.wait(std::move(token));

      /* Matrix transpose
       *  When using the normal shared memory layout, there are eight
       *  8-way shared memory bank conflict when storing to the transpose.
       *  When enabling the 128-byte swizzle pattern and using the according access pattern,
       *  they are eliminated both for load and store. */
      for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
         for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
            const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
            const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
            smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
         }
      }

      // Wait for shared memory writes to be visible to TMA engine.
      ptx::fence_proxy_async(ptx::space_shared);
      __syncthreads();

      /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
       * it will 'unswizzle' the data. */
      if (is_elected()) {
          int32_t tensor_coords[2] = { x, y };
          ptx::cp_async_bulk_tensor(
            ptx::space_global, ptx::space_shared,
            &tensor_map, tensor_coords, &smem_buffer_tr);
         ptx::cp_async_bulk_commit_group();
         ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
      }

      // Destroy barrier
      if (threadIdx.x == 0) {
        (&bar)->~barrier();
      }
   }

   // --------------------------------- main ----------------------------------------

   int main(){

   ...
      void* tensor_ptr = d_data;

      CUtensorMap tensor_map{};
      // rank is the number of dimensions of the array.
      constexpr uint32_t rank = 2;
      // global memory size
      uint64_t size[rank] = {4*8, 8};
      // global memory stride, must be a multiple of 16.
      uint64_t stride[rank - 1] = {8 * sizeof(int4)};
      // The inner shared memory box dimension in bytes, equal to the swizzle span.
      uint32_t box_size[rank] = {4*8, 8};

      uint32_t elem_stride[rank] = {1, 1};

      // Create the tensor descriptor.
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map,                // CUtensorMap *tensorMap,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
          rank,                       // cuuint32_t tensorRank,
          tensor_ptr,                 // void *globalAddress,
          size,                       // const cuuint64_t *globalDim,
          stride,                     // const cuuint64_t *globalStrides,
          box_size,                   // const cuuint32_t *boxDim,
          elem_stride,                // const cuuint32_t *elementStrides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          // Using a swizzle pattern of 128 bytes.
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
      );

      kernel_tma<<<1, 8>>>(tensor_map);
    ...
   }

**备注。** 该示例旨在展示 swizzle 的用法，按"原样"使用并不高效，也无法扩展到给定尺寸之外。

**解释。** 在数据传输期间，TMA 引擎按照 swizzle 模式对数据进行洗牌，如下面的表格所述。这些 swizzle 模式定义了沿 swizzle 宽度分布的 16 字节块到四 bank 子组的映射关系。其类型为 ``CUtensorMapSwizzle``，有四个选项：none、32 字节、64 字节和 128 字节。请注意，共享内存框的内部维度必须小于或等于 swizzle 模式的跨度。

**交错模式**

如前所述，共有四种 swizzle 模式。下面的表格展示了不同的 swizzle 模式，包括新的共享内存索引之间的关系。这些表格定义了沿 128 字节分布的 16 字节块到八个四 bank 子组的映射。

.. figure:: /_static/images/swizzle-pattern.png
   :alt: TMA swizzle 模式总览
   :align: center

   图 53 TMA swizzle 模式总览

**注意事项。** 应用 TMA swizzle 模式时，必须遵守特定的内存要求：

- **Global memory 对齐** ：global memory 必须按 128 字节对齐。
- **共享内存对齐** ：为简单起见，共享内存应按照 swizzle 模式重复一次的字节数进行对齐。当共享内存缓冲区没有按 swizzle 模式重复一次的字节数对齐时，swizzle 模式与共享内存之间存在一个偏移。见下方注释。
- **内部维度** ：共享内存块的内部维度必须满足下方汇总表中所列的大小要求。若不满足这些要求，则该指令被视为无效。此外，如果 swizzle 宽度超过内部维度，请确保共享内存的分配能够容纳完整的 swizzle 宽度。
- **粒度** ：swizzle 映射的粒度固定为 16 字节。这意味着数据以 16 字节的块进行组织和访问，在规划内存布局和访问模式时必须考虑这一点。

**Swizzle 模式指针偏移计算。** 这里介绍当共享内存缓冲区没有按 swizzle 模式重复一次的字节数对齐时，如何确定 swizzle 模式与共享内存之间的偏移。

使用 TMA 时，要求共享内存按 128 字节对齐。要据此求出共享内存缓冲区相对于 swizzle 模式被偏移了多少次，请应用相应的偏移公式。

.. list-table:: 交错模式指针偏移公式和索引关系
   :widths: 25 45 30
   :header-rows: 1

   * - 交错模式
     - 偏移公式
     - 索引关系
   * - CU_TENSOR_MAP_SWIZZLE_128B
     - ``(reinterpret_cast<uintptr_t>(smem_ptr)/128)%8``
     - ``smem[y][x] <-> smem[y][((y+offset)%8)^x]``
   * - CU_TENSOR_MAP_SWIZZLE_64B
     - ``(reinterpret_cast<uintptr_t>(smem_ptr)/128)%4``
     - ``smem[y][x] <-> smem[y][((y+offset)%4)^x]``
   * - CU_TENSOR_MAP_SWIZZLE_32B
     - ``(reinterpret_cast<uintptr_t>(smem_ptr)/128)%2``
     - ``smem[y][x] <-> smem[y][((y+offset)%2)^x]``

在 图 53 中，该偏移表示初始行偏移，因此在 swizzle 索引计算中，它被加到行索引 ``y`` 上。以下代码片段展示了如何在 ``CU_TENSOR_MAP_SWIZZLE_128B`` 模式下访问 swizzle 后的共享内存。

.. code-block:: cuda

   data_t* smem_ptr = &smem[0][0];
   int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
   smem[y][((y+offset)%8)^x] = ...

**总结。** 下表汇总了计算能力 9 的各种 swizzle 模式的要求和属性。

.. list-table:: 计算能力 9 的不同交错模式的要求和属性
   :widths: 20 15 20 15 15 15
   :header-rows: 1

   * - 模式
     - 交错宽度
     - 共享框的内部维度
     - 重复周期
     - 共享内存对齐
     - Global 内存对齐
   * - CU_TENSOR_MAP_SWIZZLE_128B
     - 128 字节
     - <=128 字节
     - 1024 字节
     - 128 字节
     - 128 字节
   * - CU_TENSOR_MAP_SWIZZLE_64B
     - 64 字节
     - <=64 字节
     - 512 字节
     - 128 字节
     - 128 字节
   * - CU_TENSOR_MAP_SWIZZLE_32B
     - 32 字节
     - <=32 字节
     - 256 字节
     - 128 字节
     - 128 字节
   * - CU_TENSOR_MAP_SWIZZLE_NONE （默认）
     -
     -
     -
     - 128 字节
     - 16 字节

.. _using-stas:

4.12.3. 使用 STAS
-----------------

使用线程块集群的 CUDA 应用程序可能需要在集群内的线程块之间移动小数据元素。
`STAS 指令 <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async>`_ （CC 9.0+）支持从寄存器直接异步数据拷贝到分布式共享内存。
STAS 仅通过 libcu++ 库中提供的较低级别的 ``cuda::ptx::st_async`` API 公开。

**维度**。STAS 支持复制 4、8 或 16 字节。

**源和目标**。STAS 异步拷贝操作支持的唯一方向是从寄存器到分布式共享内存。目标指针需要根据复制的数据大小对齐到 4、8 或 16 字节。

**异步性**。使用 STAS 的数据传输是 :ref:`异步的 <asynchronous-execution-features>` ，并建模为 :ref:`异步线程操作 <async-thread-and-async-proxy>` 。
这允许发起线程在硬件异步复制数据的同时继续计算。
数据传输是否实际异步发生取决于硬件实现，未来可能会发生变化。
STAS 操作可用于发出完成信号的完成机制是共享内存屏障。

在下面的示例中，我们展示如何使用 STAS 在线程块集群内实现生产者 - 消费者模式。
该核函数创建一个环形通信流水线：8 个线程块排列成一个环，每个块同时：

- 为序列中的下一个块生产数据。
- 消费来自序列中上一个块的数据。

要实现此模式，每个线程块需要 2 个共享内存屏障：一个用于通知消费者块数据已拷贝到共享内存缓冲区（filled），另一个用于通知生产者块消费者上的缓冲区已准备好被填充（ready）。

使用 CUDA C++ ``cuda::ptx``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/barrier>
   #include <cuda/ptx>

   __global__ __cluster_dims__(8, 1, 1) void producer_consumer_kernel()
   {
       using namespace cooperative_groups;
       using namespace cuda::device;
       using namespace cuda::ptx;
       using barrier_t = cuda::barrier<cuda::thread_scope_block>;

       auto cluster = this_cluster();

       #pragma nv_diag_suppress static_var_with_dynamic_init
       __shared__ int buffer[BLOCK_SIZE];
       __shared__ barrier_t filled;
       __shared__ barrier_t ready;

       // Initialize shared memory barriers.
       if (threadIdx.x == 0) {
           init(&filled, 1);
           init(&ready, BLOCK_SIZE);
       }

       // Sync cluster to ensure remote barriers are initialized.
       cluster.sync();

       // Define my own and my neighbor's ranks.
       int rk = cluster.block_rank();
       int rk_next = (rk + 1) % 8;
       int rk_prev = (rk + 7) % 8;

       // Get addresses of remote buffer we are writing to and remote barriers of previous and next blocks.
       auto buffer_next = cluster.map_shared_rank(buffer, rk_next);
       auto bar_next = cluster.map_shared_rank(barrier_native_handle(filled), rk_next);
       auto bar_prev = cluster.map_shared_rank(barrier_native_handle(ready), rk_prev);

       int phase = 0;
       for (int it = 0; it < 1000; ++it) {

           // As producers, send data to our right neighbor.
           st_async(&buffer_next[threadIdx.x], rk, bar_next);

           if (threadIdx.x == 0) {
               // Thread 0 arrives on local barrier and indicates it expects to receive a certain number of bytes.
               mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared,
                                         barrier_native_handle(filled), sizeof(buffer));
           }

           // As consumers, wait on local barrier for data from left neighbor to arrive.
           while (!mbarrier_try_wait_parity(barrier_native_handle(filled), phase, 1000)) {}

           // At this point, the data has been copied to our local buffer.
           int r = buffer[threadIdx.x];

           // Use the data to do something.

           // As consumers, notify our left neighbor that we are done with the data.
           mbarrier_arrive(sem_release, scope_cluster, space_cluster, bar_prev);

           // As producers, wait on local barrier until the right neighbor is ready to receive new data.
           while (!mbarrier_try_wait_parity(barrier_native_handle(ready), phase, 1000)) {}
           phase ^= 1;
       }
   }

共享内存屏障由每个块的第一个线程初始化。屏障 filled 初始化为 1，屏障 ready 初始化为块中的线程数。
执行集群范围的同步，以确保在任何线程开始通信之前所有屏障都已初始化。
每个线程确定其邻居的 rank，并用它们映射远程共享内存屏障以及要写入数据的远程共享内存缓冲区。

在每次迭代中：

- 作为生产者，每个线程向其右邻居发送数据。
- 作为消费者，线程 0 在本地 filled 屏障上执行 arrive，并指示它期望接收一定数量的字节。
- 作为消费者，每个线程在本地 filled 屏障上等待来自左邻居的数据到达。
- 作为消费者，每个线程使用数据执行某些操作。
- 作为消费者，每个线程通知左邻居它已用完数据。
- 作为生产者，每个线程在本地 ready 屏障上等待，直到右邻居准备好接收新数据。

请注意，对于每个屏障，我们需要使用正确的 space。对于映射的远程屏障，我们需要使用 ``space_cluster`` space，而对于本地屏障，我们需要使用 ``space_shared`` space。
