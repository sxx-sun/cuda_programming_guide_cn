.. _stream-ordered-memory-allocation-details:

4.3. 流序内存分配器
======================

.. _soma-introduction:

4.3.1. 简介
-----------

使用 ``cudaMalloc`` 和 ``cudaFree`` 管理内存分配，会导致 GPU 对所有正在执行的 CUDA 流之间进行同步。
而流序内存分配器（stream-ordered memory allocator）允许应用程序将内存的分配与释放操作提交到CUDA 流中, 并与流中的其它任务（如 kernel 启动和异步拷贝）一起按序执行。
它可以利用流序语义复用已分配内存，从而提升应用程序的内存使用效率。
此外，该分配器还允许应用程序控制其内存缓存行为。
若配置合理的释放阈值，当应用希望更大内存占用时，缓存机制可减少开销高昂的操作系统调用。
该分配器还支持在不同进程之间轻松且安全地共享内存分配。

流序内存分配器的好处：

- 它减少了用户对自定义内存管理的需求，同时也让有需求的应用，能够轻松构建高性能的自定义内存管理方案。
- 使多个库能够共享由驱动程序管理的公共内存池，减少过度的内存消耗。
- 允许驱动程序基于流序内存分配器及其他流管理接口，执行各类优化操作。

.. note::

   自 CUDA 11.3 起，Nsight Compute 以及新一代 CUDA 调试器均可识别该分配器。

.. _soma-memory-management:

4.3.2. 内存管理
---------------

``cudaMallocAsync`` 和 ``cudaFreeAsync`` 是实现流排序内存管理的核心 API。
这两个 API 都接受流参数，以此来定义内存何时可用以及何时停止使用。
它们将内存操作绑定到特定的 CUDA 流，从而使内存分配和释放能够在不阻塞主机或其他流的情况下异步进行。
通过避免 ``cudaMalloc`` 和 ``cudaFree`` 带来的高昂同步的开销，提高应用程序性能。

这些 API 还可以结合内存池进行更深层次的性能优化。
内存池通过管理和复用大块内存，来实现更高效的内存分配与释放。
这有助于降低系统开销并防止内存碎片化，从而显著提升频繁内存分配场景下的性能表现。

.. _soma-allocating-memory:

4.3.2.1. 异步分配内存
~~~~~~~~~~~~~~~~~~~~~

``cudaMallocAsync`` 函数触发 GPU 上的异步内存分配，并将其提交到特定的 CUDA 流上。
它允许内存分配在不阻碍主机或其他流执行的情况下进行，节省高昂的同步开销。

.. note::

   ``cudaMallocAsync`` 在决定内存分配位置时，会忽略当前的设备或上下文。相反，它会根据指定的内存池或提供的流来确定合适的设备。

下方代码示例展示了一种最基本的使用模式：在同一个流内完成内存分配、使用，最后释放。

.. code-block:: cuda
   :caption: 基本的流有序内存分配模式

   void *ptr;
   size_t size = 512;
   cudaMallocAsync(&ptr, size, cudaStreamPerThread);
   // do work using the allocation
   kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
   // An asynchronous free can be specified without synchronizing the cpu and GPU
   cudaFreeAsync(ptr, cudaStreamPerThread);

.. note::

   当从与执行内存分配的流以外的其他的流访问时，用户必须保证访问发生在分配操作之后，否则行为是未定义的。

.. _soma-freeing-memory:

4.3.2.2. 异步释放内存
~~~~~~~~~~~~~~~~~~~~~

``cudaFreeAsync()`` 以流排序的方式异步释放设备内存，这意味着内存释放操作会被分配到特定的 CUDA 流中执行，且不会阻塞主机或其他流。

用户必须确保内存释放操作发生在内存分配操作以及所有访问操作之后。如果在释放操作开始之后，该内存仍被任何程序使用，将导致未定义行为

应使用事件或流同步操作，以确保在释放操作开始之前，其他流对该内存的所有访问均已彻底完成，做法如下。

.. code-block:: cuda
   :caption: 使用事件同步释放内存

   cudaMallocAsync(&ptr, size, stream1);
   cudaEventRecord(event1, stream1);
   //stream2 must wait for the allocation to be ready before accessing
   cudaStreamWaitEvent(stream2, event1);
   kernel<<<..., stream2>>>(ptr, ...);
   cudaEventRecord(event2, stream2);
   // stream3 must wait for stream2 to finish accessing the allocation before
   // freeing the allocation
   cudaStreamWaitEvent(stream3, event2);
   cudaFreeAsync(ptr, stream3);

使用 ``cudaMalloc()`` 分配的内存可以用 ``cudaFreeAsync()`` 释放。如上所述，所有对内存的访问必须在释放操作开始之前完成。

.. code-block:: cuda

   cudaMalloc(&ptr, size);
   kernel<<<..., stream>>>(ptr, ...);
   cudaFreeAsync(ptr, stream);

同样地，使用 ``cudaMallocAsync`` 分配的内存也可以通过 ``cudaFree()`` 释放。
当通过 ``cudaFree()`` 释放时，驱动程序会假定该内存的所有访问都已完成，因此不会执行任何额外的同步操作。
用户可以使用 ``cudaStreamQuery`` 、 ``cudaStreamSynchronize`` 、 ``cudaEventQuery`` 、 ``cudaEventSynchronize`` 或 ``cudaDeviceSynchronize`` 等函数，
来确保相应的异步任务已全部完成，并且 GPU 不会再尝试访问该内存。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size,stream);
   kernel<<<..., stream>>>(ptr, ...);
   // synchronize is needed to avoid prematurely freeing the memory
   cudaStreamSynchronize(stream);
   cudaFree(ptr);

.. _soma-memory-pools:

4.3.3. 内存池
-------------

内存池（Memory Pools）封装了虚拟地址和物理内存资源，这些资源会根据内存池的属性和特征进行管理。
内存池最核心的要素，在于它所管理的内存类型及其物理位置。

``cudaMallocAsync`` 从内存池中的申请资源。
如果未显式指定内存池， ``cudaMallocAsync`` 将使用传入流所在设备的当前内存池。
可通过 ``cudaDeviceSetMempool`` 设置设备的当前内存池，并通过 ``cudaDeviceGetMempool`` 查询当前内存池。
每个设备都有一个默认内存池，用户未显式指定的情况下，默认内存池将处于激活状态。

``cudaMallocFromPoolAsync`` 以及
``cudaMallocAsync`` 的 `C++ 重载函数 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb>`_ ，
允许用户从指定内存池申请内存。
通过接口 ``cudaMemPoolSetAttribute`` 和 ``cudaMemPoolGetAttribute`` 可设置和查询内存池的各项属性。

.. note::

   设备当前的内存池仅局限于该设备本身。因此，如果不显式指定内存池，所申请的内存必定位于该流所在的设备上。


.. _soma-default-implicit-pools:

4.3.3.1. 默认（隐式）内存池
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

调用 ``cudaDeviceGetDefaultMempool`` 可获取设备的默认内存池句柄。
从设备默认内存池分配出的内存属于不可迁移的设备显存，常驻于当前设备，且该设备可随时访问这块内存。
可通过 ``cudaMemPoolSetAccess`` 修改默认内存池的跨设备访问权限，并使用 ``cudaMemPoolGetAccess`` 查询权限配置。
默认内存池无需手动创建，因此也被称作隐式内存池。设备默认内存池不支持进程间共享（IPC）。

.. _soma-explicit-pools:

4.3.3.2. 显式内存池
~~~~~~~~~~~~~~~~~~~

``cudaMemPoolCreate`` 创建一个显式内存池。相对默认内存池，它使应用程序能够为内存分配自定义各类属性，提供更强大的功能。
这些属性包括：IPC（进程间通信）能力、最大内存池大小，以及将分配驻留在特定 CPU NUMA 节点上（需要平台支持）等功能。

.. code-block:: cuda

   // create a pool similar to the implicit pool on device 0
   int device = 0;
   cudaMemPoolProps poolProps = { };
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.location.id = device;
   poolProps.location.type = cudaMemLocationTypeDevice;

   cudaMemPoolCreate(&memPool, &poolProps);

下面的代码片段，展示如何在有效的 CPU NUMA 节点上创建一个支持 IPC（进程间通信）的内存池。

.. code-block:: cuda

   // create a pool resident on a CPU NUMA node
   // that is capable of IPC sharing (via a file descriptor).

   int cpu_numa_id = 0;
   cudaMemPoolProps poolProps = { };
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.location.id = cpu_numa_id;
   poolProps.location.type = cudaMemLocationTypeHostNuma;
   poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

   cudaMemPoolCreate(&ipcMemPool, &poolProps);

.. _soma-device-accessibility:

4.3.3.3. 多 GPU 支持的设备可访问性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

内存池分配内存的可访问性不受 ``cudaDeviceEnablePeerAccess`` 或 ``cuCtxEnablePeerAccess`` 控制。
``cudaMemPoolSetAccess`` 接口用于修改内存池分配的可访问性。
默认情况下，仅内存所在设备能够访问，且此访问权限无法被撤销。
若要允许其他设备访问此内存池，访问设备必须与内存池所在设备具备对等访问能力，可通过 ``cudaDeviceCanAccessPeer`` 接口校验该能力。
若未提前校验对等访问能力， ``cudaMemPoolSetAccess`` 可能返回 ``cudaErrorInvalidDevice`` 。
然而，如果设备不具备对等能力，且尚未从池中进行任何分配， ``cudaMemPoolSetAccess`` 调用也可能成功，但是当从内存池中申请内存时将失败。

值得注意的是， ``cudaMemPoolSetAccess`` 会影响内存池中的所有分配，而不仅仅是未来的分配。
同样， ``cudaMemPoolGetAccess`` 获取的可访问性适用于该池中的所有分配，而不仅限于未来的分配。
此外，不建议频繁更改内存池对特定 GPU 的可访问性设置。
也就是说，一旦某个内存池对特定 GPU 开放了访问权限，在该内存池的整个生命周期内，都应保持该 GPU 的访问权限。

.. code-block:: cuda

   // snippet showing usage of cudaMemPoolSetAccess:
   cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
                  int accessingDevice) {
         cudaMemAccessDesc accessDesc = {};
         accessDesc.location.type = cudaMemLocationTypeDevice;
         accessDesc.location.id = accessingDevice;
         accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

         int canAccess = 0;
         cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
                  residentDevice);
         if (error != cudaSuccess) {
            return error;
         } else if (canAccess == 0) {
            return cudaErrorPeerAccessUnsupported;
         }

         // Make the address accessible
         return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
   }

.. _soma-enabling-memory-pools-for-ipc:

4.3.3.4. 为 IPC 启用内存池
~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以为内存池启用进程间通信（IPC）功能，以实现进程间轻松、高效且安全地共享 GPU 内存。
CUDA 的 IPC 内存池提供与 :ref:`virtual-memory-management` 相同的安全优势。

使用内存池在进程间共享内存分为两个步骤：首先，进程需要共享对内存池的访问权限；其次，共享该内存池的特定分配的访问权限。
第一步用于建立并强制执行安全策略；第二步则用于协调每个进程所使用的虚拟地址，以及明确映射在导入进程中的生效事件。

.. _soma-creating-sharing-ipc-pools:

4.3.3.4.1. 创建和共享 IPC 内存池
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

进程间共享内存池的访问权限，涉及以下 3 个步骤：

1. 创建内存池，并通过 ``cudaMemPoolExportToShareableHandle()`` 获取内存池的操作系统原生句柄。
2. 利用操作系统的原生 IPC 机制将该句柄传递给导入进程。
3. 在导入进程，调用 ``cudaMemPoolImportFromShareableHandle()`` 来创建导入的内存池。

需要注意的是，在第一步创建内存池时，必须在内存池属性参数中指定需要的句柄类型。

请参考 `示例 <https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC>`_ 了解在进程之间传输操作系统原生句柄的 IPC 机制。
部分代码如下：


.. code-block:: cuda
   :caption: 在导出进程中创建和共享 IPC 内存池

   // in exporting process
   // create an exportable IPC capable pool on device 0
   cudaMemPoolProps poolProps = { };
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.location.id = 0;
   poolProps.location.type = cudaMemLocationTypeDevice;

   // Setting handleTypes to a non zero value will make the pool exportable (IPC capable)
   poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

   cudaMemPoolCreate(&memPool, &poolProps);

   // FD based handles are integer types
   int fdHandle = 0;


   // Retrieve an OS native handle to the pool.
   // Note that a pointer to the handle memory is passed in here.
   cudaMemPoolExportToShareableHandle(&fdHandle,
                memPool,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0);

   // The handle must be sent to the importing process with the appropriate
   // OS-specific APIs.

.. code-block:: cuda
   :caption: 在导入进程中导入 IPC 内存池

   // in importing process
   int fdHandle;
   // The handle needs to be retrieved from the exporting process with the
   // appropriate OS-specific APIs.
   // Create an imported pool from the shareable handle.
   // Note that the handle is passed by value here.
   cudaMemPoolImportFromShareableHandle(&importedMemPool,
            (void*)fdHandle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0);

.. _soma-set-access-importing-process:

4.3.3.4.2. 在导入进程中设置访问权限
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

导入的内存池不会继承由导出进程设置的任何访问权限。内存池导入后，仅其驻留设备可访问。
如果导入进程需要从其他 GPU 访问，则必须使用 ``cudaMemPoolSetAccess`` 显式启用访问权限。

如果内存池所属设备对导入进程不可见，用户必须使用 ``cudaMemPoolSetAccess`` 增加工作 GPU 对该内存池的访问权限。（参见 :ref:`soma-device-accessibility` ）

.. _soma-creating-sharing-allocations:

4.3.3.4.3. 从导出池创建和共享分配
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一旦内存池被共享，导出进程中使用 ``cudaMallocAsync()`` 从该池分配的内存，就可以与已导入该池的其他进程共享。
由于内存池的安全策略是在池级别建立和验证的，操作系统无需进行额外的工作来为特定的池分配提供安全保障。
换句话说，从该内存池中分配的用于进程间内存共享的不透明句柄 ``cudaMemPoolPtrExportData`` ，可以通过 IPC 机制发送给导入进程（使用）。

虽然内存分配的导出和导入过程无需与分配操作所在的流进行任何同步，但导入进程在访问该内存时，必须遵循与导出进程相同的规则。
具体而言，必须在分配操作在流中执行完毕后，才能访问该内存分配。
以下两段代码示例展示了如何使用 ``cudaMemPoolExportPointer()`` 和 ``cudaMemPoolImportPointer()`` 来共享内存分配，并利用 IPC 事件来确保导入进程不会在内存分配准备就绪之前对其进行访问。

.. code-block:: cuda
   :caption: 在导出进程中准备分配

   // preparing an allocation in the exporting process
   cudaMemPoolPtrExportData exportData;
   cudaEvent_t readyIpcEvent;
   cudaIpcEventHandle_t readyIpcEventHandle;

   // ipc event for coordinating between processes
   // cudaEventInterprocess flag makes the event an ipc event
   // cudaEventDisableTiming  is set for performance reasons

   cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

   // allocate from the exporting mem pool
   cudaMallocAsync(&ptr, size,exportMemPool, stream);

   // event for sharing when the allocation is ready.
   cudaEventRecord(readyIpcEvent, stream);
   cudaMemPoolExportPointer(&exportData, ptr);
   cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

   // Share IPC event and pointer export data with the importing process using
   //  any mechanism. Here we copy the data into shared memory
   shmem->ptrData = exportData;
   shmem->readyIpcEventHandle = readyIpcEventHandle;
   // signal consumers data is ready

.. code-block:: cuda
   :caption: 导入分配

   // Importing an allocation
   cudaMemPoolPtrExportData *importData = &shmem->prtData;
   cudaEvent_t readyIpcEvent;
   cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

   // Need to retrieve the ipc event handle and the export data from the
   // exporting process using any mechanism.  Here we are using shmem and just
   // need synchronization to make sure the shared memory is filled in.

   cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

   // import the allocation. The operation does not block on the allocation being ready.
   cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

   // Wait for the prior stream operations in the allocating stream to complete before
   // using the allocation in the importing process.
   cudaStreamWaitEvent(stream, readyIpcEvent);
   kernel<<<..., stream>>>(ptr, ...);


在释放内存分配时，导入进程必须先于导出进程完成释放。
以下代码片段演示了如何使用 CUDA IPC 事件，为两个进程中的 ``cudaFreeAsync`` 操作提供所需的同步机制。
导入进程对内存分配的访问，显然会受到其自身释放操作的限制。
值得注意的是，在这两个进程中都可以使用 ``cudaFree`` 来释放该内存分配，并且除了 CUDA IPC 事件外，也可以使用其他流同步接口来替代。

.. code-block:: cuda

   // The free must happen in importing process before the exporting process
   kernel<<<..., stream>>>(ptr, ...);

   // Last access in importing process
   cudaFreeAsync(ptr, stream);

   // Access not allowed in the importing process after the free
   cudaIpcEventRecord(finishedIpcEvent, stream);

.. code-block:: cuda

   // Exporting process
   // The exporting process needs to coordinate its free with the stream order
   // of the importing process’s free.
   cudaStreamWaitEvent(stream, finishedIpcEvent);
   kernel<<<..., stream>>>(ptrInExportingProcess, ...);

   // The free in the importing process doesn’t stop the exporting process
   // from using the allocation.
   cudFreeAsync(ptrInExportingProcess,stream);

.. _soma-ipc-export-pool-limitations:

4.3.3.4.4. IPC 导出池限制
^^^^^^^^^^^^^^^^^^^^^^^^^^

IPC 内存池目前不支持将物理内存块释放回操作系统。
因此， ``cudaMemPoolTrimTo`` 对 IPC 内存池无效，且 ``cudaMemPoolAttrReleaseThreshold`` （释放阈值）属性也会被直接忽略。
该行为由底层驱动控制，而非 CUDA 运行时，可能会在未来的驱动更新中发生改变。

.. _soma-ipc-import-pool-limitations:

4.3.3.4.5. IPC 导入池限制
^^^^^^^^^^^^^^^^^^^^^^^^^^

不允许从导入的内存池中申请内存；例如，不能将导入的内存池设置为当前内存池，也不能在 ``cudaMallocFromPoolAsync`` 接口中使用它们。
因此，内存分配复用策略相关的属性对于这类内存池是没有意义的。

IPC 导入内存池与 IPC 导出内存池一样，目前均不支持将物理内存块释放回操作系统。

对导入进程而言，查询到资源使用量统计属性，仅反映已导入到该进程中的内存分配及其关联的物理内存。

.. _soma-best-practices:

4.3.4. 最佳实践和调优
----------------------

.. _soma-query-support:

4.3.4.1. 查询支持
~~~~~~~~~~~~~~~~~

应用程序可以通过调用 ``cudaDeviceGetAttribute()`` 函数查询设备属性 ``cudaDevAttrMemoryPoolsSupported`` ，来判断设备是否支持流排序内存分配器。
详见 `开发者博客 <https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/>`_。

可以通过查询属性 ``cudaDevAttrMemoryPoolSupportedHandleTypes`` 来判断设备是否支持 IPC 内存池。
该属性是在 CUDA 11.3 中引入的，如果使用旧版驱动查询此属性，将返回 ``cudaErrorInvalidValue`` 错误。

.. code-block:: cuda

   int driverVersion = 0;
   int deviceSupportsMemoryPools = 0;
   int poolSupportedHandleTypes = 0;
   cudaDriverGetVersion(&driverVersion);
   if (driverVersion >= 11020) {
      cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                              cudaDevAttrMemoryPoolsSupported, device);
   }
   if (deviceSupportsMemoryPools != 0) {
      // `device` supports the Stream-Ordered Memory Allocator
   }

   if (driverVersion >= 11030) {
      cudaDeviceGetAttribute(&poolSupportedHandleTypes,
               cudaDevAttrMemoryPoolSupportedHandleTypes, device);
   }
   if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
      // Pools on the specified device can be created with posix file descriptor-based IPC
   }

在查询之前先检查驱动版本，可以避免在旧版驱动上触发 ``cudaErrorInvalidValue`` 错误。
如果未提前检查版本，可在出错之后调用 ``cudaGetLastError`` 清除该错误。

.. _soma-physical-page-caching:

4.3.4.2. 物理页缓存行为
~~~~~~~~~~~~~~~~~~~~~~~

默认情况下，分配器尝试最小化池拥有的物理内存。
为了减少分配和释放物理内存时产生的操作系统调用开销，应用程序必须为每个内存池配置一个内存占用阈值。
这可以通过设置释放阈值属性（ ``cudaMemPoolAttrReleaseThreshold`` ）来实现。

释放阈值指内存池保留的内存量（以字节为单位），超过该数值后分配器会尝试将内存归还操作系统。
当内存池持有的内存总量超出释放阈值时，在下一次执行流同步、事件同步或设备同步操作时，分配器会主动向操作系统释放闲置内存。
若将释放阈值设置为 ``UINT64_MAX`` ，则关闭内存池主动缩容操作。

.. code-block:: cuda

   cuuint64_t setVal = UINT64_MAX;
   cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

如果应用程序将 ``cudaMemPoolAttrReleaseThreshold`` 设置得足够高，从而实际上禁用了内存池的自动缩减功能，那么它可能希望在需要时主动缩减内存池的物理内存占用量。
``cudaMemPoolTrimTo`` 允许应用程序执行此操作。
在主动缩减内存池时， ``minBytesToKeep`` 参数允许应用程序保留指定大小的内存，例如，保留其在后续执行阶段中预计所需的内存量。

.. code-block:: cuda

   Cuuint64_t setVal = UINT64_MAX;
   cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

   // application phase needing a lot of memory from the stream-ordered allocator
   for (i=0; i<10; i++) {
      for (j=0; j<10; j++) {
         cudaMallocAsync(&ptrs[j],size[j], stream);
      }
      kernel<<<...,stream>>>(ptrs,...);
      for (j=0; j<10; j++) {
         cudaFreeAsync(ptrs[j], stream);
      }
   }

   // Process does not need as much memory for the next phase.
   // Synchronize so that the trim operation will know that the allocations are no
   // longer in use.
   cudaStreamSynchronize(stream);
   cudaMemPoolTrimTo(mempool, 0);

   // Some other process/allocation mechanism can now use the physical memory
   // released by the trimming operation.


.. _soma-resource-usage-statistics:

4.3.4.3. 资源使用统计
~~~~~~~~~~~~~~~~~~~~~

通过查询 ``cudaMemPoolAttrReservedMemCurrent`` 属性，可以获取内存池当前消耗的物理 GPU 显存总量。
而查询 ``cudaMemPoolAttrUsedMemCurrent`` 属性，则返回从该内存池中分配出去且当前不可用于复用的内存总大小。

``cudaMemPoolAttr*MemHigh`` 属性是用于记录高水位线的，它们记录了自上次重置以来，相应的 ``cudaMemPoolAttr*MemCurrent`` 属性所达到的最大值。
开发者可以使用 ``cudaMemPoolSetAttribute`` 将这些高水位线重置为当前值。

.. code-block:: cuda
   :caption: 获取使用统计信息的示例辅助函数

   // sample helper functions for getting the usage statistics in bulk
   struct usageStatistics {
     cuuint64_t reserved;
     cuuint64_t reservedHigh;
     cuuint64_t used;
     cuuint64_t usedHigh;
   };

   void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
   {
     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
   }


   // resetting the watermarks will make them take on the current value.
   void resetStatistics(cudaMemoryPool_t memPool)
   {
     cuuint64_t value = 0;
     cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
     cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
   }

.. _soma-memory-reuse-policies:

4.3.4.4. 内存重用策略
~~~~~~~~~~~~~~~~~~~~~

为了满足内存分配请求，驱动程序会尝试优先复用之前通过 ``cudaFreeAsync()`` 释放的内存，然后才会尝试从操作系统中分配更多内存。
例如，在某个流中释放的内存，可以立即被同一流中的后续分配请求所复用。
当某个流与 CPU 进行同步时，该流中之前释放的内存便可供任意流中的分配请求复用。
这些内存重用策略同时适用于默认内存池和显式创建的内存池。

流序分配器提供了几种分配策略：
``cudaMemPoolReuseFollowEventDependencies`` 、 ``cudaMemPoolReuseAllowOpportunistic`` 和 ``cudaMemPoolReuseAllowInternalDependencies`` ，
具体细节如下所述。
开发者可以通过 ``cudaMemPoolSetAttribute`` 来启用或禁用这些策略。
需要注意的是，升级 CUDA 驱动程序可能会更改、增强、补充和/或重新排列这些重用策略的枚举值。

.. _soma-cudaMemPoolReuseFollowEventDependencies:

4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在分配更多物理 GPU 显存之前，分配器会检查由 CUDA 事件建立的依赖关系信息，并尝试从其他流释放的内存里进行分配。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, originalStream);
   kernel<<<..., originalStream>>>(ptr, ...);
   cudaFreeAsync(ptr, originalStream);
   cudaEventRecord(event,originalStream);

   // waiting on the event that captures the free in another stream
   // allows the allocator to reuse the memory to satisfy
   // a new allocation request in the other stream when
   // cudaMemPoolReuseFollowEventDependencies is enabled.
   cudaStreamWaitEvent(otherStream, event);
   cudaMallocAsync(&ptr2, size, otherStream);

.. _soma-cudaMemPoolReuseAllowOpportunistic:

4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当 ``cudaMemPoolReuseAllowOpportunistic`` 策略生效时，分配器会检查已释放的内存，确认是否满足释放操作的流序语义（例如，流执行已经过释放节点）。
当禁用此策略时，分配器依然会复用流同步时变为可用状态的内存。
此外，禁用此策略并不影响 ``cudaMemPoolReuseFollowEventDependencies`` 的生效。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, originalStream);
   kernel<<<..., originalStream>>>(ptr, ...);
   cudaFreeAsync(ptr, originalStream);


   // after some time, the kernel finishes running
   wait(10);

   // When cudaMemPoolReuseAllowOpportunistic is enabled this allocation request
   // can be fulfilled with the prior allocation based on the progress of originalStream.
   cudaMallocAsync(&ptr2, size, otherStream);

.. _soma-cudaMemPoolReuseAllowInternalDependencies:

4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果当前无法立即从操作系统分配并映射更多的物理显存，驱动程序会去寻找那些可用性依赖于其他流的处理进度的内存（其他流在使用的，未来会释放的内存）。
如果找到了这样的内存，驱动程序会将依赖关系插入到发起分配的流中，等待在未来复用这块内存。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, originalStream);
   kernel<<<..., originalStream>>>(ptr, ...);
   cudaFreeAsync(ptr, originalStream);

   // When cudaMemPoolReuseAllowInternalDependencies is enabled
   // and the driver fails to allocate more physical memory, the driver may
   // effectively perform a cudaStreamWaitEvent in the allocating stream
   // to make sure that future work in ‘otherStream’ happens after the work
   // in the original stream that would be allowed to access the original allocation.
   cudaMallocAsync(&ptr2, size, otherStream);

.. _soma-disabling-reuse-policies:

4.3.4.4.4. 禁用重用策略
^^^^^^^^^^^^^^^^^^^^^^^^

尽管重用策略提升了内存的复用率，但用户可能希望禁用它们。
允许机会主义重用（ ``cudaMemPoolReuseAllowOpportunistic`` ）会引入运行间的差异，因为分配模式会随 CPU 与 GPU 执行的交错情况而改变。
而内部依赖插入（ ``cudaMemPoolReuseAllowInternalDependencies`` ）可能会以意料之外且潜在非确定性的方式使工作串行化，尤其是在用户更倾向于在分配失败时显式地对事件或流进行同步的情况下。

.. _soma-synchronization-api-actions:

4.3.4.5. 同步 API 操作
~~~~~~~~~~~~~~~~~~~~~~

由于分配器是 CUDA 驱动程序的一部分，其带来的一项优化便是与同步 API 进行了深度集成。
当用户请求 CUDA 驱动程序执行同步操作时，驱动程序会等待异步任务完成。
在（同步操作）返回之前，驱动程序会确定哪些内存释放操作已因该同步操作而保证完成。
无论这些内存属于哪个流，也无论分配策略是否被禁用，这些已释放的内存都会被重新标记为可分配状态。
此外，驱动程序还会在此时检查 ``cudaMemPoolAttrReleaseThreshold`` 属性，并释放所有能够释放的多余物理显存。

.. _soma-addendums:

4.3.5. 附录
-----------

.. _soma-cudaMemcpyAsync-sensitivity:

4.3.5.1. cudaMemcpyAsync 当前上下文/设备敏感性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在当前的 CUDA 驱动程序中，任何涉及由 ``cudaMallocAsync`` 分配的内存的异步内存拷贝操作，都必须使用指定流的上下文作为调用线程的当前上下文。
不过， ``cudaMemcpyPeerAsync`` 不受此限制，因为该 API 会直接使用其参数中指定的设备主上下文，而不是当前上下文。

.. _soma-cudaPointerGetAttributes:

4.3.5.2. cudaPointerGetAttributes 查询接口
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在对某个内存分配调用 ``cudaFreeAsync`` 之后，如果再次对其调用 ``cudaPointerGetAttributes`` ，将导致未定义行为。
无论该分配是否仍可从指定的流中访问，其行为都是未定义的。

.. _soma-cudaGraphAddMemsetNode:

4.3.5.3. cudaGraphAddMemsetNode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaGraphAddMemsetNode`` 不支持使用流序分配器分配的内存。
但针对这类内存的 ``memsets`` 操作可以被流捕获。

.. _soma-pointer-attributes:

4.3.5.4. 指针属性
~~~~~~~~~~~~~~~~~

可以使用 ``cudaPointerGetAttributes`` 查询流序分配内存的属性。
由于流序分配内存不与 CUDA 上下文绑定，查询 ``CU_POINTER_ATTRIBUTE_CONTEXT`` 属性时不会报错，但会在 ``*data`` 中返回空指针。
可通过 ``CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`` 属性获取内存分配所在设备编号，这在选择上下文调用 ``cudaMemcpyPeerAsync`` 执行设备间对等拷贝时十分有用。
``CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`` 属性自 CUDA 11.3 版本新增，适用于调试场景，也可在执行进程间通信前，确认一段内存归属哪个内存池。

.. _soma-cpu-virtual-memory:

4.3.5.5. CPU 虚拟内存
~~~~~~~~~~~~~~~~~~~~~

使用 CUDA 流序内存分配相关接口时，请勿通过 `ulimit -v` 设置 VRAM 限制，因为不支持此操作。