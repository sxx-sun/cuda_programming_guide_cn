.. _locality-domains:

4.8. 局部性域（Locality Domains）
=================================

**局部性域（locality domain）** 是 GPU 中包含流式多处理器（SM）和设备内存的一个部分。
应用程序可以在某个局部性域中分配设备内存，并使用同一局部性域中的 SM 资源创建绿色上下文（green context）。
将计算与其访问的内存在位置上就近放置，可以在拥有多个局部性域的设备上提升性能。
计算局部化 API 在 :ref:`局部化计算资源 <localizing-compute-resources>` 中描述，内存局部化在 :ref:`分配局部化设备内存 <allocating-localized-device-memory>` 中描述。

CUDA 使用从零开始的 **局部性域 ID（locality domain ID）** 来标识设备的各个局部性域。
局部性域 ID 只有与其设备 ID 结合在一起时才有意义。
局部化 SM 资源和内存时使用相同的 ID，使应用程序能够为两者请求匹配的放置位置。


.. _discovering-locality-domains:

4.8.1. 发现局部性域
-------------------

使用 ``cudaDeviceGetAttribute`` 并传入以下 ``cudaDeviceAttr`` 值：

- ``cudaDevAttrLocalityDomainCount`` 返回设备上的局部性域数量。
- ``cudaDevAttrLocalityDomainMultiprocessorCount`` 返回每个局部性域中的 SM 数量。

应用程序应当查询这些属性，而不是假定某种特定拓扑。
设备上可用的局部性域数量受系统配置影响。
同一设备类型在不同场景下可能呈现不同数量的局部性域（ :ref:`第 4.8.5.3 节 <compatibility-notes>` ）。
所有设备（包括只有单个局部性域可用的设备）都可以使用局部性域 API。
在只有单个局部性域的设备上，这些 API 的效果等同于以整个设备为目标。
使用设备属性以编程方式查询设备的局部性域属性：

.. code-block:: cpp

   int localityDomainCount = 0;
   int smsPerLocalityDomain = 0;

   cudaDeviceGetAttribute(&localityDomainCount, cudaDevAttrLocalityDomainCount, device);
   cudaDeviceGetAttribute(&smsPerLocalityDomain, cudaDevAttrLocalityDomainMultiprocessorCount, device);


.. _locality-domains-compute:
.. _localizing-compute-resources:

4.8.2. 局部化计算资源
---------------------

要将 SM 资源局部化到特定的局部性域，请使用 CUDA 绿色上下文 API。
SM 的放置通过 :ref:`第 4.7 节 <green-contexts-details>` 中描述的绿色上下文资源 API 来表达。
构造局部化 SM 分区的主要步骤如下：

1. 使用 ``cudaDeviceGetDevResource`` 获取设备的 ``cudaDevResourceTypeSm`` 资源。
2. 使用 ``cudaDevSmResourceGroupParams`` 描述一个局部化 SM 组（ :ref:`详情 <locality-domains-localized-sm-group>` ）。
3. 将组参数传递给 ``cudaDevSmResourceSplit`` 。
4. 使用 ``cudaDevResourceGenerateDesc`` 生成资源描述符，使用 ``cudaGreenCtxCreate`` 创建绿色上下文，最后通过 ``cudaExecutionCtxStreamCreate`` 创建流。
5. 在该流上启动 kernel。

典型的绿色上下文创建流程在 :ref:`第 4.7.2 节 <green-contexts-ease-of-use>` 中描述。
局部化 SM 资源所需的额外改动如下所示：

.. _locality-domains-localized-sm-group:

要描述一个局部化 SM 组，需在 ``flags`` 中设置 ``cudaDevSmResourceGroupLocalityDomainId`` ，并在 ``localityDomainId`` 中设置该域的 ID。
有效 ID 介于 ``0`` 和 ``cudaDevAttrLocalityDomainCount - 1`` 之间。

.. code-block:: cpp

   cudaDevResource smResource;
   cudaDeviceGetDevResource(device, &smResource, cudaDevResourceTypeSm);

   {
       cudaDevSmResourceGroupParams params = {};
       // all fields in params must be zero-initialized
       params.flags = cudaDevSmResourceGroupLocalityDomainId;
       params.localityDomainId = localityDomainId;

       cudaDevResource result;
       cudaDevSmResourceSplit(&result, 1, &smResource, nullptr, 0, &params);
   }

.. _discovery-mode:

4.8.2.1. 发现模式
^^^^^^^^^^^^^^^^^

当满足一组约束的 SM 数量无法预先知晓时，请使用 ``cudaDevSmResourceSplit`` 的发现模式（discovery mode）。
将 ``smCount`` 设置为零即可启用发现模式。
API 会将发现到的数量写入 ``params.smCount`` 。
对 ``result`` 传入 ``nullptr`` 会执行一次试运行（dry run），只发现数量而不创建设备资源：

.. code-block:: cpp

   cudaDevSmResourceGroupParams params {};
   params.flags = cudaDevSmResourceGroupLocalityDomainId;
   params.localityDomainId = localityDomainId;
   params.smCount = 0;

   cudaDevSmResourceSplit(nullptr, 1, &smResource, nullptr, 0, &params);
   unsigned int availableSms = params.smCount;

如果 ``result`` 非空，同一次调用既会发现数量，也会返回包含相应数量 SM 的资源。


.. _supplemental-sms:

4.8.2.2. 补充 SM
^^^^^^^^^^^^^^^^

设备可以在 ``cudaDevAttrLocalityDomainMultiprocessorCount`` 所包含的 SM 之外，将额外的 *补充 SM（supplemental SM）* 关联到某个局部性域。
局部性域的补充 SM 提供与稳定 SM 相同的内存就近访问收益，但其数量可能因局部性域和设备而异。
与某个局部性域关联的实际 SM 集合，是其稳定 SM 与补充 SM 的并集。

要发现与某个局部性域关联的稳定 SM 和补充 SM 的总数，请在发现约束中加入 ``cudaDevSmResourceGroupLocalityDomainSupplementalSms`` ：

.. code-block:: cpp

   cudaDevSmResourceGroupParams supplementalParams {};
   supplementalParams.flags =
       cudaDevSmResourceGroupLocalityDomainId |
       cudaDevSmResourceGroupLocalityDomainSupplementalSms;
   supplementalParams.localityDomainId = localityDomainId;
   supplementalParams.smCount = 0;

   cudaDevSmResourceSplit(nullptr, 1, &smResource, nullptr, 0,
                          &supplementalParams);
   unsigned int totalLocalizedSms = supplementalParams.smCount;
   unsigned int supplementalSms = totalLocalizedSms - smsPerLocalityDomain;


.. _sm-split-constraints:

4.8.2.3. SM 切分约束
^^^^^^^^^^^^^^^^^^^^

可以在 ``params`` 中指定多个约束，与局部性域请求组合使用。例如：

将 ``smCount`` 设为正数以请求相应数量的 SM，或设为 ``0`` 以使用上文描述的发现模式。
在不包含补充 SM 或回填的情况下，大于 ``cudaDevAttrLocalityDomainMultiprocessorCount`` 的请求将会失败。
要包含补充 SM，请添加 ``cudaDevSmResourceGroupLocalityDomainSupplementalSms`` 标志。

当 ``coscheduledSmCount`` 为 ``0`` 时，API 不会按协同调度能力限制 SM，此时可用的局部化 SM 数量达到最大。
分区的簇能力之后可以通过 ``cudaOccupancyMaxPotentialClusterSize`` 动态查询。
当 ``coscheduledSmCount`` 非零时，API 将只选择满足所请求 ``coscheduledSmCount`` 的局部化 SM。

要从输入资源中 *任意* 局部性域选择 SM，请在 ``flags`` 中保留 ``cudaDevSmResourceGroupLocalityDomainId`` 的同时，
将 ``localityDomainId`` 设置为 ``cudaDevSmResourceGroupLocalityDomainIdAny`` 。
此值允许切分从所有可用局部性域的并集中选取，同时排除补充 SM。

.. code-block:: cpp

   group.flags = cudaDevSmResourceGroupLocalityDomainId;
   group.localityDomainId = cudaDevSmResourceGroupLocalityDomainIdAny;

未在 ``result`` 中返回的任何未使用的 SM，会通过 API 的 ``remainder`` 参数返回。


.. _backfill:

4.8.2.4. 回填
^^^^^^^^^^^^^

``cudaDevSmResourceGroupBackfill`` 标志会放宽切分约束，使 CUDA 能够填满那些否则无法达到所请求 ``smCount`` 的组。
按局部性域切分时，回填（backfill）的行为如下：

- 添加 ``cudaDevSmResourceGroupBackfill`` 后，API 会优先使用请求的局部性域中的非补充 SM 填充该组，然后使用该域的补充 SM，最后才使用其他局部性域中的 SM。
  如果在请求的局部性域中找不到任何 SM，调用将以 ``cudaErrorInvalidResourceConfiguration`` 失败。


.. _inspecting-split-results:

4.8.2.5. 检查切分结果
^^^^^^^^^^^^^^^^^^^^^

``result`` 的 ``cudaDevResource.sm.flags`` 和 ``cudaDevResource.sm.localityDomainId`` 报告资源约束。
当 ``flags`` 中存在 ``cudaDevSmResourceGroupLocalityDomainId`` 时， ``localityDomainId`` 标识所请求的局部性域，否则其值没有意义。
使用回填创建的资源可能包含来自其他局部性域的 SM，因此 ``localityDomainId`` 只反映最初请求的局部性域，而不是此类资源中出现的每个局部性域。

在一个描述符中组合 SM 资源时，这些资源必须具有匹配的约束，包括 ``flags`` 和 ``localityDomainId`` 值。


.. _creating-localized-streams:

4.8.2.6. 创建局部化流
^^^^^^^^^^^^^^^^^^^^^

以下代码查询局部性域的数量，并为每个局部性域构造一个流。

.. code-block:: cpp

   std::vector<cudaExecutionContext_t> contexts(localityDomainCount);
   std::vector<cudaStream_t> streams(localityDomainCount);

   {
       std::vector<cudaDevSmResourceGroupParams> params(localityDomainCount);
       std::vector<cudaDevResource> result(localityDomainCount);

       for (int i = 0; i < localityDomainCount; i++) {
           params[i] = {};
           params[i].flags = cudaDevSmResourceGroupLocalityDomainId;
           params[i].localityDomainId = i;
       }

       // A single split produces non-overlapping resources for all domains.
       cudaDevSmResourceSplit(result.data(), localityDomainCount, &smResource, nullptr, 0, params.data());

       for (int i = 0; i < localityDomainCount; i++) {
           cudaDevResourceDesc_t desc;
           cudaDevResourceGenerateDesc(&desc, &result[i], 1);
           cudaGreenCtxCreate(&contexts[i], desc, device, 0);
           cudaExecutionCtxStreamCreate(&streams[i], contexts[i], cudaStreamDefault, 0);
       }
   }

``streams`` 中的每个流都在具有对应索引的局部性域的 SM 上执行。
不再需要时，使用 ``cudaStreamDestroy`` 销毁这些流，然后使用 ``cudaExecutionCtxDestroy`` 销毁它们的上下文。


.. _locality-domains-memory:
.. _allocating-localized-device-memory:

4.8.3. 分配局部化设备内存
-------------------------

CUDA 使用 ``type`` 为 ``cudaMemLocationTypeDeviceLocalityDomain`` 的 ``cudaMemLocation`` 来表示局部化的内存位置。
对于此位置类型，需初始化 ``cudaMemLocation.localized`` 的两个成员：

- ``deviceId`` 标识 GPU。
- ``localityDomainId`` 标识该 GPU 上的一个局部性域。

运行时 API 将此位置与流序内存池一起使用（ :ref:`第 4.3 节 <stream-ordered-memory-allocation-details>` ）。
驱动 API 还额外将对应的 ``CUmemLocation`` 用于虚拟内存管理分配（ :ref:`第 4.18 节 <virtual-memory-management-details>` ）。

通过其他机制创建的分配（例如 ``cudaMalloc`` 或统一内存）无法局部化。
MLOPart 模式下的 `MPS（Multi-Process Service，多进程服务） <https://docs.nvidia.com/deploy/mps/index.html>`_ 可用于透明地局部化依赖 ``cudaMalloc`` 的应用程序。


.. _creating-a-localized-memory-pool:

4.8.3.1. 创建局部化内存池
^^^^^^^^^^^^^^^^^^^^^^^^^

要创建局部化内存池，需在 ``cudaMemPoolProps`` 中设置位置并调用 ``cudaMemPoolCreate`` ：

.. code-block:: cpp

   cudaMemPool_t pool {};
   cudaMemPoolProps props {};

   props.allocType = cudaMemAllocationTypePinned;
   props.location.type = cudaMemLocationTypeDeviceLocalityDomain;
   props.location.localized.deviceId = device;
   props.location.localized.localityDomainId = localityDomainId;

   cudaMemPoolCreate(&pool, &props);

使用 ``cudaMallocFromPoolAsync`` 从该池分配内存，并使用 ``cudaFreeAsync`` 释放分配。
在为该局部性域选择默认或当前的固定内存（pinned-memory）池时，该位置也可以传递给 ``cudaMemGetDefaultMemPool`` 、 ``cudaMemGetMemPool`` 和 ``cudaMemSetMemPool`` 。
有关通用的内存池编程模型，参见 :ref:`内存池 <soma-memory-pools>` 。


.. _creating-a-localized-vmm-allocation:

4.8.3.2. 创建局部化 VMM 分配
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

驱动 API 可以将由 ``cuMemCreate`` 创建的物理内存放置在某个局部性域中。
将 ``CUmemAllocationProp::location.type`` 设置为 ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`` ，并初始化 ``location.localized.deviceId`` 和 ``location.localized.localityDomainId`` 。
由此产生的物理分配的保留、映射和访问授权方式与其他 VMM 分配相同。
通过 ``cuMemSetAccess`` 设置的访问控制以整个设备为粒度，而不是单个局部性域。
完整的分配流程和粒度要求，参见 :ref:`局部化 VMM 分配小节 <virtual-memory-management-details>` 。

托管内存无法局部化：以 ``cudaMemLocationTypeDeviceLocalityDomain`` 请求 ``cudaMemAllocationTypeManaged`` 将返回 ``CUDA_ERROR_NOT_SUPPORTED`` 。
类似地， ``cudaMemPrefetchAsync`` 和 ``cudaMemAdvise`` 不接受设备局部性域作为位置，并将返回 ``CUDA_ERROR_NOT_SUPPORTED`` 。

有关局部性域与统一内存之间的关系，参见 :ref:`局部性域与统一内存 <um-details-intro>` 。


.. _querying-memory-placement:

4.8.4. 查询内存放置
-------------------

使用 ``cudaMemPoolGetAttribute`` 并传入 ``cudaMemPoolAttrLocalityDomainId`` 来查询内存池。
对于局部化的池，它返回局部性域 ID；对于未局部化的池，返回 ``-1`` 。

使用 ``cudaPointerGetAttributes`` 通过指针查询分配。
返回的 ``cudaPointerAttributes.localityDomainOrdinal`` 字段包含该分配的局部性域序号；当分配未局部化时为 ``-1`` ：

.. code-block:: cpp

   cudaPointerAttributes attributes {};
   cudaPointerGetAttributes(&attributes, ptr);
   int localityDomainId = attributes.localityDomainOrdinal;


.. _programming-guidance:

4.8.5. 编程指导
---------------

要局部化工作负载，请使用与其内存分配位于同一局部性域的计算资源来启动 kernel。

仅局部化计算资源或仅局部化内存资源通常不会提升性能。

如果一个 kernel 必须使用来自多个局部性域的内存，请实测计算局部化是否有益。


.. _localized-kernels-with-fork-join-pattern:

4.8.5.1. 采用 fork/join 模式的局部化 kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**fork/join 模式** 使用事件和流等待来表达一个并行区域。
应用程序使用 ``cudaExecutionCtxStreamCreate`` 从每个局部化绿色上下文创建一个流。
另有一个独立的主流负责协调这些局部化流。
使用 ``cudaEventCreateWithFlags`` 为主流创建一个 fork 事件，并为每个局部化流创建一个 join 事件；当事件仅用于排序时，应禁用计时。

执行 fork 时，在父流上记录一个事件，并让每个局部化流等待该事件。
因此，先前提交到父流的工作会排在局部化工作之前。
将独立的工作提交到各局部化流，并使用来自其对应局部性域的内存。

执行 join 时，在每个局部化流的工作之后记录一个事件，并让父流等待所有这些事件。
随后提交到父流的工作便会排在局部化工作之后。

.. code-block:: cuda

   produceInputs<<<grid, block, 0, parentStream>>>(...);
   cudaEventRecord(forkEvent, parentStream);

   // localizedStreams[i] belongs to the green context for locality domain i.
   for (int i = 0; i < localityDomainCount; i++) {
       cudaStreamWaitEvent(localizedStreams[i], forkEvent);
       processPartition<<<grid, block, 0, localizedStreams[i]>>>(...);
       cudaEventRecord(joinEvents[i], localizedStreams[i]);
   }

   for (int i = 0; i < localityDomainCount; i++) {
       cudaStreamWaitEvent(parentStream, joinEvents[i]);
   }
   consumeResults<<<grid, block, 0, parentStream>>>(...);


.. _locality-domains-cuda-graphs:

4.8.5.2. CUDA Graphs
^^^^^^^^^^^^^^^^^^^^

CUDA Graphs（ :ref:`第 4.2 节 <cuda-graphs>` ）通过流捕获和手动图构造支持局部化。
对于某些应用程序，CUDA 图相比基于流的编程可能带来更好的性能。


.. _stream-capture:

4.8.5.2.1. 流捕获
~~~~~~~~~~~~~~~~~

fork/join 模式可以被捕获为一个 CUDA 图。
在父流上开始捕获，记录 fork 事件，并在每个局部化流中等待该 fork 事件。
等待 fork 事件会将局部化流加入到与父流相同的捕获图中。
提交到局部化流的操作成为并行的图分支。
在所有局部化流上启动全部工作之后，记录 join 事件，并在父流中等待所有 join 事件。
这些等待将所有分支重新连接回父流。
在调用 ``cudaStreamEndCapture`` 之前，每个参与的流都必须汇合回起始流。

.. code-block:: cpp

   cudaStreamBeginCapture(parentStream, cudaStreamCaptureModeGlobal);

   // Record forkEvent, submit work to each localized stream, record the
   // joinEvents, and make parentStream wait as in the fork/join example above.

   cudaGraph_t graph;
   cudaStreamEndCapture(parentStream, &graph);

每个子流的局部性限制会被捕获的 kernel 节点保留。
即使可执行图在另一个流上启动，图重放仍使用原来的局部化 SM。
图启动流只提供排序，不会替换节点中记录的执行资源。


.. _graph-construction:

4.8.5.2.2. 图构造
~~~~~~~~~~~~~~~~~

也可以显式构造相同的图拓扑。
为每个局部性域添加一个 kernel 节点，并在 ``cudaKernelNodeParamsV2.ctx`` 中使用对应的局部化执行上下文。

.. code-block:: cpp

   cudaGraph_t graph = nullptr;
   cudaGraphCreate(&graph, 0);

   std::vector<cudaGraphNode_t> localizedNodes(localityDomainCount);
   for (int i = 0; i < localityDomainCount; i++) {
       cudaGraphNodeParams params = { CU_GRAPH_NODE_TYPE_KERNEL };
       params.kernel.kern = kernel;
       params.kernel.gridDim = grid;
       params.kernel.blockDim = block;
       params.kernel.kernelParams = kernelArgs[i];
       params.kernel.ctx = localizedContexts[i];

       cudaGraphAddNode(&localizedNodes[i], graph, nullptr, nullptr, 0, &params);
   }


.. _locality-domains-support:
.. _compatibility-notes:

4.8.5.3. 兼容性说明
^^^^^^^^^^^^^^^^^^^

vGPU 和 MIG 设备不支持局部化，将只报告 1 个局部性域。

Windows 上的 CUDA 不支持局部化，将只报告 1 个局部性域。

CUDA Checkpoint 不保证局部化绿色上下文在不同设备上恢复时仍保留其局部化属性。
恢复后应用程序的性能可能下降。

MLOPart 模式下的 `MPS（Multi-Process Service，多进程服务） <https://docs.nvidia.com/deploy/mps/index.html>`_ 将每个 MLOPart 设备呈现为具有单个局部性域的 CUDA 设备。
MPS 静态分区则将整个设备视为单个局部性域。
