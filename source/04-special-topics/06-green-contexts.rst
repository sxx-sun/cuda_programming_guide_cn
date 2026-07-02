.. _green-contexts-details:

4.6. Green Contexts
===================

绿色上下文（Green Context，GC）是一种轻量级上下文，在创建时便与一组特定的 GPU 资源绑定。
用户通过划分 GPU 资源，目前是 SM 和 工作队列（work queue，WQ）创建 GC ，使得针对该 GC 的 GPU 任务只能使用为其分配的 SM 和 WQ。
这有助于减少或更好地控制因资源共享而产生的干扰。
一个应用程序可以拥有多个 GC 。

使用 GC 无需修改任何 GPU 代码，仅需主机代码少量调整（例如创建 GC 、并为该上下文创建 CUDA 流）。
GC 适用于多种业务场景。例如，保留部分 SM 给低延迟敏感型任务，使其能够立即启动执行；也可快速验证缩减 SM 数量对性能的影响，全程无需改动 GPU 代码。

GC 首先是通过 `CUDA 驱动接口 <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS>`_ 提供的。
从 CUDA 13.1 开始，上下文开始通过执行上下文（Execution Context，EC）概念在 CUDA Runtime 中提供。
目前，一个 EC 可以对应主上下文（即 Runtime 一直用来隐式交互的上下文），也可以对应 GC 。
在本节中，当特指 GC 时，将交替使用 EC 和 GC 这两个术语。

由于当前 CUDA Runtime 已经提供 GC 接口，强烈建议直接使用 CUDA Runtime 接口。
本节也将完全基于 CUDA Runtime 讲解。

本节剩余内容的组织结构如下：

- 4.6.1 节提供了一个场景示例，介绍何时使用 GC
- 4.6.2 节强调了其易用性
- 4.6.3 节介绍了设备资源与资源描述符结构体
- 4.6.4 节讲解如何创建 GC
- 4.6.5 节说明如何启动针对该上下文的工作负载
- 4.6.6 节则介绍了其他一些相关的 GC 接口
- 4.6.7 节将通过一个完整的示例进行总结


.. _green-contexts-motivation:

4.6.1. 动机/何时使用
--------------------

启动 kernel 时，用户无法直接控制该 kernel 将在多少个 SM 上执行。
用户只能通过改变 kernel 的执行配置，或任何能影响每个 SM 上最大活跃线程块数量等因素，来间接施加影响。
此外，当多个 kernel 在 GPU 上并行执行时（例如在不同 CUDA 流上运行，或作为 CUDA 图的一部分），它们还可能争抢相同的 SM 资源。

然而，在某些应用场景中，用户需要确保始终有可用的 GPU 资源，保证延迟敏感的任务能够尽快启动并执行完毕。
GC 通过划分 SM 资源提供了一种实现该目标的方法，使得特定的 GC 只能使用指定的 SM（即在创建该 GC 时为其分配的那些 SM）。

:numref:`fig-green_contexts_motivation` 展示了该场景示例。
例如，假设一个应用程序，两个独立的核函数 A 和 B 在两个不同的非阻塞 CUDA 流上运行。
首先启动核函数 A 并开始执行，占用所有可用的 SM 资源。
稍后启动延迟敏感的核函数 B ，此时没有可用的 SM 资源。因此，核函数 B 只能等到核函数 A 释放资源（即来自核函数 A 的线程块完成执行）后才能开始执行。
第一个图说明了这种情况，关键工作 B 被延迟。y 轴显示占用的 SM 百分比，x 轴表示时间。

.. _fig-green_contexts_motivation:
.. figure:: /_static/images/green_contexts_motivation.png
   :alt: Green Contexts Motivation
   :align: center

   GC 的静态资源划分机制，可让延迟敏感任务 B 更快启动并完成运算。

使用 GC 我们可以对 GPU 的 SM 资源进行划分：核函数 A 绑定的 `GC-A` 占用 GPU 一部分 SM，核函数 B 绑定的 `GC-B` 占用剩余全部 SM。
在 GC 隔离机制下，无论核函数 A 采用何种启动配置，它仅能使用分配给 `GC-A` 的专属 SM。
如此一来，当关键核函数 B 发起执行时，若无其他资源限制，系统必定存在空闲 SM 供其立刻运行。
如 :numref:`fig-green_contexts_motivation` 中第二张曲线图所示，即便核函数 A 整体运行时长会有所增加，延迟敏感任务 B 也不会再因 SM 资源耗尽而被阻塞延迟。
图中示例仅作演示，为 `GC-A` 分配了整机 80% 的 SM 资源。

无需对核函数 A、核函数 B 做任何代码修改，即可实现该资源隔离效果。
开发者只需确保核函数被提交到属于相应 GC 的 CUDA 流上即可。
每个 GC 中能够使用的 SM 数量，应由用户在创建时根据具体情况自行决定。

**工作队列**：

SM 是可分配给 GC 的一类资源，另一种资源类型是工作队列（work queues）。
可以将工作队列视为一种黑盒资源抽象，它与其他因素一同影响 GPU 任务的执行并发度。
如果相互独立的 GPU 任务（例如在不同 CUDA 流上提交的核函数）被映射到同一个工作队列，就会在这些任务之间引入虚假的依赖关系，从而导致它们被串行执行。
用户可以通过 ``CUDA_DEVICE_MAX_CONNECTIONS`` 环境变量调整 GPU 上工作队列数量的上限（详见 :ref:`environment-variables-details` 和 :ref:`advanced-host-programming` ）。

在上一节示例的基础上，假设任务 B 与任务 A 被映射到了同一个工作队列。
在这种情况下，即使 SM 资源是可用的（在 GC 机制下），任务 B 仍然可能需要等待任务 A 完全执行完毕后才能开始。
与 SM 类似，用户无法直接控制底层实际使用的是哪些特定的工作队列。
但是，借助 GC ， 用户可基于预期的并发执行的流的数量，设定自身所需的最大并发度。
随后，驱动程序会将该值作为调度提示，尽量避免不同执行上下文的任务共用同一个工作队列，以此杜绝执行上下文之间不必要的资源抢占干扰。


.. warning::

   即便为每个 GC 分配了独立的 SM 与工作队列，也无法保证相互独立的 GPU 任务一定能够并发执行。
   应当将 GC 介绍的全部技术手段理解为消除阻碍并发执行的各种因素（即减少潜在的干扰）。

**GC 与 MIG 或 MPS 的比较**

为了内容的完整性，本节将简要对比 GC 与另外两种资源划分机制：
`多实例 GPU（ Multi-Instance GPU, MIG ） <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/index.html>`_  和
`多进程服务（ Multi-Process Service, MPS ） <https://docs.nvidia.com/deploy/mps/latest/index.html>`_ 。

MIG 可以对支持该功能的 GPU 做静态资源切分，划分出多个 MIG 实例（相当于多块小型独立 GPU）。
这种划分必须在应用程序启动之前完成，且不同的应用程序可以使用不同的 MIG 实例。
对那些应用程序未能充分利用 GPU 的用户而言，采用 MIG 会带来明显收益；而且随着 GPU 规模的不断增大，这一问题变得愈发突出。
借助 MIG，用户可以将不同的应用程序运行在不同的 MIG 实例上，从而提高 GPU 的利用率。
对于云服务提供商（CSP）来说，MIG 极具吸引力，不仅因为它能提高 GPU 利用率，还因为它能为运行在不同 MIG 实例上的客户端提供服务质量（QoS）保障和资源隔离。
更多详细信息，请参阅上方链接的 MIG 官方文档。

但是， MIG 无法解决前文所述的痛点场景：即关键任务 B 由于同一应用程序中的其他 GPU 任务占用了所有 SM 资源而被延迟。
对于运行在单个 MIG 实例上的应用程序，这一问题依然存在。为了解决这个问题，可以将 GC 与 MIG 结合使用。
在这种情况下，可用于划分的 SM 资源将是该特定 MIG 实例所拥有的资源。

MPS 主要针对不同的进程（例如 MPI 程序），允许它们同时在 GPU 上运行而无需采用时间片轮转调度。
在启动应用程序之前，需要先运行一个 MPS 守护进程（daemon）。
默认情况下，MPS 客户端会争抢其所在的 GPU 或 MIG 实例中所有可用 SM 资源。
在这种多客户端进程的场景下，MPS 可以通过 `活动线程百分比（active thread percentage）` 参数来实现 SM 资源的动态划分，该参数为 MPS 客户端进程可使用的 SM 百分比设定了上限。
与 GC 不同，MPS 的活动线程百分比划分是在进程级别进行的，并且该百分比通常需要在应用程序启动前通过环境变量来指定。
MPS 的活动线程百分比意味着，给定的客户端应用程序最多只能使用 GPU 的 `x%` 个 SM（假设为 N 个）。
然而，这 N 个 SM 可以是 GPU 上的任意 N 个，并且它们还会随时间发生变化。
与之相对的是，如果在创建 GC 时为其分配了固定 N 个 SM，那么它只能使用这特定的 N 个 SM。

从 CUDA 13.1 版本开始，在启动 MPS 控制守护进程时可以显式开启资源静态划分功能。
启用静态划分后，用户需要在启动应用时指定该 MPS 客户端进程所能使用的静态资源分区，此时基于活跃线程占比的动态资源共享机制将不再生效。
采用静态划分模式的 MPS 与 GC 有一处核心区别：MPS 针对的是不同的进程，而 GC 可在单个进程内部使用。
另外和 GC 不同，开启静态划分的 MPS 不允许对 SM 资源进行超额分配。

借助 MPS，通过 ``cuCtxCreate`` 创建的带有执行亲和性（execution affinity）的 CUDA 上下文，可以实现 SM 资源的编程式划分。
这种编程式划分允许来自一个或多个进程的不同客户端 CUDA 上下文，各自最多使用指定数量的 SM。
与 `活动线程百分比` 划分机制类似，这些 SM 可以是 GPU 上的任意 SM，并且会随时间发生变化，这与 GC 的情况不同。
即使在启用了 MPS 静态划分的情况下，该选项依然可用。
需要注意的是，与 MPS 上下文相比，创建 GC 的开销要轻量得多，因为许多底层结构由主上下文（primary context）持有并共享。


.. _green-contexts-ease-of-use:

4.6.2. GC ：易用性
-----------------------------

为了说明使用 GC 有多么简便，假设你有以下代码片段：
该片段创建了两个 CUDA 流，然后调用一个函数，通过 ``<<>>>`` 语法在这些流上启动核函数。
正如前文所述，除了修改核函数的启动配置（如网格和线程块大小）之外，开发者是无法控制这些核函数能使用多少个 SM 的。

.. code-block:: c++

   int gpu_device_index = 0; // GPU ordinal
   CUDA_CHECK(cudaSetDevice(gpu_device_index));

   cudaStream_t strm1, strm2;
   CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
   CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

   // No control over how many SMs kernel(s) running on each stream can use
   // what is abstracted in this function + the kernels is the vast majority of your code
   code_that_launches_kernels_on_streams(strm1, strm2);

   // cleanup code not shown


从 CUDA 13.1 开始，开发者可以利用 GC 来控制特定核函数能够使用的 SM 数量。
下面的代码片段展示了实现这一操作有多么简单。
只需增加几行代码，且无需对核函数本身进行任何修改，你就能控制通过不同流启动的核函数所能使用的 SM 资源。

.. code-block:: c++

   int gpu_device_index = 0; // GPU ordinal
   CUDA_CHECK(cudaSetDevice(gpu_device_index));

   /* ------------------ Code required to create green contexts --------------------------- */


   // Get all available GPU SM resources
   cudaDevResource initial_GPU_SM_resources {};
   CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index,
                                       &initial_GPU_SM_resources,
                                       cudaDevResourceTypeSm));

   // Split SM resources. Assuming your GPU has >= 24 SMs
   // This example creates one group with 16 SMs and one with 8.
   cudaDevSmResource result[2] {{}, {}};
   cudaDevSmResourceGroupParams group_params[2] =  {
         {.smCount=16, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
         {.smCount=8,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
   CUDA_CHECK(cudaDevSmResourceSplit(&result[0],
                                     2,
                                     &initial_GPU_SM_resources,
                                     nullptr,
                                     0,
                                     &group_params[0]));

   // Generate resource descriptors for each resource
   cudaDevResourceDesc_t resource_desc1 {};
   cudaDevResourceDesc_t resource_desc2 {};
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc1, &result[0], 1));
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc2, &result[1], 1));

   // Create green contexts
   cudaExecutionContext_t my_green_ctx1 {};
   cudaExecutionContext_t my_green_ctx2 {};
   CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx1, resource_desc1, gpu_device_index, 0));
   CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx2, resource_desc2, gpu_device_index, 0));

   /* ------------------ Modified code --------------------------- */

   // You just need to use a different CUDA API to create the streams
   cudaStream_t strm1, strm2;
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm1, my_green_ctx1, cudaStreamDefault, 0));
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm2, my_green_ctx2, cudaStreamDefault, 0));

   /* ------------------ Unchanged code --------------------------- */

   // No need to modify any code in this function or in your kernel(s).
   // Reminder: what is abstracted in this function + kernels is the vast majority of your code
   // Now kernel(s) running on stream strm1 will use at most 16 SMs
   // and kernel(s) on strm2 at most 8 SMs.
   code_that_launches_kernels_on_streams(strm1, strm2);

   // cleanup code not shown

各种执行上下文接口（其中部分在前面的示例中已展示）接收显式的 ``cudaExecutionContext_t`` 句柄，因此会忽略当前调用线程所绑定的上下文。
在此之前，如果不使用驱动接口，CUDA Runtime 默认只会与主上下文进行交互，该上下文在 ``cudaSetDevice()`` 中被隐式设置为当前线程的上下文。
这种向显式上下文编程的转变，提供了更易于理解的语义；与之前依赖 `thread-local` 的隐式上下文编程相比，它还具备额外的优势。

后续章节会详细讲解上一段代码片段中展示的全部操作步骤。

.. _green-contexts-device-resource-and-desc:

4.6.3. GC ：设备资源和资源描述符
--------------------------------------------

GC 的核心是绑定特定 GPU 设备的设备资源（ ``cudaDevResource`` ）。
这些资源可以被组合并封装到一个描述符（ ``cudaDevResourceDesc_t`` ）中。
GC 只能访问创建它时所用描述符中封装的资源。

目前 ``cudaDevResource`` 数据结构定义为：

.. code-block:: c++

   struct {
        enum cudaDevResourceType type;
        union {
            struct cudaDevSmResource sm;
            struct cudaDevWorkqueueConfigResource wqConfig;
            struct cudaDevWorkqueueResource wq;
        };
    };


其中支持的资源类型和描述如下表。


.. list-table:: GC 资源类型
   :widths: 50 50
   :header-rows: 1
   :align: center

   * - 类型
     - 资源

   * - ``cudaDevResourceTypeInvalid``
     - 无效的资源类型

   * - ``cudaDevResourceTypeSm``
     - 特定的一组流式 SM

   * - ``cudaDevResourceTypeWorkqueueConfig``
     - 特定的工作队列配置

   * - ``cudaDevResourceTypeWorkqueue``
     - 预先存在的工作队列资源


可以使用 ``cudaExecutionCtxGetDevResource`` 和 ``cudaStreamGetDevResource`` 接口查询给定的执行上下文或 CUDA 流是否与给定类型的 ``cudaDevResource`` 关联。
执行上下文可以关联不同类型的设备资源（例如 SM 和工作队列），而流只能与 SM 类型资源关联。

默认情况下，给定的 GPU 设备拥有全部三种设备资源：包含该 GPU 所有的 SM 资源、包含所有的工作队列配置，以及对应的工作队列。
这些资源可以通过 ``cudaDeviceGetDevResource`` 查询。

**相关设备资源结构体概述**

不同资源类型对应的结构体包含若干字段，这些字段的值既可由用户显式赋值，也可通过对应的 CUDA 接口完成设置。
建议对所有设备资源结构体执行零初始化操作。

-  SM 类型设备资源 ``cudaDevSmResource`` 包含以下相关字段：

  - ``unsigned int smCount`` ：可用的 SM 数量
  - ``unsigned int minSmPartitionSize`` ：切分该资源所需的最小 SM 数量。
  - ``unsigned int smCoscheduledAlignment`` ：保证在同一个 GPC 上协同调度的 SM 数量，该参数对线程块簇功能至关重要。当 ``flags`` 为零时， ``smCount`` 是此值的倍数。
  - ``unsigned int flags`` ：支持 0（默认）和 ``cudaDevSmResourceGroupBackfill`` （见 ``cudaDevSmResourceGroup`` ）。

  上述字段有两种赋值途径：要么由创建 SM 类型资源时调用的切分接口（ ``cudaDevSmResourceSplitByCount`` 或 ``cudaDevSmResourceSplit`` ）自动填充；
  要么由检索指定 GPU 设备 SM 资源的 ``cudaDeviceGetDevResource`` 接口填充。
  用户 **切勿直接手动修改** ，更多细节请参阅下一章节。

- 工作队列配置设备资源 ``cudaDevWorkqueueConfigResource`` 包含以下相关字段：

  - ``int device`` ：工作队列资源所在的设备
  - ``unsigned int wqConcurrencyLimit`` ：为规避虚假依赖所需的有序流任务数量
  - ``enum cudaDevWorkqueueConfigScope sharingScope`` ：工作队列资源的共享范围。
    支持 ``cudaDevWorkqueueConfigScopeDeviceCtx`` （默认）和 ``cudaDevWorkqueueConfigScopeGreenCtxBalanced`` 两个值。
    在默认选项下，所有工作队列资源由所有上下文共享；
    而在均衡（balanced）选项下，驱动程序会尽可能尝试在不同的 GC 之间使用互不重叠的工作队列资源，并将用户指定的 ``wqConcurrencyLimit`` 作为参考提示。

  这些字段需要由用户自行设置。
  CUDA 没有提供用于生成工作队列配置资源的拆分接口，唯一的例外是 ``cudaDeviceGetDevResource`` ， 该接口检索指定 GPU 设备的工作队列配置资源并自动填充。

- 最后，对于预先存在的工作队列资源（ ``cudaDevResourceTypeWorkqueue`` ），没有任何可供用户设置的字段。
  与其他资源类型一样，可以通过 ``cudaDeviceGetDevResource`` 接口来检索指定 GPU 设备的预先存在的工作队列资源。

.. _green-contexts-creation-example:

4.6.4. Green Context 创建示例
-----------------------------

Green Context 创建涉及四个主要步骤：

- **步骤 1**：初始化资源结构体，例如通过获取 GPU 的可用资源来完成初始化。
- **步骤 2**：将 SM 资源划分为一个或多个分区（调用任一可用的资源拆分接口实现）。
- **步骤 3**：创建资源描述符；如有需要，可将多种不同资源组合填入该描述符。
- **步骤 4**：基于该资源描述符创建 GC ，并为其分配对应硬件资源。

GC 创建完成后，创建归属该 GC 的 CUDA 流，后续在此类流上提交的 GPU 任务（例如通过 ``<<<>>>`` 启动的核函数），仅能使用该 GC 所分配的专属资源。
各类第三方库也可便捷使用 GC ，前提是用户向库传入一条隶属于 GC 的流。
更多细节请参阅 :ref:`green-contexts-launching-work` 章节。

.. _green-contexts-creation-example-step1:

4.6.4.1. 步骤 1：获取可用的 GPU 资源
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

创建 GC 的第一步，通过获取可用的设备资源，初始化 ``cudaDevResource`` 结构体。
目前，共有三种获取入口：设备、执行上下文或 CUDA 流。 相关的 Runtime 接口如下：

.. list-table:: 获取可用的 GPU 资源的 Runtime 接口
   :widths: 10 90
   :header-rows: 1
   :align: center

   * - 获取对象
     - 函数定义

   * - 设备
     - .. code-block:: cuda

        cudaError_t cudaDeviceGetDevResource(int device,
                                             cudaDevResource* resource,
                                             cudaDevResourceType type);

   * - 执行上下文
     - .. code-block:: cuda

        cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx,
                                                   cudaDevResource* resource,
                                                   cudaDevResourceType type);

   * - 流
     - .. code-block:: cuda

        cudaError_t cudaStreamGetDevResource(cudaStream_t hStream,
                                             cudaDevResource* resource,
                                             cudaDevResourceType type);



``cudaStreamGetDevResource`` 只支持  SM 资源类型， 即 ``cudaDevResourceTypeSm`` 。
其他接口支持任何有效的 ``cudaDevResourceType`` 。


通常情况下，我们首先通过设备获取资源数据。
下面的代码展示了如何获取指定 GPU 设备的可用 SM 资源。
在成功调用 ``cudaDeviceGetDevResource`` 之后，用户可以查看该资源中可用的 SM 数量。

.. code-block:: c++

   int current_device = 0; // assume device ordinal of 0
   CUDA_CHECK(cudaSetDevice(current_device));

   cudaDevResource initial_SM_resources = {};
   CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU device */,
                                    &initial_SM_resources /* device resource to populate */,
                                    cudaDevResourceTypeSm /* resource type*/));

   std::cout << "Initial SM resources: "
             << initial_SM_resources.sm.smCount
             << " SMs" << std::endl; // number of available SMs

   // Special fields relevant for partitioning (see Step 3 below)
   std::cout << "Min. SM partition size: "
             <<  initial_SM_resources.sm.minSmPartitionSize
             << " SMs" << std::endl;

   std::cout << "SM co-scheduled alignment: "
             <<  initial_SM_resources.sm.smCoscheduledAlignment
             << " SMs" << std::endl;

.. _green-contexts-split-sm-resources:

开发者同样可以获取可用的工作队列配置资源，示例如下。

.. code-block:: c++

   int current_device = 0; // assume device ordinal of 0
   CUDA_CHECK(cudaSetDevice(current_device));

   cudaDevResource initial_WQ_config_resources = {};
   CUDA_CHECK(cudaDeviceGetDevResource(current_device,
                                       &initial_WQ_config_resources,
                                       cudaDevResourceTypeWorkqueueConfig));

   std::cout << "Initial WQ config. resources: " << std::endl;
   std::cout << "  - WQ concurrency limit: "
             << initial_WQ_config_resources.wqConfig.wqConcurrencyLimit << std::endl;
   std::cout << "  - WQ sharing scope: "
             << initial_WQ_config_resources.wqConfig.sharingScope << std::endl;

在成功调用 ``cudaDeviceGetDevResource`` 之后，用户可以查看该资源的 ``wqConcurrencyLimit`` （工作队列并发限制）。
当起始对象为 GPU 设备时， ``wqConcurrencyLimit`` 的值将与环境变量 ``CUDA_DEVICE_MAX_CONNECTIONS`` 的设置值或其默认值一致。


4.6.4.2. 步骤 2： SM 资源切分
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

第二步，是将 SM 资源静态切分为一个或多个分区，同时可能会留下部分未分配的 SM 保留在 `剩余分区` （remaining partition）中。
这一操作可以通过 ``cudaDevSmResourceSplitByCount()`` 或 ``cudaDevSmResourceSplit()`` 来实现。
``cudaDevSmResourceSplitByCount()`` 只能创建一个或多个相同分区，外加一个可能的剩余分区。
``cudaDevSmResourceSplit()`` 创建一个或多个不同分区，外加一个可能的剩余分区。
后续章节将详细介绍这两个接口的具体功能。需要注意的是，这两个接口仅适用于 SM 资源切分。


4.6.4.2.1. cudaDevSmResourceSplitByCount
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

函数定义如下：

.. code-block:: c++

   cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource* result,
                                             unsigned int* nbGroups,
                                             const cudaDevResource* input,
                                             cudaDevResource* remaining,
                                             unsigned int useFlags,
                                             unsigned int minCount)

如 :numref:`fig-green_contexts_resource_split_by_count` ，用户请求将输入的 SM 资源拆分为 ``*nbGroups`` 个相同组，每组至少包含 ``minCount`` 个 SM。
然而，最终的结果可能会包含数量更新后的 ``*nbGroups`` 个组，且每个组包含 N 个 SM。
更新后的 ``*nbGroups`` 数量将小于或等于最初请求的组数，而 N 将大于或等于 ``minCount`` 。
这些调整是由于特定硬件架构的粒度和对齐要求所导致的。

.. _fig-green_contexts_resource_split_by_count:
.. figure:: /_static/images/green_contexts_resource_split_by_count.png
   :alt: SM resource split using the cudaDevSmResourceSplitByCount API
   :align: center

   使用 `cudaDevSmResourceSplitByCount` 接口划分 SM 资源

:numref:`compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability` 列出了在默认配置 ``useFlags=0`` 的情况下，
所有已支持的计算能力对应的最小 SM 分区大小以及 SM 协同调度对齐要求。
开发者也可以按照步骤 1 的方法， 通过 ``cudaDevSmResource`` 结构体中的 ``minSmPartitionSize`` 和 ``smCoscheduledAlignment`` 字段来获取这些值。
此外，通过设置不同的 ``useFlags`` 值，可以放宽部分资源限制条件。


:numref:`table-split-functionality` 提供了一些相关示例，重点突出了 `用户请求的资源数量` 与 `最终实际分配结果` 之间的差异，并附带了详细说明。
该表以计算能力 9.0 为例：当 ``useFlags=0`` 时，每个分区的最小 SM 数量为 8，并且 SM 总数必须是 8 的倍数。

.. _table-split-functionality:
.. list-table:: 资源切分能力 （以 GH200 为例，设备一共有 132 个 SM ）
   :header-rows: 1
   :align: center

   * - 请求
     -
     -
     - 实际分配
     -
     -

   * - ``*nbGroups``
     - minCount
     - useFlags
     - | ``*nbGroups``,
       | N SM
     - 剩余 SM
     - 原因

   * - 2
     - 72
     - 0
     - 1 group, 72 SM
     - 60
     - 总数不超过 132

   * - 6
     - 11
     - 0
     - 6 group, 16 SM
     - 36
     - 8 的倍数

   * - 6
     - 11
     - | ``CU_DEV_SM_``
       | ``RESOURCE_SPLIT_``
       | ``IGNORE_``
       | ``SM_COSCHEDULING``
     - 6 group, 12 SM
     - 60
     - 最小 2 的倍数

   * - 2
     - 1
     - 0
     - 2 group, 8 SM
     - 116
     - 最小是 8

下方代码片段展示了相关调用逻辑，其需求为将可用 SM 资源拆分为 5 组，每组包含 8 个 SM：

.. code-block:: cpp

   int current_device = 0; // assume device ordinal of 0
   cudaDevResource avail_resources = {};

   CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU device */,
                                       &avail_resources /* device resource to populate */,
                                       cudaDevResourceTypeSm /* resource type*/));

   unsigned int min_SM_count = 8;
   unsigned int actual_split_groups = 5; // may be updated

   cudaDevResource actual_split_result[5] = {{}, {}, {}, {}, {}};
   cudaDevResource remaining_partition = {};

   CUDA_CHECK(cudaDevSmResourceSplitByCount(&actual_split_result[0],
                                            &actual_split_groups,
                                            &avail_resources,
                                            &remaining_partition,
                                            0 /*useFlags */,
                                            min_SM_count));

   std::cout << "Split " << avail_resources.sm.smCount << " SMs into "
             << actual_split_groups << " groups "
             << "with " << actual_split_result[0].sm.smCount << " each "
             << "and a remaining group with " << remaining_partition.sm.smCount << " SMs"
             << std::endl;

清注意:

- 如果只是想计算分组个数，可以设置 ``result=nullptr`` 。
- 如果不关心剩余分组 SM ，可以设置 ``remaining=nullptr`` 。
- 剩余分组中的 SM， 并不具备与 ``result`` 中的分组相同的功能或性能保障。
- ``useFlags`` 默认是 0 ， 支持 ``cudaDevSmResourceSplitIgnoreSmCoscheduling`` 和 ``cudaDevSmResourceSplitMaxPotentialClusterSize`` 选项。
- 任何分组结果都不能被直接重新切分，必须先基于它创建一个资源描述符和 GC （即下文中的步骤 3 和步骤 4）。

请参考 Runtime 接口文档中关于 `cudaDevSmResourceSplitByCount <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html#group__CUDART__EXECUTION__CONTEXT_1g10ef763a79ff53245bec99b96a7abb73>`_
的更多描述。

4.6.4.2.1. cudaDevSmResourceSplit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

函数定义：

.. code-block:: c++

   cudaError_t cudaDevSmResourceSplit(cudaDevResource* result,
                                      unsigned int nbGroups,
                                      const cudaDevResource* input,
                                      cudaDevResource* remainder,
                                      unsigned int flags,
                                      cudaDevSmResourceGroupParams* groupParams)


``cudaDevSmResourceSplit`` 允许用户在单次调用中创建互不重叠的不同分区。
该接口会尝试根据 ``groupParams`` 数组中为每个组指定的要求，将输入的 SM 类型资源切分为 ``nbGroups`` 个有效的设备资源（组），并将它们放入 ``result`` 数组中。
此外，还可能会创建一个可选的剩余分区。
在切分成功的情况下（如图 47 所示）， ``result`` 数组中的每个资源可以包含不同数量的 SM，但 SM 数量绝不能为零。

.. _green-contexts-add-workqueue-resources:

4.6.4.3. 步骤 2（续）：添加工作队列资源
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果还想指定工作队列资源，则需要显式执行。以下示例展示了如何为特定设备创建工作队列配置资源，具有平衡的共享范围和四个并发限制。

.. code-block:: c++

   cudaDevResource split_result[2] = {{}, {}};
   // 填充 split_result[0] 的代码未显示；使用 split API，nbGroups=1

   // 最后一个资源将是工作队列资源
   split_result[1].type = cudaDevResourceTypeWorkqueueConfig;
   split_result[1].wqConfig.device = 0; // 假设设备序号为 0
   split_result[1].wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
   split_result[1].wqConfig.wqConcurrencyLimit = 4;

.. _green-contexts-create-resource-desc:

4.6.4.4. 步骤 3：创建资源描述符
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

分割 SM 资源（以及可选地添加工作队列配置资源）后，下一步是创建资源描述符。这可以通过 ``cudaDevResourceGenerateDesc`` API 完成。

.. code-block:: c++

   cudaDevResourceDesc_t resource_desc {};
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc, split_result, 2));

.. _green-contexts-create-green-ctx:

4.6.4.5. 步骤 4：创建 Green Context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最后一步是使用 ``cudaGreenCtxCreate`` API 创建 Green Context。

.. code-block:: c++

   cudaExecutionContext_t green_ctx {};
   CUDA_CHECK(cudaGreenCtxCreate(&green_ctx, resource_desc, device_id, 0));

.. _green-contexts-launching-work:

4.6.5. Green Contexts：启动工作
-------------------------------------

Green Context 创建后，您可以创建属于该 Green Context 的 CUDA 流。随后在此类流上启动的 GPU 工作（如通过 ``<<< >>>`` 启动的核函数）将只能访问此 Green Context 分配的资源。

.. code-block:: c++

   cudaStream_t stream;
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&stream, green_ctx, cudaStreamDefault, 0));

   // 在此流上启动的核函数将只使用 Green Context 的分配资源
   my_kernel<<<grid, block, 0, stream>>>(args);

库也可以轻松利用 Green Contexts，只要用户将属于 Green Context 的流传递给他们。

.. _green-contexts-apis:

4.6.6. Green Contexts：API
--------------------------

以下是与 Green Contexts 相关的一些额外 API：

- ``cudaGreenCtxDestroy`` ：销毁 Green Context
- ``cudaExecutionCtxGetDevResource`` ：获取执行上下文的设备资源
- ``cudaStreamGetDevResource`` ：获取 CUDA 流的设备资源
- ``cudaDevSmResourceSplitByCount`` ：通过计数分割 SM 资源
- ``cudaDevSmResourceSplit`` ：分割 SM 资源为异构组
- ``cudaDevResourceGenerateDesc`` ：生成资源描述符

.. _green-contexts-example:

4.6.7. Green Contexts：示例
----------------------------

以下是一个完整的示例，展示了如何使用 Green Contexts：

.. code-block:: c++

   #include <cuda_runtime.h>
   #include <iostream>

   __global__ void kernel_a() {
       // 长时间运行的核函数
       for (int i = 0; i < 1000000; i++) {
           // 一些计算
       }
   }

   __global__ void kernel_b() {
       // 延迟敏感的核函数
       // 一些关键计算
   }

   int main() {
       int device = 0;
       cudaSetDevice(device);

       // 获取初始 SM 资源
       cudaDevResource initial_resources {};
       cudaDeviceGetDevResource(device, &initial_resources, cudaDevResourceTypeSm);

       // 分割 SM 资源：80% 用于 Green Context A，20% 用于 Green Context B
       cudaDevSmResource result[2] {{}, {}};
       cudaDevSmResourceGroupParams group_params[2] =  {
               {.smCount=initial_resources.sm.smCount * 80 / 100, .flags=0},
               {.smCount=initial_resources.sm.smCount * 20 / 100, .flags=0}};
       cudaDevSmResourceSplit(&result[0], 2, &initial_resources, nullptr, 0, &group_params[0]);

       // 创建资源描述符
       cudaDevResourceDesc_t desc_a {}, desc_b {};
       cudaDevResourceGenerateDesc(&desc_a, &result[0], 1);
       cudaDevResourceGenerateDesc(&desc_b, &result[1], 1);

       // 创建 Green Contexts
       cudaExecutionContext_t ctx_a {}, ctx_b {};
       cudaGreenCtxCreate(&ctx_a, desc_a, device, 0);
       cudaGreenCtxCreate(&ctx_b, desc_b, device, 0);

       // 创建属于 Green Contexts 的流
       cudaStream_t stream_a, stream_b;
       cudaExecutionCtxStreamCreate(&stream_a, ctx_a, cudaStreamDefault, 0);
       cudaExecutionCtxStreamCreate(&stream_b, ctx_b, cudaStreamDefault, 0);

       // 启动长时间运行的核函数
       kernel_a<<<grid, block, 0, stream_a>>>();

       // 立即启动延迟敏感的核函数
       // 即使核函数 A 正在运行，它也有保证的 SM 可用
       kernel_b<<<grid, block, 0, stream_b>>>();

       // 清理
       cudaStreamDestroy(stream_a);
       cudaStreamDestroy(stream_b);
       cudaGreenCtxDestroy(ctx_a);
       cudaGreenCtxDestroy(ctx_b);

       return 0;
   }

.. note::

   有关 Green Contexts 的更多详细信息，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html>`_。
