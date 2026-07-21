.. _green-contexts-details:

4.6. Green Contexts
===================

绿色上下文（Green Context，GC）是一种轻量级上下文，在创建时便与一组特定的 GPU 资源绑定。
用户通过划分 GPU 资源创建 GC ，使得针对该 GC 的 GPU 任务只能使用为其分配的 SM 和 工作队列（work queue，WQ）。
这有助于减少或更好地控制因资源共享而产生的干扰。
一个应用程序可以拥有多个 GC 。

使用 GC 无需修改任何 GPU 代码，仅需主机代码少量调整（例如创建 GC 、并为该上下文创建 CUDA 流）。
GC 适用于多种业务场景。例如，保留部分 SM 给低延迟敏感型任务，使其能够立即启动执行；也可快速验证缩减 SM 数量对性能的影响，全程无需改动 GPU 代码。

GC 最早是通过 `CUDA 驱动接口 <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS>`_ 提供的。
从 CUDA 13.1 开始， GC 开始以执行上下文（Execution Context，EC）概念在 CUDA Runtime 中提供。
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
GC 通过对 SM 资源的划分提供了一种实现该目标的方法，使得特定的 GC 中的任务只能使用指定的 SM（即在创建该 GC 时为其分配的那些 SM）。

:numref:`fig-green_contexts_motivation` 展示了该场景示例。
例如，假设一个应用程序，两个独立的核函数 A 和 B 在两个不同的非阻塞 CUDA 流上运行。
核函数 A 先启动执行并占用所有可用的 SM 资源，稍后启动延迟敏感的核函数 B 。
此时由于没有 SM 可用，核函数 B 只能等到核函数 A 释放资源（即来自核函数 A 的线程块完成执行）后才能开始执行。
第一个图说明了这种情况，关键工作 B 被延迟。y 轴显示占用的 SM 百分比，x 轴表示时间。

.. _fig-green_contexts_motivation:
.. figure:: /_static/images/green_contexts_motivation.png
   :alt: Green Contexts Motivation
   :align: center

   GC 的静态资源划分机制，可让延迟敏感任务 B 更快启动并完成运算。

使用 GC 我们可以对 GPU 的 SM 资源进行划分：核函数 A 绑定的 `GC-A` 占用 GPU 一部分 SM，核函数 B 绑定的 `GC-B` 占用剩余 SM。
在 GC 隔离机制下，无论核函数 A 采用何种启动配置，它仅能使用分配给 `GC-A` 的专属 SM。
如此一来，当关键核函数 B 发起执行时，若无其他资源限制，系统必定存在空闲 SM 供其立刻运行。
如 :numref:`fig-green_contexts_motivation` 中第二张曲线图所示，即便核函数 A 整体运行时长会有所增加，延迟敏感任务 B 不会因 SM 资源耗尽而被阻塞延迟。
图中示例仅作演示，为 `GC-A` 分配了整机 80% 的 SM 资源。

无需对核函数 A、核函数 B 做任何代码修改，即可实现该资源隔离效果。
开发者只需确保核函数被提交到属于相应 GC 的 CUDA 流上即可。
每个 GC 中能够使用的 SM 数量，由用户在创建时根据具体情况自行决定。

**工作队列**：

SM 是可分配给 GC 的一种资源，另一种资源是工作队列（work queues）。
可以将工作队列视为一种黑盒资源抽象，它与其他因素一同影响 GPU 任务的执行并发度。
如果相互独立的 GPU 任务（例如在不同 CUDA 流上提交的核函数）被映射到同一个工作队列，就会在这些任务之间引入虚假的依赖关系，从而导致它们被串行执行。
用户可以通过 ``CUDA_DEVICE_MAX_CONNECTIONS`` 环境变量调整 GPU 上工作队列数量的上限（详见 :ref:`environment-variables-details` 和 :ref:`advanced-host-programming` ）。

在上一节示例的基础上，假设任务 B 与任务 A 被映射到了同一个工作队列。
在这种情况下，即使 SM 资源是可用的（在 GC 机制下），任务 B 仍然可能需要等待任务 A 完全执行完毕后才能开始。
与 SM 类似，用户无法直接控制底层实际使用的是哪些特定的工作队列。
但是，借助 GC ， 用户可基于预期的并发执行的流的数量，设定自身所需的最大并发度。
随后，驱动程序会将该值作为调度提示，尽量避免不同 EC 中的任务共用同一个工作队列，以此杜绝 EC 之间不必要的资源抢占干扰。


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

   // cleanup code
   CUDA_CHECK(cudaStreamDestroy(strm2));
   CUDA_CHECK(cudaStreamDestroy(strm1));

   CUDA_CHECK(cudaExecutionCtxDestroy(my_green_ctx2));
   CUDA_CHECK(cudaExecutionCtxDestroy(my_green_ctx1));

   CUDA_CHECK(cudaDeviceReset());


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

- SM 类型设备资源 ``cudaDevSmResource`` 包含以下相关字段：

  - ``unsigned int smCount`` ：可用的 SM 数量
  - ``unsigned int minSmPartitionSize`` ：切分该资源所需的最小 SM 数量。
  - ``unsigned int smCoscheduledAlignment`` ：保证在同一个 GPC 上协同调度的 SM 数量，该参数对线程块簇功能至关重要。当 ``flags`` 为 `0` 时， ``smCount`` 必须是此值的倍数。
  - ``unsigned int flags`` ：支持 0（默认）和 ``cudaDevSmResourceGroupBackfill`` （见 ``cudaDevSmResourceGroup`` ）。

  上述字段有两种赋值途径：由 SM 类型资源切分接口（ ``cudaDevSmResourceSplitByCount`` 或 ``cudaDevSmResourceSplit`` ）填充；
  或者由检索指定 GPU 设备 SM 资源的接口（ ``cudaDeviceGetDevResource`` ）填充。
  **切勿直接手动修改** ，更多细节请参阅下一章节。

- 工作队列配置设备资源 ``cudaDevWorkqueueConfigResource`` 包含以下相关字段：

  - ``int device`` ：WQ 所在的设备
  - ``unsigned int wqConcurrencyLimit`` ：期望并行执行的流的数量
  - ``enum cudaDevWorkqueueConfigScope sharingScope`` ： WQ 资源的共享范围。取值如下：

    - ``cudaDevWorkqueueConfigScopeDeviceCtx`` （默认），所有 WQ 由所有上下文共享。
    - ``cudaDevWorkqueueConfigScopeGreenCtxBalanced`` ，驱动程序会尽可能尝试在不同的 GC 之间使用互不重叠的 WQ 资源，并将用户指定的 ``wqConcurrencyLimit`` 作为参考提示。

  这些字段需要由用户自行设置。
  CUDA 没有提供用于生成 WQ 配置资源的拆分接口，但是通过 ``cudaDeviceGetDevResource`` 接口可以检索指定 GPU 设备的 WQ 配置资源。

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
   :header-rows: 1
   :align: center

   * - 对象
     - 函数定义
     - 支持的资源
   * - 设备
     - .. code-block:: cuda

        cudaError_t
        cudaDeviceGetDevResource(int device,
                                 cudaDevResource* resource,
                                 cudaDevResourceType type);

     - ALL
   * - 执行上下文
     - .. code-block:: cuda

        cudaError_t
        cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx,
                                       cudaDevResource* resource,
                                       cudaDevResourceType type);

     - ALL
   * - 流
     - .. code-block:: cuda

        cudaError_t
        cudaStreamGetDevResource(cudaStream_t hStream,
                                 cudaDevResource* resource,
                                 cudaDevResourceType type);

     - SM


``cudaStreamGetDevResource`` 只支持 SM 资源类型， 即 ``cudaDevResourceTypeSm`` 。
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


.. _green-contexts-split-sm-resources:

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

.. code-block:: c++

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

4.6.4.2.2. cudaDevSmResourceSplit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如前所述，单次 ``cudaDevSmResourceSplitByCount`` 接口调用只能创建同构分区（即具有相同 SM 数量的分区）加上一个可能的剩余分区。
对于在不同 GC 上运行的工作负载具有不同 SM 数量需求的异构工作负载而言，这存在局限性。
若要使用按计数切分接口实现异构分区，通常需要重复执行步骤 1-4（多次）来对现有资源重新切分。
或者，在某些情况下，可以在步骤 2 中创建每组 SM 数量等于所有异构分区所需 SM 数量的最大公约数（GCD）的同构分区，
然后在步骤 3 中将所需数量的分组合并在一起。但不推荐采用最后这种方式，因为如果在初始时请求更大的 SM 数量，CUDA 驱动程序可能能够创建更优的分区。

``cudaDevSmResourceSplit`` 接口旨在解决这些局限，允许用户在单次调用中创建互不重叠的异构分区。
其函数定义如下：

.. code-block:: c++

   cudaError_t cudaDevSmResourceSplit(cudaDevResource* result,
                                      unsigned int nbGroups,
                                      const cudaDevResource* input,
                                      cudaDevResource* remainder,
                                      unsigned int flags,
                                      cudaDevSmResourceGroupParams* groupParams)


该接口会尝试根据 ``groupParams`` 数组中为每个组指定的要求，将输入的 SM 类型资源切分为 ``nbGroups`` 个有效的设备资源（组），并将它们放入 ``result`` 数组中。
此外，还可能会创建一个可选的剩余分区。
在切分成功的情况下（如 :numref:`fig-green_contexts_resource_split` 所示）， ``result`` 数组中的每个资源可以包含不同数量的 SM，但 SM 数量绝不能为零。

.. _fig-green_contexts_resource_split:
.. figure:: /_static/images/green_contexts_resource_split.png
   :alt: SM resource split using the cudaDevSmResourceSplit API
   :align: center

   使用 `cudaDevSmResourceSplit` 接口划分 SM 资源

请求异构切分时，需要为 ``result`` 中的每个资源指定 SM 数量（即对应 ``groupParams`` 条目中的 ``smCount`` 字段）。
该 SM 数量必须始终是 2 的倍数。
对于上图所示场景， ``groupParams[0].smCount`` 应为 ``X`` ， ``groupParams[1].smCount`` 应为 ``Y`` ，依此类推。
然而，如果应用程序使用了线程块簇，仅指定 SM 数量是不够的。
由于一个簇中的所有线程块都会被保证协同调度，因此用户还需要指定某个资源组应支持的最大簇大小（如有）。
这可以通过对应 ``groupParams`` 条目中的 ``coscheduledSmCount`` 字段来完成。
对于计算能力 10.0 及以上（CC 10.0+）的 GPU，簇还可以具有一个首选维度，该维度是其默认簇维度的倍数。
在受支持系统上进行单次 kernel 启动时，会尽可能使用此较大的首选簇维度，否则使用较小的默认簇维度。
用户可以通过对应 ``groupParams`` 条目中的 ``preferredCoscheduledSmCount`` 字段来表达此首选簇维度提示。
最后，在某些情况下，用户可能希望放宽 SM 数量要求并将更多可用 SM 拉入某个组中；
用户可以通过将对应 ``groupParams`` 条目的 ``flags`` 字段设置为其非默认标志值来表达此回填（backfill）选项。

.. _green-contexts-split-api-overview:

**cudaDevSmResourceSplit API 概览**

为提供更多灵活性， ``cudaDevSmResourceSplit`` 接口还提供了一种发现模式（discovery mode），
用于在一个或多个组的精确 SM 数量预先未知的情况。
例如，用户可能希望创建一个具有尽可能多 SM 的设备资源，同时满足某些协同调度要求（例如支持大小为 4 的簇）。
要启用此发现模式，用户可以将对应 ``groupParams`` 条目（或多个条目）的 ``smCount`` 字段设置为零。
在成功调用 ``cudaDevSmResourceSplit`` 之后， ``groupParams`` 的 ``smCount`` 字段将被填充一个有效的非零值；
我们将此值称为实际的 ``smCount`` 值。
如果 ``result`` 不为空（即非干运行），则 ``result`` 中对应的组也会将其 ``smCount`` 设置为相同的值。
``nbGroups`` 个 ``groupParams`` 条目的指定顺序至关重要，因为它们会从左（索引 0）到右（索引 ``nbGroups-1`` ）依次进行评估。

:numref:`table-split-api-overview` 提供了 ``cudaDevSmResourceSplit`` 接口所支持参数的高层概览。

.. _table-split-api-overview:
.. list-table:: ``cudaDevSmResourceSplit`` 切分接口概览
   :widths: 25 25 25 25
   :header-rows: 1
   :align: center

   * - 参数
     - 含义
     - 取值
     - 说明
   * - ``result``
     - 存放切分结果的数组
     - 探索性干运行时为 ``nullptr`` ；否则为非空指针
     - 成功时填充 ``nbGroups`` 个有效的 ``cudaDevResource`` 组
   * - ``nbGroups``
     - 组的数量
     - 有效的正整数
     - 切分后 ``groupParams`` 数组中条目的数量
   * - ``input``
     - 要切分的资源
     - 有效的 ``cudaDevResource`` SM 类型资源
     - 通常通过 ``cudaDeviceGetDevResource`` 获取
   * - ``remainder``
     - 剩余分区
     - 不需要剩余组时为 ``nullptr``
     - 剩余组没有 SM 数量或协同调度要求的约束
   * - ``flags``
     - 切分标志
     - 0
     - 当前仅支持默认值 0
   * - ``groupParams[i].smCount``
     - 对应组的 SM 数量
     - 0（发现模式）或其他有效 ``smCount`` 值
     - 控制对应组 ``result[i]`` 的 SM 数量
   * - ``groupParams[i].coscheduledSmCount``
     - 协同调度 SM 数量
     - 0（默认）或有效的协同调度 SM 数量
     - 影响簇支持，适用于计算能力 9.0+
   * - ``groupParams[i].preferredCoscheduledSmCount``
     - 首选协同调度 SM 数量（提示）
     - 0（默认）或有效的首选协同调度 SM 数量
     - 适用于计算能力 10.0+ 的首选簇维度功能
   * - ``groupParams[i].flags``
     - 组标志
     - 0（默认）或 ``cudaDevSmResourceGroupBackfill``
     - 控制是否允许 SM 回填到组中

.. note::

   ``cudaDevSmResourceSplit`` 接口的返回值取决于 ``result`` ：

   - ``result != nullptr`` ：仅当切分成功且创建了 ``nbGroups`` 个满足指定要求的有效 ``cudaDevResource`` 组时，该接口才返回 ``cudaSuccess`` ；否则返回错误。由于不同类型的错误可能返回相同的错误码（例如 ``CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`` ），建议在开发期间使用 ``CUDA_LOG_FILE`` 环境变量以获得更具信息量的错误描述。

   - ``result == nullptr`` ：即使某个组的最终 ``smCount`` 为零，该接口也可能返回 ``cudaSuccess`` ；而在 ``result`` 非空的情况下，这种情况本应返回错误。可将此模式视为一种干运行测试，可用于探索支持的功能，尤其是在发现模式中。

   - 当 ``result != nullptr`` 且调用成功时，生成的 ``result[i]`` 设备资源（i 取值范围为 ``[0, nbGroups)`` ）类型将为 ``cudaDevResourceTypeSm`` ，
     其 ``result[i].sm.smCount`` 或者是用户指定的非零 ``groupParams[i].smCount`` 值，或者是发现的值。
     在两种情况下， ``result[i].sm.smCount`` 都将满足以下所有约束：

     - 是 2 的倍数；并且
     - 在 ``[2, input.sm.smCount]`` 范围内；并且
     - ``(flags == 0) ? (是实际 group_params[i].coscheduledSmCount 的倍数) : (>= groups_params[i].coscheduledSmCount)``

   - 为 ``coscheduledSmCount`` 和 ``preferredCoscheduledSmCount`` 字段中的任意一个指定零，表示应使用这些字段的默认值；这些默认值会因 GPU 而异。
     这些默认值都等于通过 ``cudaDeviceGetDevResource`` 接口为指定设备检索到的 SM 资源的 ``smCoscheduledAlignment`` （而非任意 SM 资源）。
     要查看这些默认值，可以在成功调用 ``cudaDevSmResourceSplit`` 之后（初始将它们设置为 0），检查对应 ``groupParams`` 条目中更新后的值；参见下方代码。

   .. code-block:: c++

      int gpu_device_index = 0;
      cudaDevResource initial_GPU_SM_resources {};
      CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));
      std::cout << "Default value will be equal to " << initial_GPU_SM_resources.sm.smCoscheduledAlignment << std::endl;

      int default_split_flags = 0;
      cudaDevSmResourceGroupParams group_params_tmp = {.smCount=0, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0};
      CUDA_CHECK(cudaDevSmResourceSplit(nullptr, 1, &initial_GPU_SM_resources, nullptr /*remainder*/, default_split_flags, &group_params_tmp));
      std::cout << "coscheduledSmcount default value: " << group_params.coscheduledSmCount << std::endl;
      std::cout << "preferredCoscheduledSmcount default value: " << group_params.preferredCoscheduledSmCount << std::endl;

   - 如果存在剩余组，则其 SM 数量或协同调度要求没有任何约束。这需要由用户自行探索。

在详细介绍各个 ``cudaDevSmResourceGroupParams`` 结构体字段之前， :numref:`table-split-api-use-cases` 通过一些示例用例展示了这些值可能的情况。
假设 ``initial_GPU_SM_resources`` 设备资源已经按前面代码片段所示的方式进行了填充，且这就是将要被切分的资源。
表中的每一行都以此相同的起点开始。
为简化起见，该表仅展示 ``nbGroups`` 值以及每个用例对应的 ``groupParams`` 字段，这些值可应用于如下代码片段：

.. code-block:: c++

   int nbGroups = 2; // update as needed
   unsigned int default_split_flags = 0;
   cudaDevResource remainder {}; // update as needed
   cudaDevResource result_use_case[2] = {{}, {}}; // Update depending on number of groups planned. Increase size if you plan to also use a workqueue resource
   cudaDevSmResourceGroupParams group_params_use_case[2] = {{.smCount = X, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0},
                                                            {.smCount = Y, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0}}
   CUDA_CHECK(cudaDevSmResourceSplit(&result_use_case[0], nbGroups, &initial_GPU_SM_resources, remainder, default_split_flags, &group_params_use_case[0]));

.. _table-split-api-use-cases:
.. list-table:: ``cudaDevSmResourceSplit`` 切分接口用例
   :widths: 8 35 10 12 8 12 18 18 12
   :header-rows: 1
   :align: center

   * - 用例 #
     - 目标 / 用例
     - nbGroups
     - remainder
     - i
     - ``smCount``
     - ``coscheduledSmCount``
     - ``preferredCoscheduledSmCount``
     - ``flags``
   * - 1
     - | 一个具有 16 个 SM 的资源。
       | 不关心剩余的 SM。
       | 可能使用簇。
     - 1
     - nullptr
     - 0
     - 16
     - 0
     - 0
     - 0
   * - 2a
     - | 一个具有 16 个 SM 的资源，
       | 一个具有其余所有 SM 的资源。
       | 不使用簇。
       | （注：展示两个选项。在选项 (2a) 中，第 2 个资源是 remainder；在选项 (2b) 中，它是 ``result_use_case[1]`` ）
     - 1
     - 非 nullptr
     - 0
     - 16
     - 2
     - 2
     - 0
   * - 2b
     - （续上一用例的另一选项）
     - 2
     - nullptr
     - 0
     - 16
     - 2
     - 2
     - 0
   * - 2b
     - （续上一行：第二个组启用回填）
     - 2
     - nullptr
     - 1
     - 0
     - 2
     - 2
     - ``cudaDevSmResourceGroupBackfill``
   * - 3
     - | 两个资源，分别具有 28 和 32 个 SM。
       | 将使用大小为 4 的簇。
     - 2
     - nullptr
     - 0
     - 28
     - 4
     - 4
     - 0
   * - 3
     - （续上一行：第二个组）
     - 2
     - nullptr
     - 1
     - 32
     - 4
     - 4
     - 0
   * - 4
     - | 一个具有尽可能多 SM 的资源，能运行大小为 8 的簇，以及一个剩余分区。
     - 1
     - 非 nullptr
     - 0
     - 0
     - 8
     - 8
     - 0
   * - 5
     - | 一个具有尽可能多 SM 的资源，能运行大小为 4 的簇，以及一个具有 8 个 SM 的资源。
       | （注：顺序很重要！更改 ``groupParams`` 数组中条目的顺序可能意味着没有 SM 留给 8-SM 组）
     - 2
     - nullptr
     - 0
     - 8
     - 2
     - 2
     - 0
   * - 5
     - （续上一行：第二个组使用发现模式）
     - 2
     - nullptr
     - 1
     - 0
     - 4
     - 4
     - 0

.. note::

   上表中，用例 2b 的第二个组（ i=1 ） ``smCount`` 设为 0 表示发现模式，实际可用 SM 数量将在切分成功后填充；
   ``flags`` 设为 ``cudaDevSmResourceGroupBackfill`` 表示该组可以容纳超过请求的 SM 数量。

.. _green-contexts-groupparams-fields:

**关于 ``cudaDevSmResourceGroupParams`` 各字段的详细信息**

``smCount`` ：

- 控制 ``result`` 中对应组的 SM 数量。

- 取值：0（发现模式）或有效的非零值（非发现模式）。

- 有效的非零 ``smCount`` 值要求：
  ``(2 的倍数) 且在 [2, input->sm.smCount] 范围内
  且 ((flags == 0) ? 是实际 coscheduledSmCount 的倍数 : 大于等于 coscheduledSmCount)``

- 用例：在 SM 数量未知 / 不固定时使用发现模式探索可能性；使用非发现模式请求特定数量的 SM。

- 注意：在发现模式下，使用非空 ``result`` 成功调用切分接口后，实际的 SM 数量将满足有效的非零值要求。

``coscheduledSmCount`` ：

- 控制组合在一起（"协同调度"）的 SM 数量，以支持在计算能力 9.0+ 上启动不同的簇。因此，它可能影响生成组中的 SM 数量以及它们能支持的簇大小。

- 取值：0（当前架构的默认值）或有效的非零值。

- 有效的非零值要求： ``(2 的倍数)`` 直到上限。

- 用例：为簇使用默认值或手动选择的值，同时考虑给定架构上的最大可移植簇大小。如果代码不使用簇，可以使用最小支持值 2 或默认值。

- 注意：使用默认值时，成功调用切分后，实际的 ``coscheduledSmCount`` 也将满足有效的非零值要求。
  如果 ``flags`` 不为零，生成的 ``smCount`` 将 ``>= coscheduledSmCount`` 。
  可将 ``coscheduledSmCount`` 视为对有效生成组提供某些保证的底层 "结构"
  （即在最坏情况下，该组至少可以运行一个 ``coscheduledSmCount`` 大小的簇）。
  这种结构保证不适用于剩余组；在那里需要由用户自行探索可启动的簇大小。

``preferredCoscheduledSmCount`` ：

- 作为提示，提示驱动程序在可能的情况下将实际 ``coscheduledSmCount`` 个 SM 的组合并为
  ``preferredCoscheduledSmCount`` 大小的更大组。
  这样做可以让代码利用计算能力 10.0 及以上设备上提供的首选簇维度功能。
  参见 ``cudaLaunchAttributeValue::preferredClusterDim`` 。

- 取值：0（当前架构的默认值）或有效的非零值。

- 有效的非零值要求： ``(实际 coscheduledSmCount 的倍数)``

- 用例：如果使用首选簇并且位于计算能力 10.0（Blackwell）或更高版本的设备上，
  则使用大于 2 的手动选择值。
  如果不使用簇，则选择与 ``coscheduledSmCount`` 相同的值：选择最小支持值 2，或将两者都设置为 0。

- 注意：使用默认值时，成功调用切分后，实际的 ``preferredCoscheduledSmCount`` 也将满足有效的非零值要求。

``flags`` ：

- 控制组的最终 SM 数量是实际协同调度 SM 数量的倍数（默认）还是可以将 SM 回填到该组中（backfill）。
  在回填情况下，最终的 SM 数量（ ``result[i].sm.smCount`` ）将大于或等于指定的 ``groupParams[i].smCount`` 。

- 取值：0（默认）或 ``cudaDevSmResourceGroupBackfill``

- 用例：使用零（默认），使生成的组具有支持多个 ``coScheduledSmCount`` 大小簇的保证灵活性。如果希望在组中获取尽可能多的 SM，请使用回填选项；其中一些 SM（回填的那些）不提供任何协同调度保证。

- 注意：使用回填标志创建的组仍然可以支持簇（例如，保证至少支持一个 ``coscheduledSmCount`` 大小）。

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

工作队列并发限制为 4，向驱动程序提示用户期望最多 4 个并发流有序工作负载。
驱动程序会尝试在可能的情况下尊重此提示来分配工作队列。
.. _green-contexts-create-resource-desc:

4.6.4.4. 步骤 3：创建资源描述符
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

分割 SM 资源（以及可选地添加工作队列配置资源）后，下一步是使用 ``cudaDevResourceGenerateDesc`` 接口为预计可供 GC 使用的所有资源生成资源描述符。

相关的 CUDA Runtime 接口函数定义如下：

.. code-block:: c++

   cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t *phDesc, cudaDevResource *resources, unsigned int nbResources)

可以将多个 ``cudaDevResource`` 资源组合在一起。例如，下面的代码片段展示了如何生成一个封装了三组资源的资源描述符。
只需确保这些资源在 ``resources`` 数组中是连续分配的即可。

.. code-block:: c++

   cudaDevResource actual_split_result[5] = {};
   // 填充 actual_split_result 的代码未显示

   // 生成封装 3 个资源的描述符：actual_split_result[2] 到 [4]
   cudaDevResourceDesc_t resource_desc;
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc, &actual_split_result[2], 3));

也支持组合不同类型的资源。例如，可以生成一个同时包含 SM 和工作队列资源的描述符。

要使 ``cudaDevResourceGenerateDesc`` 调用成功，需满足以下条件：

- 所有 ``nbResources`` 个资源必须属于同一 GPU 设备。

- 如果组合多个 SM 类型资源，它们必须由同一次切分接口调用生成，并且（如果不在剩余分区中）具有相同的 ``coscheduledSmCount`` 值。

- 最多只能存在一个工作队列配置或工作队列类型的资源。

.. _green-contexts-create-green-ctx:

4.6.4.5. 步骤 4：创建 Green Context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最后一步是使用 ``cudaGreenCtxCreate`` 接口从资源描述符创建 GC。
该 GC 只能访问创建时所用资源描述符中封装的资源（如 SM、工作队列）。
这些资源将在这一步中被分配（provision）。

相关的 CUDA Runtime 接口函数定义如下：

.. code-block:: c++

   cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t *phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags)

``flags`` 参数应设置为 0。
还建议在创建 GC 之前，通过 ``cudaInitDevice`` 接口或 ``cudaSetDevice`` 接口显式初始化设备的主上下文（ ``cudaSetDevice`` 同时会将主上下文设置为调用线程的当前上下文）。
这样做可以确保在创建 GC 时不会产生额外的主上下文初始化开销。

参见下方代码片段：

.. code-block:: c++

   int current_device = 0; // assume single GPU
   CUDA_CHECK(cudaSetDevice(current_device)); // Or cudaInitDevice

   cudaDevResourceDesc_t resource_desc {};
   // Code to generate resource_desc not shown

   // Create a green_ctx on GPU with current_device ID with access to resources from resource_desc
   cudaExecutionContext_t green_ctx {};
   CUDA_CHECK(cudaGreenCtxCreate(&green_ctx, resource_desc, current_device, 0));

GC 创建成功后，用户可以通过对该执行上下文调用 ``cudaExecutionCtxGetDevResource`` 来验证其资源（针对每种资源类型）。

.. _green-contexts-create-multiple-green-ctxs:

**创建多个 Green Context**

一个应用程序可以拥有多个 GC，此时需要重复上述部分步骤。
对于大多数用例，每个 GC 都会有一个独立的、互不重叠的已分配 SM 集合。
例如，对于 5 个同构 ``cudaDevResource`` 组（ ``actual_split_result`` 数组）的情况，
一个 GC 的描述符可能封装了 ``actual_split_result[2]`` 到 ``[4]`` 的资源，
而另一个 GC 的描述符可能封装了 ``actual_split_result[0]`` 到 ``[1]`` 的资源。
在这种情况下，某个特定的 SM 只会被分配给应用程序两个 GC 中的一个。

但 SM 超额分配（oversubscription）也是可能的，并且在某些情况下会被使用。
例如，可以让第二个 GC 的描述符封装 ``actual_split_result[0]`` 到 ``[2]`` 。
在这种情况下， ``actual_split_resource[2]`` 的所有 SM 都将被超额分配，即同时为两个 GC 所拥有；
而 ``actual_split_resource[0]`` 到 ``[1]`` 以及 ``actual_split_resource[3]`` 到 ``[4]`` 的资源可能仅被两个 GC 中的一个使用。
SM 超额分配应根据具体情况谨慎使用。

.. _green-contexts-launching-work:

4.6.5. Green Contexts：启动工作
-------------------------------------

要启动针对前述步骤创建的 GC 的 kernel，首先需要使用 ``cudaExecutionCtxStreamCreate`` 接口为该 GC 创建一个流。
在该流上使用 ``<<< >>>`` 语法或 ``cudaLaunchKernel`` 接口启动 kernel，将确保该 kernel 只能使用该流通过其执行上下文获得的资源（SM、工作队列）。
例如：

.. code-block:: c++

   // Create green_ctx_stream CUDA stream for previously created green_ctx green context
   cudaStream_t green_ctx_stream;
   int priority = 0;
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&green_ctx_stream,
                                           green_ctx,
                                           cudaStreamDefault,
                                           priority));

   // Kernel my_kernel will only use the resources (SMs, work queues, as applicable) available to green_ctx_stream's execution context
   my_kernel<<<grid_dim, block_dim, 0, green_ctx_stream>>>();
   CUDA_CHECK(cudaGetLastError());

鉴于 ``green_ctx`` 是一个 GC ，传递给上述流创建接口的默认流创建标志等效于 ``cudaStreamNonBlocking`` 。

.. _green-contexts-cuda-graphs:

**CUDA 图**

对于作为 CUDA 图（参见 :doc:`/04-special-topics/02-cuda-graphs` ）一部分启动的 kernel，还有一些更细微的注意事项。
与 kernel 不同，CUDA 图所启动的 CUDA 流并不决定所使用的 SM 资源，因为该流仅用于依赖关系跟踪。

kernel 节点（以及其他适用的节点类型）将执行的执行上下文是在节点创建期间设置的。
如果 CUDA 图将通过流捕获（stream capture）创建，则参与捕获的流的执行上下文将决定相关图节点的执行上下文。
如果图将通过图 API 创建，则用户应为每个相关节点显式设置执行上下文。
例如，要添加一个 kernel 节点，用户应使用多态的 ``cudaGraphAddNode`` 接口，
类型为 ``cudaGraphNodeTypeKernel`` ，并显式指定 ``.kernel`` 下 ``cudaKernelNodeParamsV2`` 结构体的 ``.ctx`` 字段。
``cudaGraphAddKernelNode`` 不允许用户指定执行上下文，因此应避免使用。
请注意，一个图中的不同图节点可以属于不同的执行上下文。

为了进行验证，可以使用 Nsight Systems 的节点追踪模式（ ``--cuda-graph-trace node`` ）来观察特定图节点将在哪些 GC 上执行。
注意，在默认的图追踪模式下，整个图将显示在它所启动的流的 GC 之下，但正如前文所解释的，这并不能提供关于各个图节点执行上下文的任何信息。

要以编程方式验证，可以使用 CUDA 驱动接口 ``cuGraphKernelNodeGetParams(graph_node, &node_params)`` ，
并将 ``node_params.ctx`` 上下文句柄字段与该图节点的预期上下文句柄进行比较。
由于 ``CUgraphNode`` 和 ``cudaGraphNode_t`` 可以互换使用，因此使用驱动接口是可行的，但用户需要包含相关的 ``cuda.h`` 头文件并直接与驱动程序链接（ ``-lcuda`` ）。

.. _green-contexts-thread-block-clusters:

**线程块簇**

带有线程块簇（参见 :ref:`thread-block-clusters` ）的 kernel 可以像其他任何 kernel 一样在 GC 流上启动，从而使用该 GC 分配的资源。
:ref:`green-contexts-split-sm-resources` 节展示了在切分设备资源时如何指定需要协同调度的 SM 数量以支持簇。
但与任何使用簇的 kernel 一样，用户应使用相关的占用率接口来确定 kernel 的最大潜在簇大小（通过 ``cudaOccupancyMaxPotentialClusterSize`` ），
以及在需要时确定最大活跃簇数量（通过 ``cudaOccupancyMaxActiveClusters`` ）。
如果用户将 GC 流指定为相关 ``cudaLaunchConfig`` 的 ``stream`` 字段，
则这些占用率接口将考虑为该 GC 分配的 SM 资源。
此用例对于可能会从用户那里获得 GC CUDA 流的库尤为相关，在 GC 由剩余设备资源创建的情况下也同样如此。

下面的代码片段展示了如何使用这些接口。

.. code-block:: c++

   // Assume cudaStream_t gc_stream  has already been created and a __global__ void cluster_kernel exists.

   // Uncomment to support non portable cluster size, if possible
   // CUDA_CHECK(cudaFuncSetAttribute(cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1))

   cudaLaunchConfig_t config = {0};
   config.gridDim          = grid_dim; // has to be a multiple of cluster dim.
   config.blockDim         = block_dim;
   config.dynamicSmemBytes = expected_dynamic_shared_mem;

   cudaLaunchAttribute attribute[1];
   attribute[0].id = cudaLaunchAttributeClusterDimension;
   attribute[0].val.clusterDim.x = 1;
   attribute[0].val.clusterDim.y = 1;
   attribute[0].val.clusterDim.z = 1;
   config.attrs = attribute;
   config.numAttrs = 1;

   config.stream=gc_stream; // Need to pass the CUDA stream that will be used for that kernel

   int max_potential_cluster_size = 0;
   // the next call will ignore cluster dims in launch config
   CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_potential_cluster_size, cluster_kernel, &config));
   std::cout << "max potential cluster size is " << max_potential_cluster_size << " for CUDA stream gc_stream" << std::endl;

   // Could choose to update launch config's clusterDim with max_potential_cluster_size.
   // Doing so would result in a successful cudaLaunchKernelEx call for the same kernel and launch config.

   int num_clusters= 0;
   CUDA_CHECK(cudaOccupancyMaxActiveClusters(&num_clusters, cluster_kernel, &config));
   std::cout << "Potential max. active clusters count is " << num_clusters << std::endl;

.. _green-contexts-verify-use:

**验证 Green Contexts 的使用**

除了通过经验观察 GC 资源分配对 kernel 执行时间的影响之外，用户还可以利用 Nsight Systems 或 Nsight Compute CUDA 开发者工具在一定程度上验证 GC 的使用是否正确。

例如，在属于不同 GC 的 CUDA 流上启动的 kernel 会显示在 Nsight Systems 报告中 CUDA HW 时间线下不同的 Green Context 行下。
Nsight Compute 在其 Session 页面提供了 Green Context Resources 概览，以及 Details 部分中 Launch Statistics 下更新的 # SMs 信息。
前者提供了已分配资源的可视化位掩码。如果应用程序使用不同的 GC，这特别有用，因为用户可以确认 GC 之间预期的重叠情况
（无重叠，或者如果 SM 被超额分配，则预期存在非零重叠）。

:numref:`fig-green_contexts_ncu_mask` 描绘了一个示例中两个分别分配了 112 个和 16 个 SM 的 GC 的资源情况，两者之间没有 SM 重叠。
提供的视图可以帮助用户验证每个 GC 分配的 SM 资源数量。它还有助于确认没有 SM 被超额分配，因为没有方框在两个 GC 中同时被标记为绿色（即分配给该 GC）。

.. _fig-green_contexts_ncu_mask:
.. figure:: /_static/images/green_contexts_ncu_mask.png
   :alt: Green contexts resources section from Nsight Compute
   :align: center

   Nsight Compute 中的 Green Contexts 资源部分

Launch Statistics 部分还明确列出了为该 GC 分配的 SM 数量，因此该 kernel 可以使用这些 SM。
请注意，这些是 kernel 在其执行期间可以访问的 SM，而不是该 kernel 实际运行的 SM 数量。前面显示的资源概览也同样如此。
kernel 实际使用的 SM 数量取决于多种因素，包括 kernel 本身（启动几何等）、GPU 上同时运行的其他工作等。

.. _green-contexts-apis:

4.6.6. Green Contexts：API
--------------------------

本节介绍一些额外的 GC 接口。如需完整列表，请参阅相关的 CUDA Runtime API 文档章节。

对于使用 CUDA 事件进行同步，可以利用 ``cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event)``
和 ``cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event)`` 接口。

``cudaExecutionCtxRecordEvent`` 记录一个 CUDA 事件，捕获该调用时指定执行上下文的所有工作 / 活动；
而 ``cudaExecutionCtxWaitEvent`` 使提交给该执行上下文的所有未来工作等待指定事件中捕获的工作完成。

如果执行上下文有多个 CUDA 流，使用 ``cudaExecutionCtxRecordEvent`` 比使用 ``cudaEventRecord`` 更方便。
要达到等效行为而不使用此执行上下文接口，需要在每个执行上下文流上通过 ``cudaEventRecord`` 分别记录一个 CUDA 事件，然后让依赖工作分别等待所有这些事件。
类似地，如果需要所有执行上下文流等待一个事件完成，使用 ``cudaExecutionCtxWaitEvent`` 比使用 ``cudaStreamWaitEvent`` 更方便。
替代方案是对该执行上下文中的每个流分别调用一次 ``cudaStreamWaitEvent`` 。

对于 CPU 端的阻塞同步，可以使用 ``cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx)`` 。
该调用将阻塞，直到指定的执行上下文完成其所有工作。
如果指定的执行上下文不是通过 ``cudaGreenCtxCreate`` 创建的，而是通过 ``cudaDeviceGetExecutionCtx`` 获取的（即设备的主上下文），调用该函数还将同步在同一设备上创建的所有 GC 。

要检索给定执行上下文所关联的设备，可以使用 ``cudaExecutionCtxGetDevice`` 。
要检索给定执行上下文的唯一标识符，可以使用 ``cudaExecutionCtxGetId`` 。

最后，显式创建的执行上下文可以通过 ``cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx)`` 接口销毁。

.. _green-contexts-example:

4.6.7. Green Contexts：示例
----------------------------

本节通过示例说明 GC 如何使关键工作能够更早启动并完成。
与 :ref:`green-contexts-motivation` 节中使用的场景类似，该应用程序有两个 kernel ，将运行在两个不同的非阻塞 CUDA 流上。
从 CPU 端看到的时间线如下：首先在一个长时间运行的 kernel （ ``delay_kernel_us`` ，在整块 GPU 上需要执行多波次）在 CUDA 流 ``strm1`` 上启动。
然后经过短暂的等待时间（短于 kernel 的运行时长）之后，一个较短但关键的 kernel （ ``critical_kernel`` ）在流 ``strm2`` 上启动。
两个 kernel 的 GPU 运行时长以及从 CPU 启动到完成的时间都会被测量。

作为长时间运行 kernel 的代理，使用了一个延迟 kernel ，其中每个线程块运行固定的微秒数，并且线程块数量超过 GPU 可用的 SM 数量。

初始时不使用 GC ，但关键 kernel 启动在一个比长时间运行 kernel 更高优先级的 CUDA 流上。
由于其所在流具有更高优先级，关键 kernel 可以在长时间运行 kernel 的部分线程块完成时立即开始执行。
然而，它仍需等待一些可能长时间运行的线程块完成，这会延迟其执行开始时间。

:numref:`fig-green_contexts_nsys_example_no_GCs_with_prio` 在 Nsight Systems 报告中展示了此场景。
长时间运行的 kernel 启动在流 13 上，而短而关键的 kernel 启动在流 14 上（具有更高的流优先级）。
如图像中高亮所示，关键 kernel 在能够开始执行之前需要等待 0.9ms（在本例中）。
如果两个流具有相同优先级，关键 kernel 的执行会延后更多。

.. _fig-green_contexts_nsys_example_no_GCs_with_prio:
.. figure:: /_static/images/green_contexts_nsys_example_no_GCs_with_prio.png
   :alt: Nsight Systems timeline without green contexts
   :align: center

   不使用 Green Contexts 的 Nsight Systems 时间线

为利用 GC 功能，创建了两个 GC ，每个分配有独立且互不重叠的 SM 集合。
在本例中，对于具有 132 个 SM 的 H100 ，出于演示目的，选择了 16 个 SM 用于关键 kernel （Green Context 3），112 个 SM 用于长时间运行的 kernel （Green Context 2）。
如 :numref:`fig-green_contexts_nsys_example_w_GCs` 所示，关键 kernel 现在几乎可以瞬时启动，因为存在只有 Green Context 3 才能使用的 SM。

.. _fig-green_contexts_nsys_example_w_GCs:
.. figure:: /_static/images/green_contexts_nsys_example_w_GCs.png
   :alt: Nsight Systems timeline with green contexts
   :align: center

   使用 Green Contexts 的 Nsight Systems 时间线

与单独运行时相比，短 kernel 的运行时长可能会增加，因为现在它能使用的 SM 数量受到限制。
长时间运行的 kernel 也是如此，它不再能使用 GPU 的所有 SM ，而是受到其 GC 分配资源的约束。
然而，关键结果是，关键 kernel 工作现在可以比之前显著更早地启动并完成。
当然，这要排除其他限制因素，正如前文所述，并行执行是无法保证的。

在所有情况下，具体的 SM 切分方案应根据具体情况通过实验来决定。

.. note::

   有关 Green Contexts 的更多详细信息，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html>`_。
