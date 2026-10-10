.. _compute-fabric-transport:

4.18. Compute Fabric Transport
===========================================

Compute Fabric Transport （计算网络传输， CFT ） 基于 NVLink Fabric 提供大规模 GPU 间通信能力，
可为 NCCL、NVSHMEM 等通信库提供底层支撑
（通过 `NCCL Device API <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html>`_
和 `NVSHMEM <https://docs.nvidia.com/nvshmem/api/index.html>`_ 接口实现）。
当这些库能够满足所需的通信模式时，大多数开发者应优先选用它们。
CFT 主要面向以下两类场景：一是通信库本身的开发者；二是现有库无法良好匹配其通信模式、需要细粒度控制的应用程序开发者。
此外，自定义通信算子的开发者也可以将 Fabric 操作与 NCCL/NVSHMEM Device API 结合使用，以获得更直接的硬件访问能力。

:ref:`单播 <virtual-memory-management-unicast>` 和 :ref:`多播 <virtual-memory-management-multicast>` 内存共享 API 允许应用程序通过将远程物理内存映射到本地虚拟地址空间，
来实现跨 NVLink Fabric 的 GPU 内存访问。
一旦完成对端内存分配的映射并授予相应访问权限， kernel 即可像访问本地内存一样，通过普通的加载/存储（Load/Store）指令来访问该内存（对于组播映射，还可使用 ``multimem`` 指令）。
该模型既 **以地址为中心（address-centric）** 又 **以内存为中心（memory-centric）** ：其共享的基本单元是虚拟地址到物理地址的映射关系，
并且对端 kernel 访问的每一个字节，都必须落在发起进程已预留并映射的地址范围内。

在大规模 NVLink fabric 上，该模型有两个局限，并且两者都会影响应用程序的编写方式。

虚拟地址不提供错误传播通道。
当 kernel 对对端地址发起加载（load）或存储（store）操作时，硬件只会返回数据或触发内存故障——不存在任何中间状态供发起线程检查并以编程方式处理失败。
在大规模互联网络（fabric）系统中，数据包丢失或链路重置等瞬态错误是可能发生的；当系统规模足够大时，这些错误便成为构建高弹性应用时必须考虑的因素。
在远程访问过程中遇到的网络互联错误，通常会导致该次访问触发故障或直接终止进程，而 kernel 对此类故障无能为力：既无法重试，也无法重新路由。

虚拟地址仅能用于寻址内存。映射关系是基于物理内存定义的，因此网络上不属于内存类型的资源，完全无法通过这种映射方式访问。

CFT 为通过 NVLink 互连的 GPU 提供了一种互补的、 **以资源为中心（resource-centric）** 的模型。
应用程序不再将远端内存导入本地地址空间，而是创建一个 **逻辑端点（logical endpoint）** ：一个命名的传输对象，代表网络上可达的某个资源。
逻辑端点由一个 32 位整数（即 **逻辑端点 id（logical endpoint id）** ）标识，其内部的目标则通过该 ID 加上一个 64 位偏移量来命名，而非使用虚拟地址。
由于该 ID 直接命名的是资源本身，因此该模型并不局限于内存，尽管目前逻辑端点所暴露的资源类型仅有内存一种。

即使端点的所有者更改了该端点的底层内存，已导入的端点依然有效。
对端导入的是端点本身，将其与自身预留的一个端点 ID 关联，而并非导入所有者的具体内存分配。
因此，所有者可以通过在本地重新绑定来扩展、收缩或替换端点背后的缓冲区，而对端可以继续使用其本地的端点 ID，无需重新导入。
不过，所有者与其对端仍需在重新绑定期间进行同步，以防止对端在所有者将端点重新绑定到新内存的过程中访问该端点。
此外，如果发起的操作落在当前绑定范围之外，则属于未定义行为。
相比之下，VMM IPC 导入映射的是特定的内存分配，因此若所有者调整该分配的大小，对端就必须重新导入并映射新的分配。

一旦端点被创建、共享给需要它的对端，并绑定了底层资源，就可以针对对端的逻辑端点 ID 和偏移量发起异步网络操作，
包括 put（推送/写入）、get（拉取/读取）、原子操作（atomics）和归约操作（reductions）。
与对已映射的对端地址进行 load 或 store 不同，网络操作会报告一个显式的完成状态，供发起操作的代码检查。
在发生故障时，应用程序可以进行重试、重新路由或采取其他恢复措施，而不会导致进程崩溃。


.. _prerequisites-and-scope:
.. _cft-prerequisites-scope:

4.18.1. 前提与范围
------------------

使用逻辑端点有以下要求：

- **CUDA 驱动 API。** 逻辑端点 API 是底层 :ref:`CUDA Driver API <driver-api>` 的一部分。
- **NVLink fabric 连通性与设备支持。** 参与的 GPU 必须能够通过 NVLink fabric 相互到达。
  设备是否支持逻辑端点取决于其体系结构、驱动程序和系统配置，
  因此应用程序应当通过查询 :ref:`第 4.18.2.2 节 <cft-query-for-support>` 中描述的设备属性来确定支持情况，而不是根据现有硬件进行推断。
- **IMEX 守护进程与通道。** 在进程之间共享端点使用 fabric IPC 句柄，这要求 NVIDIA IMEX 守护进程处于运行状态且已配置 IMEX 通道。

端点可以与同一节点上的进程，或同一 NVLink 网络内不同节点上的进程共享。
其网络 IPC 句柄机制与 VMM API 类似。
有关网络句柄和 IMEX 通道的详细信息，请参阅 :ref:`虚拟内存管理 <virtual-memory-management-details>` 。

.. _preliminaries:

4.18.2. 预备知识
----------------

以下定义确立了本节其余部分使用的关键对象和操作。


.. _definitions:

4.18.2.1. 定义
^^^^^^^^^^^^^^

**逻辑端点 id（Logical Endpoint Id）：** ``CUlogicalEndpointId`` 是一个 32 位值，用于命名进程内的一个逻辑端点。
端点 id 是进程本地的：每个进程使用 ``cuLogicalEndpointIdReserve`` 保留自己的 id 范围，并选择将哪个保留的 id 与给定端点相关联。
fabric 操作通过该 id 加上一个偏移来寻址端点。
单个端点 id 命名一个端点，但多个 id 可以命名同一个端点。
端点 id 通过 ``cuLogicalEndpointCreate`` 或 ``cuLogicalEndpointImport`` 与端点相关联。
在用 ``cuLogicalEndpointIdRelease`` 释放端点 id 之前，必须先用 ``cuLogicalEndpointDestroy`` 移除这种关联。
在端点 id 仍与某个端点保持关联时释放它属于未定义行为。
之后的保留操作可能会重用已释放的 id。
:ref:`第 4.18.6 节 <cft-logical-endpoint-ids>` 描述逻辑端点 id 的生命周期。

**逻辑端点（Logical Endpoint）：** 逻辑端点是一个传输对象，表示可通过 NVLink fabric 到达的资源，并以一个有界的偏移空间的形式暴露。
创建端点会分配用于路由寻址到它的 fabric 操作的资源。
随后，应用程序必须显式地将目标资源绑定到端点偏移空间的一个或多个范围。
fabric 操作使用相关联的端点 id 加上该空间内的一个偏移来命名其目标。
当进程使用 ``cuLogicalEndpointCreate`` 创建端点或使用 ``cuLogicalEndpointImport`` 导入对等端点时，端点会与一个保留的 id 相关联。
在显式解绑端点的资源之前销毁该端点关联，会解绑这些资源，但不会销毁资源本身。
在资源尚未从端点解绑之前就销毁它们属于未定义行为。
:ref:`第 4.18.3 节 <cft-api-overview>` 概述了端点的生命周期。

**单播逻辑端点（Unicast Logical Endpoint）：** 单播逻辑端点表示一个点对点传输目标。
它有一个单一的所有者设备，在创建端点时指定，端点绑定的内存就位于该设备上。
对等设备针对该端点 id 发起 put 或 get 操作，将数据移入或移出所有者绑定的内存。

**多播逻辑端点（Multicast Logical Endpoint）：** 多播逻辑端点表示一个在每个偏移处暴露多个资源的传输目标。
在绑定自己的资源之前，每个参与的进程必须先将该资源添加到多播端点。
然后该进程将资源绑定到端点。
这些绑定共同实现为每个团队成员创建一个副本。
以多播端点为目标的 fabric 操作会访问每个副本中对应的偏移。
多播逻辑端点是 :ref:`多播内存共享 <virtual-memory-management-multicast>` 中描述的 VMM 多播对象在 CFT 中的对应物。

**CFT 句柄（Compute Fabric Transport Handle）：** CFT 句柄是传统指针的一种以资源为中心的替代方案。
它由一对逻辑端点 id 和偏移组成：逻辑端点 id 选择 NVLink fabric 上的一个目的地——单播端点的所有者设备或一个多播团队——而偏移则选择该目的地绑定内存中的一个位置。
fabric 操作使用这种句柄来代替对等虚拟地址。

**逻辑端点 clique（Logical Endpoint Clique）：** 逻辑端点 clique 是一个动态的 GPU 组，这些 GPU 在要求的能力级别上可以通过 CFT 操作相互到达。
*clique 类型（clique type）* 描述所需的能力级别。
逻辑端点 clique 有两种类型： ``CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT`` 和 ``CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT`` 。
*clique id（clique id）* 是一个命名该动态 GPU 组的 32 位值。
试图将逻辑端点导入到一个与导出者设备不属于同一 clique 成员的设备上将会失败。
试图通过添加不属于同一 clique 的设备来创建多播逻辑端点将会失败。
单个设备可以是多个 clique 的成员。
程序可以使用 ``cuDeviceGetCliqueCount`` 和 ``cuDeviceGetCliqueInfo`` 查询某个设备所属的 clique 。

**端点绑定（Endpoint Binding）：** 端点绑定是逻辑端点偏移空间中的一个范围与应用程序提供的物理内存分配之间的关联。
应用程序可以使用 ``cuLogicalEndpointBindAddr`` 通过已映射的虚拟地址绑定内存，或使用 ``cuLogicalEndpointBindMem`` 通过分配句柄绑定内存。
fabric 操作仅对已绑定的端点范围有效，并且绑定必须满足端点的对齐和大小限制。
参见 :ref:`第 4.18.8 节 <cft-binding-memory>` 和 :ref:`第 4.18.6.1 节 <cft-limits-alignment>` 。

所有 fabric 操作都是异步的：发起一个操作只是启动传输。
它们的完成情况和状态通过每个操作提供的完成机制显式报告（参见 :ref:`第 4.18.10 节 <cft-completion-status>` ）。

**Put 与 Get 操作：** put 和 get 是 CFT 提供的数据移动原语。
put 操作将数据从发起方 GPU 移动到目标逻辑端点。
其 multimem 变体 ``cuda::ptx::fabric_try_put_multimem`` 向多播逻辑端点执行广播，将源数据写入每个副本中对应的偏移。
get 操作将数据从目标逻辑端点移动到发起方 GPU 。
在两个方向上，远程一侧都由逻辑端点 id 和偏移来标识，而不是由对等虚拟地址来标识，并且远程分配不需要被映射到发起进程的虚拟地址空间中。

**规约与拉取规约操作：** 规约操作（ *red* ）使用规约运算符将来自发起方 GPU 的数据合并到目标逻辑端点中。
拉取规约操作（ *pull_red* ）从目标多播逻辑端点读取数据，使用规约运算符合并多播团队各副本之间的值，并将结果返回给发起方 GPU 。
它是 ``multimem.ld_reduce`` 在 fabric 中的对应物。

**原子操作：** 原子操作（ *atom* ）对目标逻辑端点执行原子的读-改-写。
它读取原始值，用新值将其覆盖，并返回原始值。
相比之下，规约操作不返回原始值。
用于计算新值的运算符包括 ``add`` 、 ``min`` 、 ``max`` 、按位 ``and`` 、 ``or`` 和 ``xor`` 。
交换（exchange）操作存储用户提供的值，而比较并交换（compare-and-swap）仅在目标持有期望的比较值时才存储新值。

.. _cft-counted-operation:

**计数操作（Counted Operation）：** 计数操作是一种 fabric 操作，随着数据到达，它按写入目的地的字节数递增一个目标计数器。
接收方在消费数据之前会等待计数器达到 *期望字节数（expected byte count）* 。
在达到该期望字节数之前，接收方并不知道哪些字节已被写入。
一旦计数器达到期望字节数，接收方就可以安全地消费数据。
计数器必须在端点内存中按 256 字节边界对齐。
创建端点时必须启用计数操作支持（ ``CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS`` ），
并且应用程序在使用该能力之前必须查询设备支持情况（ ``CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED`` ）。

**完成状态与错误报告：** fabric 操作由一个 *完成对象（completion object）* 跟踪，它在操作完成时发出信号并记录该操作的状态。
等待该对象可以表明被跟踪的操作是成功完成还是发生了 fabric 错误。
当报告失败时，应用程序可以查询详细的逐操作错误状态，然后进行重试、重新路由或以其他方式恢复，而不是让进程出错。

由于 fabric 操作的更新是乱序交付的，观察到某一个目的地位置已被写入——例如目的地区的最后一个字节——并不意味着传输的其余部分已经落地。
观察到每一个目的地位置都已被写入，也并不表示操作已成功完成、目的地资源可以被重用。
例如，如果响应被丢弃，生产者可能会收到错误状态，从而导致其重试该操作。
因此，应用程序必须等待完成并检查所报告的状态，然后才能将操作视为完成。
参见 :ref:`第 4.18.10 节 <cft-completion-status>` 。


.. _query-for-support:
.. _cft-query-for-support:

4.18.2.2. 支持查询
^^^^^^^^^^^^^^^^^^

应用程序在使用逻辑端点 API 之前应当查询支持情况，因为可用性取决于 GPU 体系结构、驱动程序和系统配置。
以下设备属性描述了相关能力。
这些特性相互独立，因此应用程序应当只查询其打算使用的那些特性。

**单播逻辑端点支持**

查询设备是否支持单播逻辑端点：

.. code-block:: cpp

   int unicastSupported = 0;
   cuDeviceGetAttribute(&unicastSupported,
                        CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                        device);
   if (unicastSupported != 0) {
       // `device` supports unicast logical endpoints
   }

**多播逻辑端点支持**

查询设备是否支持多播逻辑端点：

.. code-block:: cpp

   int multicastSupported = 0;
   cuDeviceGetAttribute(&multicastSupported,
                        CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED,
                        device);
   if (multicastSupported != 0) {
       // `device` supports multicast logical endpoints
   }

**逻辑端点 IPC 句柄支持**

在导出或导入端点之前，查询设备为逻辑端点支持的 IPC 句柄类型。
该属性返回 ``CUlogicalEndpointIpcHandleType`` 值的位掩码。
以下代码检查是否支持用于逻辑端点的 fabric IPC 句柄类型：

.. code-block:: cpp

   int supportedHandleTypes = 0;
   cuDeviceGetAttribute(
       &supportedHandleTypes,
       CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES,
       device);
   if ((supportedHandleTypes &
        CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC) != 0) {
       // `device` supports fabric IPC handles for logical endpoints
   }

**计数操作支持**

计数操作是一种可选能力。
在端点上请求 ``CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS`` 之前，先查询设备是否支持计数操作：

.. code-block:: cpp

   int countedOpsSupported = 0;
   cuDeviceGetAttribute(&countedOpsSupported,
                        CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
                        device);
   if (countedOpsSupported != 0) {
       // `device` supports counted operations via logical endpoints
   }

**从所有者设备进行单播访问**

单播端点的所有者设备在传递给 ``cuLogicalEndpointCreate`` 的端点属性中指定。
``CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED`` 设备属性指示该设备是否可以访问其拥有的单播逻辑端点。
如果不支持这种访问，设备可以通过指针访问绑定的内存，而这目前为所有者设备提供最佳性能。
除所有者以外的设备可以通过与该端点相关联的逻辑端点 id 来访问单播端点。

.. code-block:: cpp

   int ownerAccessSupported = 0;
   cuDeviceGetAttribute(&ownerAccessSupported,
                        CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED,
                        device);
   if (ownerAccessSupported != 0) {
       // `device` supports unicast logical endpoint access on the owner device
   }


.. _api-overview:
.. _cft-api-overview:

4.18.3. API 概览
----------------

逻辑端点 API 是底层 :ref:`CUDA Driver API <driver-api>` 的一部分，使用逻辑端点遵循一个明确定义的生命周期。
下面的步骤概述了工作流程，后续各节将详细介绍每个步骤。
它们分为以下几个阶段：

- **设置阶段（步骤 1-11）：** 校验参与设备是同一 clique 的成员，保留一个 id ，描述并创建端点，与对等端共享，确认其就绪，然后在数据移动开始之前分配并绑定其后备内存。
- **使用阶段（步骤 12）：** 从设备针对端点发起 fabric 操作。
- **清理阶段（步骤 13）：** 在所有使用完成之后，解绑并销毁端点，释放其 id ，并释放后备内存。

1. **校验 clique 成员资格。** 使用 ``cuDeviceGetCliqueCount`` 和 ``cuDeviceGetCliqueInfo`` 校验参与设备属于同一 clique ，
   如 :ref:`第 4.18.4 节 <cft-clique-verification>` 所述。
2. **保留 id。** 使用 ``cuLogicalEndpointIdReserve`` 保留一个逻辑端点 id 范围。
   保留以进程为单位。
   id 的生命周期以及重用和释放的规则在 :ref:`第 4.18.6 节 <cft-logical-endpoint-ids>` 中描述。
3. **描述端点。** 填写 ``CUlogicalEndpointProp`` ，包括端点类型（单播或多播）、单播的所有者设备或多播的组大小、大小、
   请求的 IPC 句柄类型（ ``ipcHandleTypes`` ）以及任何标志——例如，
   ``CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS`` 用于请求计数操作支持（参见 :ref:`计数操作 <cft-counted-operation>` ）。
4. **检查对齐和大小限制。** 针对拟定的属性调用 ``cuLogicalEndpointGetLimits`` ，以获得所需的 ``bindAlignment`` 和 ``maxSize`` 。
   ``maxSize`` 是该配置允许的最大端点大小。
   端点大小和所有绑定都必须满足这些限制。
5. **创建端点。** 调用 ``cuLogicalEndpointCreate`` ，将保留的 id 之一与由上面填写的 ``CUlogicalEndpointProp`` 所描述的新建端点相关联。
6. **共享端点。** 使用 ``cuLogicalEndpointExport`` 将端点导出为 fabric IPC 句柄，
   通过任意 IPC 机制将该句柄传输给需要它的对等进程，并让每个对等进程使用 ``cuLogicalEndpointImport`` 导入它。
   导出和导入端点并不要求已有内存绑定到该端点。
7. **添加设备（仅多播）。** 对于多播端点，每个参与设备都必须使用 ``cuLogicalEndpointAddDevice`` 被添加到组中。
   每个进程在本地持有端点之后才添加自己的设备，因此对于多播，端点在设备被添加之前就已共享。
8. **确认就绪。** ``cuLogicalEndpointCreate`` 和 ``cuLogicalEndpointImport`` 都是非阻塞的。
   使用 ``cuLogicalEndpointQuery`` 在使用之前确认就绪。
   对于本地创建的端点，在绑定其内存之前进行确认；对于导入的端点，在 kernel 针对其发起操作之前进行确认。
9. **分配后备内存。** 使用虚拟内存管理 API （ ``cuMemCreate`` ）或 ``cudaMallocAsync`` 分配后备内存。
   如果端点将与另一个进程共享，绑定的内存必须可外部共享。
   当通过导入的多播端点绑定内存时，同样适用该要求。
   在创建分配时请求 fabric 句柄类型（ ``CU_MEM_HANDLE_TYPE_FABRIC`` ）以将分配配置为可外部共享，并使分配大小同时满足 ``bindAlignment`` 和分配粒度。
   此步骤与端点创建相互独立，但两者都必须在绑定之前完成。
10. **绑定内存。** 使用 ``cuLogicalEndpointBindMem`` （按分配句柄）或 ``cuLogicalEndpointBindAddr`` （按已映射地址）将物理内存与端点相关联。
    内存不能绑定到导入的单播端点，但可以绑定到导入的多播端点。
    fabric 操作仅对已绑定的范围有效。
11. **绑定后同步。** 在任何进程发起 fabric 操作之前，参与进程必须在绑定之后进行同步，如 :ref:`第 4.18.8 节 <cft-binding-memory>` 所述。
12. **使用端点。** 启动针对对等端点 id 发起 fabric 操作的 kernel 。
13. **清理。** 解绑内存，销毁端点，释放 id 范围，并释放后备内存。


.. _verify-clique-membership:
.. _cft-clique-verification:

4.18.4. 校验 clique 成员资格
----------------------------

在发起任何 CFT 操作之前，我们需要确保所有参与设备都是支持所需 CFT 操作集合的某个 clique 的成员，并且对于特定的 clique 类型，它们都是同一 clique 的成员。
否则，这些设备无法通过网络以该特定类型的操作相互到达。

首先，我们查询此设备是多少个 clique 的成员：

.. code-block:: cpp

     size_t cliqueCount = 0; // not modified if there are errors
     cuDeviceGetCliqueCount(&cliqueCount, cuDevice);

然后，我们查询此设备所属所有 clique 的 clique 信息，并找出我们将要使用的 clique 的 clique id 。
在此示例中，我们将使用类型为 ``CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT`` 的 clique 。

.. code-block:: cpp

     std::vector<CUcliqueInfo> cliqueInfos(cliqueCount);
     cuDeviceGetCliqueInfo(cliqueInfos.data(), &cliqueCount, cuDevice);
     unsigned int localCliqueId = 0;
     bool found = false;
     for (size_t idx = 0; idx < cliqueCount; ++idx) {
       if (cliqueInfos[idx].type == CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT) {
         localCliqueId = cliqueInfos[idx].id;
         found = true;
         break;
       }
     }

校验所有参与设备对该 clique 类型具有相同的 clique id 。

.. code-block:: cpp

     std::vector<unsigned int> allCliqueIds(numRanks);
     MPI_Allgather(&localCliqueId, sizeof(unsigned int), MPI_BYTE,
                             allCliqueIds.data(), sizeof(unsigned int), MPI_BYTE, MPI_COMM_WORLD);
     for (int i = 0; i < numRanks; i++) {
       if (allCliqueIds[i] != localCliqueId) {
         fprintf(stderr, "Rank %d: Clique id %u does not match expected %u\n", myRank, allCliqueIds[i], localCliqueId);
         return EXIT_FAILURE;
       }
     }


.. _creating-an-endpoint:

4.18.5. 创建 Endpoint
---------------------

创建和导入端点的第一步是确定进程需要多少个逻辑端点 id 。
然后使用 ``cuLogicalEndpointIdReserve`` 保留一个 id 范围，该函数返回此范围的基准 id 。
选择的数量要足以覆盖进程将创建或导入的每一个端点。
与该范围相关联的端点可以具有不同的属性。
例如，应用程序可以为每个 GPU 保留一个 id ，再为一个多播端点保留一个 id 。
该示例为每个 rank 保留一个 id ：

.. code-block:: cpp

     CUlogicalEndpointId leId = 0;
     cuLogicalEndpointIdReserve(&leId, (uint32_t)numRanks);

保留范围之后，在创建或导入端点时将其中的 id 相关联。
对本地创建的端点使用 ``cuLogicalEndpointCreate`` ，当远程端点的句柄可用时使用 ``cuLogicalEndpointImport`` 。
对于进程创建的每个端点，在 ``CUlogicalEndpointProp`` 中描述其属性。
单播端点指定将持有其绑定内存的唯一所有者设备。
设置端点类型、请求的 IPC 句柄类型（ ``ipcHandleTypes`` ）以及任何标志，仅在需要时才选择启用计数操作：

.. code-block:: cpp

     CUlogicalEndpointProp endpointProp {};
     endpointProp.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
     endpointProp.unicast = {.device = cuDevice};
     endpointProp.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
     endpointProp.flags = useCounted ? CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS : CU_LOGICAL_ENDPOINT_FLAG_NONE;

端点大小则在其对齐和最大大小限制已知之后单独设置。
有关如何查询这些限制以及如何选择大小，参见 :ref:`第 4.18.6.1 节 <cft-limits-alignment>` 。

通过调用 ``cuLogicalEndpointCreate`` ，将一个保留的 id 与所描述的属性相关联。
创建本地端点并导入对等端点的进程通常为每个参与者保留一个 id ，从而以基准 id 加上其 rank 来寻址某个对等端。
每个 rank 在 ``leId + myRank`` 处创建自己的端点：

.. code-block:: cpp

     cuLogicalEndpointCreate(leId + myRank, &endpointProp);

``cuLogicalEndpointCreate`` 是非阻塞的，因此端点在它返回的那一刻还不可用。
在向其绑定内存之前，使用 ``cuLogicalEndpointQuery`` 确认它已完全构造好，如 :ref:`第 4.18.7.1 节 <cft-confirming-readiness>` 所述。

**多播端点。** 多播端点遵循相同的步骤，区别在于其描述方式，以及需要添加一组设备。
它以 ``type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST`` 描述，并将 ``multicast.numDevices`` 设置为多播组中的设备数量。
一旦创建并共享，每个参与设备都必须使用 ``cuLogicalEndpointAddDevice`` 加入该组。
进程只有在本地持有端点之后才添加自己的设备——创建者在 ``cuLogicalEndpointCreate`` 之后，每个对等端在 ``cuLogicalEndpointImport`` 之后。
在多播组完整之前，端点尚不可用。
成员资格在端点的整个生命周期内是永久的。


.. _logical-endpoint-ids:
.. _cft-logical-endpoint-ids:

4.18.6. 逻辑端点 id
-------------------

逻辑端点 id 是一个进程本地的名字，其生命周期与它所标识的端点的生命周期相互独立。
端点 id 总是处于以下状态之一：

1. **未保留（Not reserved）。** 该 id 不归此进程所有，也不能与端点相关联。 ``cuLogicalEndpointIdReserve`` 会保留它。
2. **已保留（Reserved）。** 该 id 归此进程所有，但不命名任何东西。
   使用一个已保留的 id 调用 ``cuLogicalEndpointCreate`` 或 ``cuLogicalEndpointImport`` 会将其与一个端点相关联，并使其进入已关联状态。
   ``cuLogicalEndpointIdRelease`` 将其退回未保留状态。
3. **已关联（Associated）。** 该 id 命名一个端点，使用该 id 的 fabric 操作以该端点为目标。 ``cuLogicalEndpointDestroy`` 移除这种关联，并将该 id 退回已保留状态。

``cuLogicalEndpointIdReserve`` 保留一个包含 ``count`` 个端点 id 的范围，并返回其基准 id ``baseLeId`` ，因此所保留的范围是 ``[baseLeId, baseLeId + count)`` 。
基准 id 是一个输出：调用者并不选择自己收到哪些 id ，并且如果没有 ``count`` 个 id 的连续范围可用，保留就会失败，例如因为进程中的另一个库持有保留。

一个端点可以同时与多个 id 相关联。
例如，它可以通过 ``cuLogicalEndpointCreate`` 与所有者的 id 相关联，并通过 ``cuLogicalEndpointImport`` 与额外的 id 相关联。
这些 id 是同一个端点的 *别名（alias）* ，针对其中任何一个 id 发起的操作都会到达同一个目的地。

``cuLogicalEndpointDestroy`` 移除其中一个 id 的关联。
之后，该 id 可以由后续的创建或导入操作与一个不同的端点相关联，而无需再次保留。
销毁还会解绑本进程通过该 id 绑定的所有内存。
只有当端点的最后一个别名被销毁时，端点本身才会被释放。

``cuLogicalEndpointIdRelease(baseLeId, count)`` 释放范围 ``[baseLeId, baseLeId + count)`` 内最多 ``count`` 个 id 。
该范围内的每个 id 都必须已被保留，并且与该范围内某个 id 相关联的每个端点都必须已被销毁。

由于保留是以进程为单位的，同一个端点不必在每个导入进程中以相同的 id 导入。
一个方便的约定是：每个进程保留一个大小相同的范围，每个参与者一个 id ，并在本地以 ``baseLeId + peer`` 来寻址特定对等端的端点。
每个进程有自己的已保留 ``baseLeId`` ，它们不必在进程之间保持一致。
如果同一个对等端点在不同进程中以不同的 id 导入， CUDA 不会检测或报告这种不一致。
维护正确的本地对等端到端点 id 的映射是应用程序的责任。


.. _limits-and-alignment:
.. _cft-limits-alignment:

4.18.6.1. 限制与对齐
^^^^^^^^^^^^^^^^^^^^

端点的对齐和最大大小要求取决于其属性，因此必须针对正在创建的特定配置进行查询，而不能凭空假设。 ``cuLogicalEndpointGetLimits`` 针对给定的 ``CUlogicalEndpointProp`` 返回这两个值：

.. code-block:: cpp

   cuuint64_t bindAlignment = 0;
   cuuint64_t maxSize = 0;
   cuLogicalEndpointGetLimits(&bindAlignment, &maxSize, &prop);

端点大小和每一个绑定偏移都必须是所返回的 ``bindAlignment`` 的倍数。 ``maxSize`` 是逻辑端点的最大大小。
如果 ``maxSize`` 小于 ``CUlogicalEndpointProp::size`` ，应用程序必须将请求调整为这个较小的值。
为了暴露整个请求的范围，应用程序可以将其划分到多个端点上，每个端点都不大于 ``maxSize`` 。

用单个分配覆盖整个端点是最简单的安排，但这并不是要求。
端点的偏移空间同样可以由多个绑定来支撑，这些绑定映射各不相同、可能并不连续的物理分配，只要每一个已绑定的范围都是 ``bindAlignment`` 的倍数即可。


.. _sharing-an-endpoint:

4.18.7. 共享 Endpoint
---------------------

与对等端共享端点遵循导出—传输—导入的交换流程：所有者将端点导出为 fabric IPC 句柄，将该句柄传输给需要它的对等端，然后每个对等端导入该端点，
并将其与自己保留的、尚未与任何端点相关联的某个逻辑端点 id 相关联。
共享使用 fabric IPC 句柄，因此端点必须绑定到可外部共享的内存，并且环境必须满足 :ref:`第 4.18.1 节 <cft-prerequisites-scope>` 中描述的 fabric IPC 要求。

使用 ``cuLogicalEndpointExport`` 导出端点，请求与描述该端点时相同的 IPC 句柄类型（ ``ipcHandleTypes`` ）。
这将产生一个 ``CUlogicalEndpointFabricHandle`` ，即表示该逻辑端点的 fabric IPC 句柄。
使用应用程序自行选择的进程间通信机制将其传输给对等端，然后让每个对等端使用 ``cuLogicalEndpointImport`` 导入它。
导入所用的 id 在本地选择：导入者可以选择在哪个逻辑端点 id 处导入该端点。
这个 id 必须已被成功保留，并且不得与任何其他逻辑端点相关联，但它不必与所有者创建该端点时使用的 id 一致。
每个设备都可以在不同的 id 处导入同一个端点，甚至可以在多个不同的 id 处导入（即允许多次导入同一个端点）。

下面的单播点对点示例在一个环上共享端点。
每个 rank 导出自己的端点，并通过一次 ``MPI_Sendrecv`` ，在将句柄发送给接收邻居的同时收集发送邻居的句柄。
它将该句柄导入为 ``leId + sendRank`` ，之后就用这个 id 针对所导入端点的已绑定内存发起 fabric 操作：

.. code-block:: cpp

     // Export our own endpoint to a fabric IPC handle, hand it to the peers that
     // need it, and import each peer's handle to a locally reserved id.
     CUlogicalEndpointFabricHandle localHandle;
     CUlogicalEndpointFabricHandle sendRankHandle;
     cuLogicalEndpointExport(&localHandle, leId + myRank, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);
     MPI_Sendrecv(
         &localHandle, sizeof(CUlogicalEndpointFabricHandle), MPI_BYTE, recvRank, 0,
         &sendRankHandle, sizeof(CUlogicalEndpointFabricHandle), MPI_BYTE, sendRank, 0,
         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     cuLogicalEndpointImport(leId + sendRank, &sendRankHandle, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);


.. _confirming-readiness:
.. _cft-confirming-readiness:

4.18.7.1. 确认就绪
^^^^^^^^^^^^^^^^^^

``cuLogicalEndpointCreate`` 和 ``cuLogicalEndpointImport`` 在请求被接受后就立即返回。
在 ``cuLogicalEndpointQuery`` 报告就绪之前，将某个逻辑端点 id 用于要求端点已完全构造好的 API 属于未定义行为。
``cuLogicalEndpointQuery`` 本身是非阻塞的：如果所查询范围内 *任一* id 尚未就绪，它返回 0 ；一旦给定范围内 *所有* id 都已就绪，它返回非零值，因此它通常在轮询循环中调用。

在查询就绪状态之前，先保留逻辑端点 id ，并通过 ``cuLogicalEndpointCreate`` 或 ``cuLogicalEndpointImport`` 将其与一个端点相关联。
多播端点在所有设备都通过 ``cuLogicalEndpointAddDevice`` 被添加进来之前不会就绪，因此多进程程序需要先导出（ ``cuLogicalEndpointExport`` ）并共享它们，然后再阻塞等待其就绪。

在对逻辑端点执行以下任何操作之前，调用 ``cuLogicalEndpointQuery`` 并等待它报告该逻辑端点已完全构造好：

- 使用 ``cuLogicalEndpointBindAddr`` 或 ``cuLogicalEndpointBindMem`` 绑定内存。
- 使用 ``cuLogicalEndpointUnbind`` 解绑内存。
- 发起任何 ``cuda::ptx::fabric_try_*`` 操作。

以下 API 不要求 ``cuLogicalEndpointQuery`` 已报告就绪：

- ``cuLogicalEndpointExport``
- ``cuLogicalEndpointDestroy``
- ``cuLogicalEndpointIdRelease``

这些 API 不要求就绪，但它们的其他生命周期前提条件仍然适用。


.. _binding-memory:
.. _cft-binding-memory:

4.18.8. 绑定内存
----------------

端点不拥有任何内存。
绑定将端点偏移空间中的一个范围与一个物理分配相关联，并且 fabric 操作仅对已绑定的范围有效。
绑定是按设备建立的，因此单播绑定适用于所有者设备，而多播绑定适用于多播团队中的单个设备。

内存可以通过两种方式之一进行绑定，具体取决于应用程序如何持有该分配：

- ``cuLogicalEndpointBindMem`` 按分配句柄绑定，使用由 ``cuMemCreate`` 返回的 ``CUmemGenericAllocationHandle`` 。
- ``cuLogicalEndpointBindAddr`` 按已映射的虚拟地址绑定。
  例如由 ``cudaMallocAsync`` 返回的指针，以及用 ``cuMemMap`` 映射的范围中的地址。
  当前支持的分配类型请参见 CUDA Memory Management API 。

.. note::

   并非每个 CUDA 可访问的分配都能绑定到逻辑端点。
   有关绑定 API 当前支持的分配类型，请参见 CUDA Driver API 文档。

两者都接受端点 id 、绑定所适用的设备、端点偏移空间内的一个偏移、要绑定的分配以及绑定的大小。
``cuLogicalEndpointBindMem`` 还额外接受一个进入该分配的 ``memOffset`` ，允许只绑定该句柄的一个子范围。
端点偏移、大小以及（对于 ``cuLogicalEndpointBindMem`` ） ``memOffset`` 都必须是 ``bindAlignment`` 的倍数，
并且所绑定的范围必须位于端点的大小之内（参见 :ref:`第 4.18.6.1 节 <cft-limits-alignment>` ）。

.. code-block:: cpp

     // Bind the whole backing allocation at endpoint offset 0, by allocation handle.
     cuLogicalEndpointBindMem(leId + myRank, cuDevice, 0, exportHandle, 0, exportSize, 0);
     // No rank may issue an operation until every destination has been bound.
     MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

在建立初始绑定之后，参与进程必须在任何进程发起 fabric 操作之前进行同步。
对于单播端点，每个导入者都必须在所有者绑定其后备内存之后与导出进程同步。
对于多播端点，所有参与进程都必须等到每个参与者都已绑定其后备内存。
应用程序可以使用任何能够建立这种先后顺序的进程间同步机制。

由于对等端导入的是端点而不是所有者的分配，所有者可以替换端点背后的内存，而无需任何对等端重新导入。
使用 ``cuLogicalEndpointUnbind`` 解绑当前范围，并通过同一个端点 id 将一个新的分配绑定到端点偏移空间的同一范围。
所有者与其对等端必须跨这次重新绑定进行同步，以免任何对等端在端点处于未绑定状态时对其发起操作。
发起落在当前已绑定范围之外的操作属于未定义行为。


.. _fabric-operations:

4.18.9. Fabric 操作
-------------------

线程通过执行 *fabric 操作* 来访问端点资源，这些操作接受一个 CFT 句柄——由逻辑端点 id 和偏移组成的一对值。
fabric 操作是异步的，也就是说，程序必须显式等待其完成才能观察到它们的效果。
fabric 操作的名字带有 ``try_`` 前缀，表示它们可能失败。
也就是说，在等待完成之后，线程必须检查该操作的完成机制（参见 :ref:`第 4.18.10 节 <cft-completion-status>` ），以确定操作是否成功。
这一点将 fabric 操作与普通的指针操作区分开来：基于指针的访问所产生的失败，是通过销毁上下文甚至整个应用程序来向应用程序报告的。
想要处理这些错误的应用程序——例如从指针访问失败中恢复——只能在上下文或进程级别这样做。
fabric 操作则允许应用程序在程序线程级别处理错误。

设备代码通过以下 ``cuda::ptx`` 指令发起 fabric 操作：

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - 指令
     - 操作
   * - ``cuda::ptx::fabric_try_put``
     - 写入一个 CFT 句柄。
   * - ``cuda::ptx::fabric_try_get``
     - 从一个 CFT 句柄读取。
   * - ``cuda::ptx::fabric_try_red``
     - 向一个 CFT 句柄进行规约。
   * - ``cuda::ptx::fabric_try_pullred``
     - 从一个 CFT 句柄进行拉取规约。
   * - ``cuda::ptx::fabric_try_atom``
     - 对一个 CFT 句柄执行原子的读-改-写。

上述操作——除 ``try_atom`` 之外——以 16 字节为单位移动数据，因此传输的两端都必须 16 字节对齐：源指针和进入端点的目的偏移必须各自都是 16 字节的倍数。

该示例发起 fabric put 。 put 从一个共享内存缓冲区读取数据，并将其写入一个目标句柄。
多个线程将输入数据写入共享内存，然后使用 ``__syncthreads`` 与发起 put 的线程同步。
put 经由异步代理读取共享内存，因此发起线程调用 ``cuda::ptx::fence_proxy_async(cuda::ptx::space_shared)`` ，将通用代理的共享内存写入同步到异步代理。
put 接受目标句柄、源缓冲区基地址、字节数，以及将用于跟踪完成的 ``mbarrier`` ：

.. code-block:: cpp

       // Make the staged shared-memory source visible to the async proxy before the put reads it.
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put(cuda::ptx::space_shared, cuda::ptx::sem_relaxed,
                                 cuda::ptx::scope_sys, putLeId, offset, smemBuf,
                                 FABRIC_CHUNK_SIZE, &putBar);

该调用设置了若干指令限定符，其完整集合——每条指令所接受的访问大小、内存排序模式和作用域——由 PTX ISA 规定
（参见 `Fabric Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fabric-operations>`_ ）。
这里值得专门指出该操作的两个方面，因为它们将这个操作与程序的其余部分联系起来：它的完成如何被跟踪，以及内存排序如何指定。

**事务计数。** 使用 ``mbarrier`` 跟踪完成的 fabric 操作遵循与 :ref:`使用 TMA 传输数据 <async-copies-tma-one-dim>` 中描述的按指针寻址的 ``cuda::ptx::cp_async_bulk`` 拷贝相同的 ``mbarrier`` 事务核算模型。
随着 put 完成，它每交付 16 个字节就将其 ``mbarrier`` 的事务计数减 1 。
当所有预期的到达都已发生且事务计数达到零时，屏障 phase 完成。 put 本身并不设置预期事务计数。
在操作提交之后发起的一个单独的 ``cuda::ptx::mbarrier_arrive_expect_tx`` 会记录一次到达，并将事务计数增加字节数除以 16 ，因为这正是 put 将会把它减去的数量。
取数类操作则改用基于字节的事务核算。 ``cuda::ptx::fabric_try_get`` 和 ``cuda::ptx::fabric_try_pullred`` 每交付一个字节就将事务计数减 1 ，
因此 ``cuda::ptx::mbarrier_arrive_expect_tx`` 要加上完整的字节数。
:ref:`第 4.18.10 节 <cft-completion-status>` 介绍了 mbarrier 的布局要求以及 fabric 状态如何被读回。

**计数的（counted）** put 还会随着数据到达而递增目的地一侧的一个字节计数器，这样接收方就可以在访问本地数据之前对该计数器进行轮询，而不必交换单独的完成消息。
下面的代码用 ``cuda::ptx::fabric_try_put_counted`` ——普通 put 的 ``.counted::bytes`` 变体——做到这一点，它额外传入第二个目的偏移，即计数器的端点偏移：

.. code-block:: cpp

       // Make the staged shared-memory source visible to the async proxy before the put reads it.
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put_counted(cuda::ptx::space_shared, cuda::ptx::sem_relaxed,
                                         cuda::ptx::scope_sys, putLeId, offset, counterOffset,
                                         smemBuf, FABRIC_CHUNK_SIZE, &putBar);

计数操作要求在创建端点时启用计数操作支持（ ``CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS`` ，
参见 :ref:`第 4.18.2.2 节 <cft-query-for-support>` ），并且目标计数器必须在端点内存中按 256 字节边界对齐。
应用程序在布局端点的后备内存时即可满足这一点，将计数器保留在一个适当对齐的区域中。

尽管该示例只发起 put ， get 、规约和拉取规约操作使用 :ref:`第 4.18.10 节 <cft-completion-status>` 中描述的相同提交、状态报告和等待序列。
如上所述，它们的事务计数单位是随操作而异的。

**内存排序。** 该示例以 ``sys`` 作用域的 ``relaxed`` 排序发起其 fabric put 。
一次 fabric put 跨越多个代理：它通过 *fabric 代理* 访问远程数据（对于计数的 put ，还有计数器），
通过 *异步代理* 读取其 ``.shared::cta`` 源，并通过 *通用代理* 更新完成 ``mbarrier`` 。
由于 mbarrier 是通过通用代理更新的，只需在屏障处用块作用域操作等待即可观察到完成。
然而，与 ``cuda::ptx::cp_async_bulk`` 操作不同，观察到 fabric 操作完成并不会将 fabric 代理的访问排序到通用代理。
在通过指针经通用代理访问数据之前，必须使用 ``cuda::ptx::fence_proxy_generic_fabric_alias`` 将 fabric 代理与通用代理同步。

仅有端点 id 并不能提供对远程目的地的指针访问。
导入端点并不会将其绑定的资源映射到导入者的虚拟地址空间中。
因此，下面的示例使用一个由发起线程的设备拥有的单播端点。 ``leId`` 是端点 id ，并且绑定在 ``dataOffset`` 处的内存也被映射到了该设备的虚拟地址空间中。 ``dataPtr`` 指向同一个位置。

在发起 put 之后，线程等待其完成。
在通过 ``dataPtr`` 经通用代理读取目的地之前，线程必须先以 ``cuda::ptx::sem_release`` 、再以 ``cuda::ptx::sem_acquire`` 发起 ``cuda::ptx::fence_proxy_generic_fabric_alias`` 。
仅靠完成等待并不会把 fabric 代理的写入排序在通用代理的指针加载之前。
``waitForSuccessfulCompletion`` 代表 :ref:`第 4.18.10 节 <cft-completion-status>` 中描述的 ``mbarrier`` 等待和状态检查。
初始时目的地为 0 ，而 ``smemSrcData`` 是一个 16 字节的共享内存源，其第一个 ``uint32_t`` 元素存放 42 。

.. list-table::
   :widths: 100
   :header-rows: 1

   * - 由同一个线程发起并加载（线程 0 ）
   * - .. code-block:: cpp

        namespace ptx = cuda::ptx;

        ptx::fabric_try_put(
            ptx::space_shared,
            ptx::sem_relaxed,
            ptx::scope_sys,
            leId, dataOffset,
            smemSrcData, 16, &mBar);
        ptx::fabric_submit();
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_acquire,
            ptx::scope_cta,
            ptx::space_shared,
            &mBar, 1);
        waitForSuccessfulCompletion(&mBar);

        // Order the subsequent generic-proxy load after the fabric put.
        ptx::fence_proxy_generic_fabric_alias(ptx::sem_release);
        ptx::fence_proxy_generic_fabric_alias(ptx::sem_acquire);

        assert(*dataPtr == 42);

mbarrier 等待返回之后， put 已经完成，但 fabric 代理的写入相对于通用代理的加载仍然是无序的。 release 和 acquire 栅栏建立了这一顺序，因此指针加载观察到值 42 。

当一个线程用 fabric 操作发布数据、而端点所有者设备上的另一个线程通过指针消费绑定的目的地时，同样需要这个 release/acquire 代理栅栏。
在下面这个示意性的消息传递示例中，发送方通过 *fabric 代理* 用 fabric 操作写入数据和信号标志。
接收方使用 *通用代理* 通过指针轮询该标志并读取数据。 ``dataPtr`` 和 ``flagPtr`` 寻址绑定到接收方所拥有的端点的目的内存。
初始时，目的地数据和标志都为 0 。
``smemSrcData`` 和 ``smemSrcFlag`` 是由四个 ``uint32_t`` 元素组成的 16 字节共享内存源。
``smemSrcData`` 的第一个元素存放 42 ， ``smemSrcFlag`` 的第一个元素存放 1 。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - 发送方（线程 0 ）
     - 接收方（线程 1 ）
   * - .. code-block:: cpp

        namespace ptx = cuda::ptx;

        ptx::fabric_try_put(
            ptx::space_shared,
            ptx::sem_relaxed,
            ptx::scope_sys,
            leId, dataOffset,
            smemSrcData, 16, &mBar);
        ptx::fabric_submit();
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_acquire,
            ptx::scope_cta,
            ptx::space_shared,
            &mBar, 1);
        waitForSuccessfulCompletion(&mBar);

        ptx::fence_proxy_generic_fabric_alias(
            ptx::sem_release);

        ptx::fabric_try_put(
            ptx::space_shared,
            ptx::sem_relaxed,
            ptx::scope_sys,
            leId, flagOffset,
            smemSrcFlag, 16, &mBar);
        ptx::fabric_submit();
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_acquire,
            ptx::scope_cta,
            ptx::space_shared,
            &mBar, 1);
        waitForSuccessfulCompletion(&mBar);

     - .. code-block:: cpp

        namespace ptx = cuda::ptx;

        cuda::atomic_ref<uint32_t,
        cuda::thread_scope_system> flag(*flagPtr);

        while (flag.load(cuda::memory_order_relaxed) != 1);

        // Order the data load after the observed flag update.
        ptx::fence_proxy_generic_fabric_alias(
            ptx::sem_acquire);

        assert(*dataPtr == 42);

数据 put 完成之后，发送方在发起标志 put 之前先发起 ``cuda::ptx::fence_proxy_generic_fabric_alias(cuda::ptx::sem_release)`` 。
当接收方对标志的加载观察到由那次 put 写入的值 1 时，接收方就知道发送方已经发布了数据。
在消费数据之前，接收方必须发起 ``cuda::ptx::fence_proxy_generic_fabric_alias(cuda::ptx::sem_acquire)`` 。
这个 acquire 栅栏防止后续对数据的指针加载被排序到所观察到的标志更新之前。
发送方的 release 栅栏和接收方的 acquire 栅栏共同将发送方的数据 put 排序在接收方的数据加载之前。
在没有对数据的任何其他写入的情况下，该加载因此观察到值 42 。


.. _per-thread-and-warp-collective-operations:
.. _cft-collective-operations:

4.18.9.1. 每线程与 warp 集合操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每条 fabric 指令都定义了自己的执行粒度。
``fabric.try_get`` 、 ``fabric.try_get.tensor`` 、 ``fabric.try_put`` 、 ``fabric.try_put.tensor`` 和 ``fabric.try_red`` 是每线程的：每次执行都为发起线程启动一个独立的操作。
``fabric.try_pullred`` 是 warp 集合的：一个 warp 的全部 32 个 lane 必须一起执行该指令。
PTX ISA 规定了每条指令所要求的参与方式。

执行粒度只描述线程如何发起 fabric 操作。
完成跟踪与该粒度无关；参见 :ref:`第 4.18.10.1 节 <cft-configuring-barrier>` 。


.. _completion-status-and-error-reporting:
.. _cft-completion-status:

4.18.10. 完成状态与错误报告
---------------------------

fabric 操作通过其完成对象报告自己的状态。
这里描述的操作使用共享内存中的 ``mbarrier`` 作为完成对象，因此同一个 ``mbarrier`` 既发出完成信号又记录状态。
通过以 ``cuda::ptx::layout_v1`` 调用 ``cuda::ptx::mbarrier_init`` 来初始化这个 ``mbarrier`` 。
该布局为对象扩展了存放 fabric 操作状态的字段。
默认的 ``cuda::ptx::layout_v0`` 只跟踪到达计数和事务计数，无法携带该状态，因此不能与这种报告机制一起使用：

.. code-block:: cpp

       cuda::ptx::mbarrier_init(cuda::ptx::layout_v1, &putBar, 1);

``cuda::ptx::fabric_try_put`` 接受用于跟踪该操作的 mbarrier 。
随着操作的数据被交付， fabric 引擎在屏障上完成相应的事务，并在那里记录该操作的状态。
因此，传输已经落地的信号来自 put 本身，而不是来自发起线程。

put 会交付它的字节，但并不告诉屏障应该期待多少字节。
对于这里展示的每线程 put ，发起线程在一个单独的步骤中提供该计数。
它先用 ``fabric_submit`` 提交操作，使 fabric 引擎最终开始消费它，
然后调用 ``cuda::ptx::mbarrier_arrive_expect_tx`` 记录一次到达，
并加上预期事务计数——即该操作将交付的字节数，以 16 字节为单位计数。
每个 fabric 操作都必须在跟踪其完成的屏障 phase 推进之前被提交。
提交操作之后，发起线程要么执行该 phase 推进所需的屏障操作（例如 ``cuda::ptx::mbarrier_arrive_expect_tx`` 或 ``cuda::ptx::mbarrier_expect_tx`` ），
要么与另一个执行此类操作的线程同步。
发起线程不必自己等待屏障。
可以由另一个线程执行该等待。
一旦所有预期的到达都已发生且事务计数达到零，屏障 phase 就会推进，等待返回。
程序必须在 grid 退出之前等待每一个 fabric 操作完成。
在 grid 退出时留下未完成的操作属于未定义行为。

下面的序列提交该 put 并等待其完成：

.. code-block:: cpp

       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta,
                                            cuda::ptx::space_shared, &putBar,
                                            (FABRIC_CHUNK_SIZE / FABRIC_TX_GRANULARITY_BYTES));
       // Spin while the barrier's phase hasn't flipped yet. The report outputs are
       // only valid once the barrier completes, so inspect them after.
       bool isReportSeen = false;
       uint8_t reportValue = 0;
       bool complete = waitWithDebugTimeout(
           [&]() {
             return !cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::mbarrier_phase_primary, cuda::ptx::sem_relaxed,
                                                         cuda::ptx::scope_cta, isReportSeen, reportValue, &putBar,
                                                         /*phaseParity=*/0);
           },
           "cftTryPutKernel: mbarrier try_wait parity");
       if (!complete) { __trap(); }
       if (isReportSeen) {
         reportFabricError(&reportValue, offset);
       }

``cuda::ptx::mbarrier_try_wait_parity`` 的 ``phase_type::primary`` 形式除了 *完成谓词（completion predicate）* 之外，
还有 *报告谓词（report predicate）* 和 *报告值（report value）* 两个目的操作数。
如果 *完成谓词* 为 false ， *报告谓词* 和 *报告值* 包含未指定的值。
当 ``try_wait`` 观察到 phase 完成时——即 *完成谓词* 为 true 时—— *报告谓词* 和 *报告值* 一起包含关于该屏障 phase 所跟踪的操作的附加信息。
如果该屏障 phase 跟踪的所有操作都是 fabric 操作，那么 *报告谓词* 为 false 表示所有 fabric 操作都成功了。
否则，如果 *报告谓词* 为 true ， *报告值* 可能包含关于这些失败的更多信息，程序可以使用 ``cudaFabricOpErrorStatusCount`` 和 ``cudaFabricOpErrorStatusGet`` 来检查这些信息。

.. code-block:: cpp

   inline __device__ void reportFabricError(uint8_t *reportValue, uint64_t offset) {
     const unsigned long long off = offset;
     unsigned int errCount = 0;
     if (cudaFabricOpErrorStatusCount(reportValue, cudaFabricOpStatusSourceMbarrierV1, &errCount) != cudaSuccess) {
       printf("cudaFabricOpErrorStatusCount failed at offset %llu\n", off);
       return;
     }
     for (unsigned int i = 0; i < errCount; ++i) {
       cudaFabricOpStatusInfo info = cudaFabricOpStatusInfoSuccess;
       if (cudaFabricOpErrorStatusGet(reportValue, cudaFabricOpStatusSourceMbarrierV1, i, &info) == cudaSuccess) {
         printf("fabric put error[%u/%u] at offset %llu: status=%d\n", i, errCount, off, (int)info);
       } else {
         printf("cudaFabricOpErrorStatusGet failed for error[%u/%u] at offset %llu\n", i, errCount, off);
       }
     }
   }

解码后的状态指出了操作失败的原因，这使应用程序能够恢复，而不是让进程出错。
诸如丢包或链路重置之类的瞬态 fabric 错误可以被重试、重新路由，或由应用程序特定的回退方案来处理。
该示例打印解码后的错误，而通信库通常会根据它们采取行动以保持前进进度。


.. _configuring-barrier-arrivals-and-transaction-counts:
.. _cft-configuring-barrier:

4.18.10.1. 配置屏障到达与事务计数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上面的完成序列记录了由发起 put 的线程贡献的到达和预期事务。
哪个线程可以记录它们，取决于操作的执行粒度。
对于每线程操作，发起线程必须为自己的操作记录它们。
对于 warp 集合的 ``fabric.try_pullred`` ，要么由参与 warp 的单个 lane 代表整个 warp 记录它们，要么由每个 lane 记录自己的到达。
``cuda::ptx::mbarrier_arrive_expect_tx`` 是一个融合的单一操作：它在同一步骤中计入一次到达并增加预期事务计数，因此这两者必须由同一个线程完成。
它必须在操作通过 ``fabric.submit`` 提交之后、且等待观察到完成之前运行。
记录到达和预期事务计数的线程不必是等待屏障的那个线程。

多个独立的操作可以共享一个屏障。
当 warp 集合的 ``fabric.try_pullred`` 使用单次到达时，可以由任意一个 lane 加上等于该 warp 交付字节数的预期事务计数。
而当改为由每个 lane 记录一次到达时，其余 31 个 lane 贡献零个事务，因此屏障初始化的到达计数必须是完整的 warp 宽度。

当许多操作共享一个屏障时，每线程记录一次到达会使屏障的到达计数膨胀。 ``layout::v1`` 的 mbarrier 将其到达计数存放在一个 9 位字段中，因此计数上限为 511 ；超过它就会使屏障溢出。
一个由 512 个线程组成的块，每个线程都在同一个共享屏障上到达，就已经越过了这个上限。
在粒度允许的情况下，优先每组只记录一次到达：同步 warp 并选出单个 lane ，或者同步整个块并在块范围的共享屏障上记录一次到达。


.. _cleaning-up:

4.18.11. 清理
-------------

清理要求在任何可能对某个端点发起操作的进程与任何已向该端点绑定资源的进程之间进行同步。
对于单播端点，这意味着当某个导入者仍可能对端点发起操作时，所有者不得解绑或销毁该端点，也不得释放其后备资源。
对于多播端点，每个参与进程拥有自己绑定的副本，因此当另一个进程仍可能对端点发起操作时，任何参与者都不得解绑或释放自己的副本。
重要的是先后顺序，而不是强制实现它的机制：任何能够在绑定被移除之前建立这种顺序的进程间同步都是足够的。

.. code-block:: cpp

     // ---- Clean up endpoints and free the backing allocation. ----
     // Once every rank's stream has synchronized, all fabric puts
     // have completed and no rank will issue another operation. Making that global
     // with a barrier ensures no peer still targets an endpoint we clean up below.
     MPI_Barrier(MPI_COMM_WORLD);

     // Destroy the imported endpoint (the peer target our puts addressed). An import
     // is only a local reference to the peer's endpoint id and maps no peer memory,
     // so once our puts have drained this just drops that reference.
     cuLogicalEndpointDestroy(leId + sendRank);

     // Unbind and destroy our owned endpoint, then release the reserved id range.
     cuLogicalEndpointUnbind(leId + myRank, cuDevice, 0, exportSize);
     cuLogicalEndpointDestroy(leId + myRank);
     cuLogicalEndpointIdRelease(leId, numRanks);

     // Free the backing allocation in VMM cleanup order: unmap the range, release
     // the physical handle, then free the reserved virtual address range.
     cuMemUnmap(exportPtr, exportSize);
     cuMemRelease(exportHandle);
     cuMemAddressFree(exportPtr, exportSize);

**等待 fabric 操作以及所有发起方停止。** 在释放跟踪 fabric 操作的完成对象之前，必须先等待该对象。
对于由某个 grid 发起的 fabric 操作，必须由该 grid 中的一个线程在 grid 退出之前等待完成。
不执行该等待就退出属于未定义行为。
在该示例中，每个 kernel 在返回之前都等待完成，而流同步则确认这些 kernel 及其等待已经完成。

由另一个进程发起、以绑定到同一个逻辑端点的资源为目标的 fabric 操作可能仍在进行中。
因此，在移除单播绑定之前，所有者必须等待，直到每个导入者都已停止发起操作。
在移除任何多播绑定之前，每个拥有已绑定副本的参与者都必须确知所有进程已停止发起操作。
该示例在所有 rank 同步其 CUDA 流之后，用一个屏障来做到这一点。
当应用程序能够保持所要求的先后顺序时，可以使用更窄的同步方案。

**销毁导入的端点 id 。** 每个导入进程通过调用 ``cuLogicalEndpointDestroy`` 来销毁与其导入的端点相关联的逻辑端点 id 。
这只丢弃那个本地别名，不会在所有者一侧解绑任何东西，因此销毁本身不需要跨进程协调。

**销毁所有者的端点 id 。** 销毁所有者的逻辑端点 id 也会解绑本进程通过该 id 绑定的所有内存，因此先调用 ``cuLogicalEndpointUnbind`` 是可选的，但推荐这样做。
一旦每个别名——所有者的 id 以及任何导入——都已被销毁，端点的资源就会被释放。

同样的路径适用于单播和多播端点。
由于多播成员资格是永久的，更改多播组需要销毁端点并创建一个新端点。
在端点存在期间，设备无法离开该组。

**重用或释放端点 id 。** 在用 ``cuLogicalEndpointDestroy`` 销毁某个端点关联之后，其 id 仍保持已保留但不再关联的状态，可以立即重用。
要将 id 归还给系统，对先前用 ``cuLogicalEndpointIdReserve`` 保留的范围调用 ``cuLogicalEndpointIdRelease`` 。

**释放后备分配。** 端点不拥有其后备内存，因此该分配要单独释放。
使用与内存分配方式相对应的释放机制。
对于 VMM 分配，通常的顺序是：用 ``cuMemUnmap`` 移除映射，用 ``cuMemRelease`` 释放物理句柄，再用 ``cuMemAddressFree`` 归还保留的虚拟地址范围。
程序不得在某个分配仍绑定到端点时释放它。
