.. _driver-entry-point-access-details:

4.20. Driver Entry Point Access
================================

.. _introduction-driver-entry-point-access:

4.20.1. 简介
------------

``Driver Entry Point Access APIs`` 提供了一种获取 CUDA driver 函数地址的方法。从 CUDA 11.3 开始，用户可以使用通过这些 API 获取的函数指针调用可用的 CUDA driver API。

这些 API 提供了类似于 POSIX 平台上 dlsym 和 Windows 上 GetProcAddress 的功能。提供的 API 允许用户：

- 使用 ``CUDA Driver API.`` 获取 driver 函数的地址。
- 使用 ``CUDA Runtime API.`` 获取 driver 函数的地址。
- 请求 CUDA driver 函数的 *per-thread default stream* 版本。更多详情请参阅 :ref:`retrieve-per-thread-default-stream-versions`。
- 在旧版 toolkit 但新版 driver 的情况下访问新的 CUDA 功能。

.. _driver-function-typedefs:

4.20.2. Driver Function Typedefs
---------------------------------

为了帮助获取 CUDA Driver API 入口点，CUDA Toolkit 提供了包含所有 CUDA driver API 函数指针定义的头文件访问。这些头文件随 CUDA Toolkit 一起安装，位于 toolkit 的 ``include/`` 目录中。下表总结了每个 CUDA API 头文件对应的包含 ``typedefs`` 的头文件。

.. _tbl:typedefs-header-files:

.. table:: CUDA driver API 的 Typedefs 头文件

   =====================  ==============================
   API header file        API Typedef header file
   =====================  ==============================
   ``cuda.h``             ``cudaTypedefs.h``
   ``cudaGL.h``           ``cudaGLTypedefs.h``
   ``cudaProfiler.h``     ``cudaProfilerTypedefs.h``
   ``cudaVDPAU.h``        ``cudaVDPAUTypedefs.h``
   ``cudaEGL.h``          ``cudaEGLTypedefs.h``
   ``cudaD3D9.h``         ``cudaD3D9Typedefs.h``
   ``cudaD3D10.h``        ``cudaD3D10Typedefs.h``
   ``cudaD3D11.h``        ``cudaD3D11Typedefs.h``
   =====================  ==============================

上述头文件本身并不定义实际的函数指针；它们定义的是函数指针的 typedef。例如， ``cudaTypedefs.h`` 中有 driver API ``cuMemAlloc`` 的以下 typedef：

.. code-block:: c++

   typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
   typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);

CUDA driver 符号采用基于版本的命名方案，在名称中带有 ``_v*`` 扩展（第一个版本除外）。当特定 CUDA driver API 的签名或语义发生变化时，我们会增加相应 driver 符号的版本号。以 ``cuMemAlloc`` driver API 为例，第一个 driver 符号名称是 ``cuMemAlloc`` ，下一个符号名称是 ``cuMemAlloc_v2`` 。在 CUDA 2.0 (2000) 中引入的第一个版本的 typedef 是 ``PFN_cuMemAlloc_v2000`` 。在 CUDA 3.2 (3020) 中引入的下一个版本的 typedef 是 ``PFN_cuMemAlloc_v3020`` 。

``typedefs`` 可用于更轻松地在代码中定义适当类型的函数指针：

.. code-block:: c++

   PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
   PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;

.. _driver-function-retrieval:

4.20.3. Driver Function Retrieval
----------------------------------

使用 Driver Entry Point Access API 和适当的 typedef，我们可以获取任何 CUDA driver API 的函数指针。

.. _using-the-driver-api:

4.20.3.1. 使用 Driver API
^^^^^^^^^^^^^^^^^^^^^^^^^

driver API 需要 CUDA 版本作为参数，以获取请求的 driver 符号的 ABI 兼容版本。CUDA Driver API 具有按函数划分的 ABI，用 ``_v*`` 扩展表示。例如，考虑 ``cuStreamBeginCapture`` 的版本及其在 ``cudaTypedefs.h`` 中对应的 ``typedefs`` ：

.. code-block:: c++

   // cuda.h
   CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
   CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);

   // cudaTypedefs.h
   typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
   typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);

从上面代码片段中的 ``typedefs`` 来看，版本后缀 ``_v10000`` 和 ``_v10010`` 表示上述 API 分别在 CUDA 10.0 和 CUDA 10.1 中引入。

.. code-block:: c++

   #include <cudaTypedefs.h>

   // Declare the entry points for cuStreamBeginCapture
   PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
   PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;

   // Get the function pointer to the cuStreamBeginCapture driver symbol
   cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   // Get the function pointer to the cuStreamBeginCapture_v2 driver symbol
   cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

参考上面的代码片段，要获取 driver API ``cuStreamBeginCapture`` 的 ``_v1`` 版本的地址，CUDA 版本参数应该正好是 10.0 (10000)。同样，获取 API 的 ``_v2`` 版本地址的 CUDA 版本应该是 10.1 (10010)。指定更高的 CUDA 版本来获取特定版本的 driver API 可能并不总是可移植的。例如，在这里使用 11030 仍然会返回 ``_v2`` 符号，但如果在 CUDA 11.3 中发布了假设的 ``_v3`` 版本，当与 CUDA 11.3 driver 配对时， ``cuGetProcAddress`` API 将开始返回较新的 ``_v3`` 符号。由于 ``_v2`` 和 ``_v3`` 符号的 ABI 和函数签名可能不同，使用为 ``_v2`` 符号设计的 ``_v10010`` typedef 调用 ``_v3`` 函数将表现出未定义行为。

注意，使用无效的 CUDA 版本请求 driver API 将返回错误 ``CUDA_ERROR_NOT_FOUND`` 。在上面的代码示例中，传入小于 10000 (CUDA 10.0) 的版本将是无效的。

.. _using-the-runtime-api:

4.20.3.2. 使用 Runtime API
^^^^^^^^^^^^^^^^^^^^^^^^^^

runtime API ``cudaGetDriverEntryPointByVersion`` 使用用户提供的 CUDA 版本，以与 ``cuGetProcAddress`` 相同的方式获取请求的 driver 符号的 ABI 兼容版本。在下面的代码片段中，所需的最低 CUDA 版本是 CUDA 11.2，因为 ``cuMemAllocAsync`` 是在那时引入的。

.. code-block:: c++

   #include <cudaTypedefs.h>

   int cudaVersion;
   // Ensure a CUDA driver >= 11.2 is installed or we will get an error from cuGetProcAddress
   status = cuDriverGetVersion(&cudaVersion);
   if (cudaVersion >= 11020) {

      // Declare the entry point
      PFN_cuMemAllocAsync_v11020 pfn_cuMemAllocAsync;

      // Initialize the entry point
      cudaGetDriverEntryPointByVersion("cuMemAllocAsync", &pfn_cuMemAllocAsync, 11020, cudaEnableDefault, &driverStatus);

      // Call the entry point
      if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
          pfn_cuMemAllocAsync(...);
      }
   }

.. _retrieve-per-thread-default-stream-versions:

4.20.3.3. 获取 Per-thread Default Stream 版本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些 CUDA driver API 可以配置为具有 *default stream* 或 *per-thread default stream* 语义。具有 *per-thread default stream* 语义的 driver API 在其名称中带有 *_ptsz* 或 *_ptds* 后缀。例如， ``cuLaunchKernel`` 有一个名为 ``cuLaunchKernel_ptsz`` 的 *per-thread default stream* 变体。使用 Driver Entry Point Access API，用户可以请求 driver API ``cuLaunchKernel`` 的 *per-thread default stream* 版本，而不是 *default stream* 版本。配置 CUDA driver API 的 *default stream* 或 *per-thread default stream* 语义会影响同步行为。更多详情可以在 `这里 <https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream>`_ 找到。

driver API 的 *default stream* 或 *per-thread default stream* 版本可以通过以下方式之一获取：

- 使用编译标志 ``--default-stream per-thread`` 或定义宏 ``CUDA_API_PER_THREAD_DEFAULT_STREAM`` 来获取 *per-thread default stream* 行为。
- 使用标志 ``CU_GET_PROC_ADDRESS_LEGACY_STREAM/cudaEnableLegacyStream`` 或 ``CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM/cudaEnablePerThreadDefaultStream`` 分别强制 *default stream* 或 *per-thread default stream* 行为。

.. _access-new-cuda-features:

4.20.3.4. 访问新的 CUDA 功能
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

始终建议安装最新的 CUDA toolkit 以访问新的 CUDA driver 功能，但如果由于某种原因用户不想更新或无法访问最新的 toolkit，该 API 可以仅通过更新的 CUDA driver 来访问新的 CUDA 功能。为了讨论，假设用户使用的是 CUDA 12.3，并且想要使用 CUDA 12.5 driver 中可用的新 driver API ``cuFoo`` 。下面的代码片段说明了这个用例：

.. code-block:: c++

   int main()
   {
       // Manually define the prototype as cudaTypedefs.h in CUDA 12.3 does not have the cuFoo typedef
       typedef CUresult (CUDAAPI *PFN_cuFoo_v12050)(...);
       PFN_cuFoo_v12050 pfn_cuFoo = NULL;
       CUdriverProcAddressQueryResult driverStatus;
       int cudaVersion;

       // Ensure a CUDA driver >= 12.5 is installed or we will get an error from cuGetProcAddress
       CUresult status = cuDriverGetVersion(&cudaVersion);
       if (cudaVersion >= 12050) {
           // Get the address for cuFoo API using cuGetProcAddress. Specify CUDA version as
           // 12050 since cuFoo was introduced then
           CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, 12050, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

           if (status == CUDA_SUCCESS && pfn_cuFoo) {
               pfn_cuFoo(...);
           }
           else {
               printf("Cannot retrieve the address to cuFoo - driverStatus = %d\n", driverStatus);
               assert(0);
           }
       }

       // rest of code here
   }

在下一个示例中，我们讨论如何获取在 CUDA Toolkit 的某个次版本中发布的 API 的新版本。请注意，在 cuda.h 头文件中，将 ``cuDeviceGetUuid`` 升级为 _v2 的版本宏直到主版本边界才会生效。因此在 11.4+ 的发布版本期间，下面的示例说明了如何获取 _v2 版本。

注意在这种情况下，原始（不是 _v2 版本）的 typedef 看起来像：

.. code-block:: c++

   typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);

但 _v2 版本的 typedef 看起来像：

.. code-block:: c++

   typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);

.. code-block:: c++

   #include <cudaTypedefs.h>

   CUuuid uuid;
   CUdevice dev;
   CUresult status;
   int cudaVersion;
   CUdriverProcAddressQueryResult driverStatus;

   status = cuDeviceGet(&dev, 0); // Get device 0
   // handle status

   // Ensure a CUDA driver >= 11.4 is installed or we will get an error from cuGetProcAddress
   status = cuDriverGetVersion(&cudaVersion);
   if (cudaVersion >= 11040) {
      PFN_cuDeviceGetUuid_v11040 pfn_cuDeviceGetUuid;
      status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, 11040, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
      if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
         pfn_cuDeviceGetUuid(&uuid, dev);
      }
   }

.. _guidelines-for-cugetprocaddress:

4.20.4. cuGetProcAddress 的准则
-------------------------------

下面是使用 ``cuGetProcAddress`` 时需要牢记的准则。

- 传递给 ``cuGetProcAddress`` 的 CUDA 版本应编码为与 typedef 版本相匹配（不要使用诸如 CUDA_VERSION 这样的编译时常量，也不要使用诸如 ``cuDriverGetVersion`` 返回的动态版本）。
- 在调用 ``cuGetProcAddress`` 之前，检查当前的 driver 版本（例如通过 ``cuDriverGetVersion`` 获取）是否足够，否则预期会得到错误，或者可能返回意外的符号。

.. _guidelines-for-runtime-api-usage:

4.20.4.1. Runtime API 使用的准则
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除非另有说明，CUDA runtime API ``cudaGetDriverEntryPointByVersion`` 将具有与 driver 入口点 ``cuGetProcAddress`` 类似的准则，因为它允许用户请求特定的 CUDA driver 版本。

.. _determining-cugetprocaddress-failure-reasons:

4.20.5. 确定 cuGetProcAddress 失败原因
---------------------------------------

cuGetProcAddress 有两种类型的错误。它们是 (1) API/使用错误和 (2) 无法找到请求的 driver API。第一种错误类型将通过 CUresult 返回值从 API 返回错误代码。比如将 NULL 作为 ``pfn`` 变量传递或传递无效的 ``flags`` 。

第二种错误类型编码在 ``CUdriverProcAddressQueryResult *symbolStatus`` 中，可用于帮助区分 driver 无法找到请求符号的潜在问题。请看以下示例：

.. code-block:: c++

   // cuDeviceGetExecAffinitySupport was introduced in release CUDA 11.4
   #include <cuda.h>
   CUdriverProcAddressQueryResult driverStatus;
   cudaVersion = ...;
   status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
   if (CUDA_SUCCESS == status) {
       if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
           printf("We can use the new feature when you upgrade cudaVersion to 11.4, but CUDA driver is good to go!\n");
           // Indicating cudaVersion was < 11.4 but run against a CUDA driver >= 11.4
       }
       else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
           printf("Please update both CUDA driver and cudaVersion to at least 11.4 to use the new feature!\n");
           // Indicating driver is < 11.4 since string not found, doesn't matter what cudaVersion was
       }
       else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
           printf("You're using cudaVersion and CUDA driver >= 11.4, using new feature!\n");
           pfn();
       }
   }

第一个返回代码 ``CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`` 表示在 CUDA driver 中搜索时找到了 ``symbol`` ，但它是在提供的 ``cudaVersion`` 之后添加的。在示例中，将 ``cudaVersion`` 指定为 11030 或更低，并在 CUDA driver >= CUDA 11.4 上运行时，会得到 ``CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`` 的结果。这是因为 ``cuDeviceGetExecAffinitySupport`` 是在 CUDA 11.4 (11040) 中添加的。

第二个返回代码 ``CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`` 表示在 CUDA driver 中搜索时未找到 ``symbol`` 。这可能是由于多种原因，例如由于 driver 较旧而不支持 CUDA 函数，以及仅仅是拼写错误。在后一种情况下，类似于最后一个示例，如果用户将 ``symbol`` 设为 CUDeviceGetExecAffinitySupport——注意开头的字母 CU 是大写的——``cuGetProcAddress`` 将无法找到该 API，因为字符串不匹配。在前一种情况下，示例可能是用户针对支持新 API 的 CUDA driver 开发应用程序，并将应用程序部署在较旧的 CUDA driver 上。使用最后一个示例，如果开发人员针对 CUDA 11.4 或更高版本进行开发，但部署在 CUDA 11.3 driver 上，在开发期间他们可能成功调用了 ``cuGetProcAddress`` ，但在部署针对 CUDA 11.3 driver 运行的应用程序时，该调用将不再工作，并在 ``driverStatus`` 中返回 ``CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`` 。