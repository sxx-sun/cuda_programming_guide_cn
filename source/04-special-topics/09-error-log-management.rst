.. _error-log-management-details:

4.9. 错误日志管理
==========================

错误日志管理（Error Log Management）机制允许 CUDA API 错误以通俗易懂的英语格式向开发者报告，描述问题的原因。

.. _error-log-management-background:

4.9.1. 背景
------------------

传统上，CUDA API 调用失败的唯一指示是非零代码的返回。截至 CUDA Toolkit 12.9，CUDA Runtime 为错误条件定义了超过 100 种不同的返回代码，但其中许多是通用的，无法帮助开发者调试原因。

.. _error-log-management-activation:

4.9.2. 激活
------------------

设置 ``CUDA_LOG_FILE`` 环境变量。可接受的值为 ``stdout`` 、 ``stderr`` 或系统中用于写入文件的有效路径。
即使程序执行前未设置 ``CUDA_LOG_FILE`` ，也可以通过 API 转储日志缓冲区。

.. note::
   无错误的执行可能不会打印任何日志。

.. _error-log-management-output:

4.9.3. 输出
------------------

日志按以下格式输出：

.. code-block:: c++

   [Time][TID][Source][Severity][API Entry Point] Message

以下是一行实际的错误消息，当开发者尝试将错误日志管理日志转储到未分配的缓冲区时会生成：

.. code-block:: c++

   [22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL

而在之前，开发者只能从返回代码中获得 ``CUDA_ERROR_INVALID_VALUE`` ，如果调用 ``cuGetErrorString`` 可能会得到 `invalid argument` 。

.. _error-log-management-api-description:

4.9.4. API 描述
----------------------

错误日志管理可通过 CUDA Driver 和 Runtime API 两者使用。Runtime 接口操作由 Runtime 和 Driver API 两者生成的消息；两个接口使用相同的底层日志缓冲区。

.. _callback-registration:

4.9.4.1. 回调注册
^^^^^^^^^^^^^^^^^^^^^^

Driver API 回调类型和注册函数如下：

.. code-block:: c++

   typedef void (CUDA_CB *CUlogsCallback)(
       void *data, CUlogLevel logLevel, char *message, size_t length);

   CUresult CUDAAPI cuLogsRegisterCallback(
       CUlogsCallback callbackFunc,
       void *userData,
       CUlogsCallbackHandle *callback_out);

   CUresult CUDAAPI cuLogsUnregisterCallback(
       CUlogsCallbackHandle callback);

对应的 Runtime API 回调类型和函数如下：

.. code-block:: c++

   typedef void (CUDART_CB *cudaLogsCallback_t)(
       void *data, cudaLogLevel logLevel, char *message, size_t length);

   cudaError_t CUDARTAPI cudaLogsRegisterCallback(
       cudaLogsCallback_t callbackFunc,
       void *userData,
       cudaLogsCallbackHandle *callback_out);

   cudaError_t CUDARTAPI cudaLogsUnregisterCallback(
       cudaLogsCallbackHandle callback);

Driver 回调将错误报告为 ``CU_LOG_LEVEL_ERROR`` ，将警告报告为 ``CU_LOG_LEVEL_WARNING`` 。对应的 Runtime 值为 ``cudaLogLevelError`` 和 ``cudaLogLevelWarning`` 。格式化输出分别以 ``E`` 和 ``W`` 表示这些级别。

CUDA 将 ``userData`` 不经修改地传递给回调。 ``callback_out`` 参数是可选的，但如果应用程序打算注销回调，则必须保留返回的句柄。

回调接收消息文本及其显式的字节长度。该消息不包含添加到格式化文件输出和转储输出中的时间戳、线程 ID、来源或严重性前缀。除非 API 参考明确保证特定行为，否则应用程序不应依赖特定的回调线程或调用时机。

.. _iterators-and-log-dumps:

4.9.4.2. 迭代器和日志转储
^^^^^^^^^^^^^^^^^^^^^^^^^

日志迭代器（log iterator）标识内部缓冲区中的一个位置。调用 ``cuLogsCurrent`` 或 ``cudaLogsCurrent`` 会将迭代器设置为缓冲区的当前末尾，因此之后使用该迭代器进行的转储仅返回随后生成的消息。

Driver API 函数如下：

.. code-block:: c++

   CUresult CUDAAPI cuLogsCurrent(
       CUlogIterator *iterator_out, unsigned int flags);

   CUresult CUDAAPI cuLogsDumpToFile(
       CUlogIterator *iterator,
       const char *pathToFile,
       unsigned int flags);

   CUresult CUDAAPI cuLogsDumpToMemory(
       CUlogIterator *iterator,
       char *buffer,
       size_t *size,
       unsigned int flags);

对应的 Runtime API 函数如下：

.. code-block:: c++

   cudaError_t CUDARTAPI cudaLogsCurrent(
       cudaLogIterator *iterator_out, unsigned int flags);

   cudaError_t CUDARTAPI cudaLogsDumpToFile(
       cudaLogIterator *iterator,
       const char *pathToFile,
       unsigned int flags);

   cudaError_t CUDARTAPI cudaLogsDumpToMemory(
       cudaLogIterator *iterator,
       char *buffer,
       size_t *size,
       unsigned int flags);

所有 current 和 dump 函数的 ``flags`` 参数保留供未来使用，必须为 0。

以下 Runtime API 示例标记日志的当前末尾，之后仅转储该时间点之后生成的消息：

.. code-block:: c++

   #include <cstdio>
   #include <cuda_runtime_api.h>

   cudaError_t printNewCudaLogs()
   {
       cudaLogIterator iterator;
       cudaError_t status = cudaLogsCurrent(&iterator, 0);
       if (status != cudaSuccess) {
           return status;
       }

       int deviceCount;
       status = cudaGetDeviceCount(&deviceCount);
       cudaError_t deviceCountStatus = status;

       char buffer[25600];
       size_t size = sizeof(buffer);
       status = cudaLogsDumpToMemory(&iterator, buffer, &size, 0);
       if (status != cudaSuccess) {
           return status;
       }
       if (size != 0) {
           std::fwrite(buffer, 1, size, stderr);
       }
       return deviceCountStatus;
   }

中间的 CUDA 调用仍然需要正常的返回码处理。此示例演示了迭代器如何选择日志消息的一个窗口；它不需要故意进行无效的 CUDA 调用。固定大小的数组使用了文档所述的最大转储容量 25,600 字节。

.. _dump-behavior:

4.9.4.3. 转储行为
^^^^^^^^^^^^^^^^^^^^^^

如果 ``iterator`` 为 NULL，转储将请求内部缓冲区中当前保留的所有条目。如果提供了迭代器，转储将从该位置开始。成功转储后，CUDA 会将所提供的迭代器推进到日志的当前末尾。

内部缓冲区最多保留 100 条条目。如果所请求的消息已被覆盖且不再保留，CUDA 会添加一条说明输出已被截断的注释，并返回仍然可用的条目。

``cuLogsDumpToMemory`` 和 ``cudaLogsDumpToMemory`` 还有以下额外要求：

#. ``size`` 最初指定 ``buffer`` 的容量。

#. CUDA 会以 null 终止缓冲区。各个条目之间用换行符分隔，而不是 null 字符。

#. 25,600 字节的输出缓冲区足以容纳所有保留的日志数据。

#. 成功时， ``size`` 包含写入的字节数，不包括终止 null 字符。

#. 如果没有要转储的消息，函数将成功执行并将 ``size`` 设置为 0。

#. 如果缓冲区能容纳部分但并非全部所请求的消息，CUDA 会在开头添加一条截断注释，省略无法容纳的最旧消息，并在输出中保留最近的消息。

#. 如果缓冲区无法容纳任何消息，函数将返回 ``CUDA_ERROR_INVALID_VALUE`` 或 ``cudaErrorInvalidValue`` ，并将 ``size`` 设置为 0。

.. _error-log-management-limitations:

4.9.5. 限制和已知问题
----------------------------

#. 日志缓冲区限制为 100 条条目。达到此限制后，最旧的条目将被替换，日志转储将包含一条说明翻转的行。

#. 并非所有 CUDA API 都已覆盖。这是一个持续进行的项目，旨在为所有 API 提供更好的使用错误报告。

#. 错误日志管理的日志位置（如果给定）将不会测试其有效性，直到/除非生成日志。

#. 错误日志管理 API 目前仅通过 CUDA Driver 提供。CUDA Runtime API 将在未来版本中添加。

#. 日志消息未本地化为任何语言，所有提供的日志均为美式英语。