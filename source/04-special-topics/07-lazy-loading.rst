.. _lazy-loading-details:

4.7. Lazy Loading
=================

.. _lazy-loading-introduction:

4.7.1. 简介
-----------

延迟加载通过推迟 CUDA 模块加载直到需要时才加载，从而减少程序初始化时间。
延迟加载对于只使用少量 kernel 的程序特别有效，这在使用库时很常见。
当遵循 CUDA 编程模型时，延迟加载对用户是透明的。
:ref:`lazy-loading-potential-hazards` 详细解释了这一点。
从 CUDA 12.3 开始，延迟加载在所有平台上默认启用，但可以通过 ``CUDA_MODULE_LOADING`` 环境变量进行控制。

.. _lazy-loading-change-history:

4.7.2. 变更历史
---------------

.. _lazy-loading-version-history:
.. list-table:: 按 CUDA 版本选择的延迟加载变更
   :header-rows: 1

   * - CUDA 版本
     - 变更
   * - 12.3
     - 延迟加载性能改进。现在在 Windows 上默认启用。
   * - 12.2
     - 延迟加载在 Linux 上默认启用。
   * - 11.7
     - 延迟加载首次引入，默认禁用。

.. _requirements-for-lazy-loading:

4.7.3. 延迟加载的要求
---------------------

延迟加载是 CUDA runtime 和 driver 的联合功能。只有满足 runtime 和 driver 版本要求时，延迟加载才可用。

.. _cuda-runtime-version-requirement:

4.7.3.1. CUDA Runtime 版本要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

延迟加载从 CUDA runtime 版本 11.7 开始可用。
由于 CUDA runtime 通常静态链接到程序和库中，只有使用 CUDA 11.7+ toolkit 编译的程序和库才能从延迟加载中受益。
使用较旧 CUDA runtime 版本编译的库将立即加载所有模块。

.. _cuda-driver-version-requirement:

4.7.3.2. CUDA Driver 版本要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

延迟加载需要 driver 版本 515 或更新版本。
即使使用 CUDA toolkit 11.7 或更新版本，对于早于 515 的 driver 版本，延迟加载也不可用。

.. _compiler-requirements:

4.7.3.3. 编译器要求
^^^^^^^^^^^^^^^^^^^

延迟加载不需要任何编译器支持。
使用 11.7 之前编译器编译的 SASS 和 PTX 都可以在启用延迟加载的情况下加载，并将获得该功能的全部好处。
但是，仍然需要 11.7+ 版本的 CUDA runtime，如上所述。

.. _kernel-requirements:

4.7.3.4. Kernel 要求
^^^^^^^^^^^^^^^^^^^^

延迟加载不影响包含托管变量的模块，这些模块仍将被急切地加载。

.. _lazy-loading-usage:

4.7.4. 使用
-----------

.. _lazy-loading-enabling-disabling:

4.7.4.1. 启用和禁用
^^^^^^^^^^^^^^^^^^^^

通过将 ``CUDA_MODULE_LOADING`` 环境变量设置为 ``LAZY`` 来启用延迟加载。通过将 ``CUDA_MODULE_LOADING`` 环境变量设置为 ``EAGER`` 来禁用延迟加载。从 CUDA 12.3 开始，延迟加载在所有平台上默认启用。

.. _checking-if-lazy-loading-is-enabled-at-runtime:

4.7.4.2. 在运行时检查延迟加载是否启用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以使用 CUDA driver API 中的 ``cuModuleGetLoadingMode`` API 来确定是否启用了延迟加载。请注意，在运行此函数之前必须初始化 CUDA。下面的代码片段显示了示例用法。

.. code-block:: c
   :caption: 检查延迟加载模式示例

   #include <cuda.h>
   #include <assert.h>
   #include <iostream>

   int main() {
       CUmoduleLoadingMode mode;

       assert(CUDA_SUCCESS == cuInit(0));
       assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

       std::cout << "CUDA Module Loading Mode is "
                 << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager")
                 << std::endl;

       return 0;
   }

.. _forcing-a-module-to-load-eagerly-at-runtime:

4.7.4.3. 在运行时强制模块立即加载
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

加载 kernel 和变量是自动发生的，无需显式加载。
即使不执行 kernel，也可以通过以下方式显式加载 kernel：

- ``cuModuleGetFunction()`` 函数将导致模块加载到设备内存中
- ``cudaFuncGetAttributes()`` 函数将导致 kernel 加载到设备内存中

.. note::

   ``cuModuleLoad()`` 不保证模块会立即加载。

.. _lazy-loading-potential-hazards:

4.7.5. 潜在风险
---------------

延迟加载的设计使得应用程序不需要任何修改即可使用它。尽管如此，还是有一些注意事项，特别是当应用程序不完全符合 CUDA 编程模型时，如下所述。

.. _impact-on-concurrent-kernel-execution:

4.7.5.1. 对并发 Kernel 执行的影响
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些程序错误地假设并发 kernel 执行是有保证的。如果需要跨 kernel 同步，但 kernel 执行已被序列化，则可能发生死锁。为了尽量减少延迟加载对并发 kernel 执行的影响，请执行以下操作：

- 在启动之前预加载希望并发执行的所有 kernel，或
- 使用 ``CUDA_MODULE_LOADING = EAGER`` 运行应用程序，以强制急切地加载数据，而不强制每个函数都急切地加载

.. _large-memory-allocations:

4.7.5.2. 大内存分配
^^^^^^^^^^^^^^^^^^^

延迟加载将 CUDA 模块的内存分配从程序初始化延迟到更接近执行时间。如果应用程序在启动时分配了整个 VRAM，CUDA 可能无法在运行时为模块分配内存。可能的解决方案：

- 使用 ``cudaMallocAsync()`` 代替在启动时分配整个 VRAM 的分配器
- 添加一些缓冲区来补偿 kernel 的延迟加载
- 在尝试初始化分配器之前预加载程序中将使用的所有 kernel

.. _impact-on-performance-measurements:

4.7.5.3. 对性能测量的影响
^^^^^^^^^^^^^^^^^^^^^^^^^

延迟加载可能会通过将 CUDA 模块初始化移动到测量执行窗口中来影响性能测量。为了避免这种情况：

- 在测量之前至少执行一次预热迭代
- 在启动之前预加载要基准测试的 kernel