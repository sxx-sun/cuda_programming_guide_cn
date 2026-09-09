.. _cpp-language-extensions:

5.4. C/C++ 语言扩展
===================

.. _function-and-variable-annotations:

5.4.1. 函数和变量注释
---------------------

.. _execution-space-specifiers:

5.4.1.1. 执行空间说明符
^^^^^^^^^^^^^^^^^^^^^^^

执行空间说明符 ``__host__``、``__device__``、``__tile__``、``__global__`` 和 ``__tile_global__`` 指示函数是在主机、SIMT 还是 tile 上下文中执行。

.. list-table:: 执行空间说明符
   :header-rows: 2
   :widths: 25 11 11 11 11 11 11

   * - 执行空间说明符
     - 执行位置
     -
     -
     - 可调用位置
     -
     -
   * -
     - 主机
     - SIMT
     - Tile
     - 主机
     - SIMT
     - Tile
   * - ``__host__``，无说明符
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
   * - ``__device__``
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌
   * - ``__global__``
     - ❌
     - ✅
     - ❌
     - ✅
     - ✅
     - ❌
   * - ``__tile__``
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
   * - ``__tile_global__``
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
   * - ``__host__ __device__ __tile__``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

``__global__`` 和 ``__tile_global__`` 函数的约束：

- 必须返回 ``void``。
- 不能是 ``class``、``struct`` 或 ``union`` 的成员。
- 需要执行配置，如 :ref:`内核配置 <kernel-configuration>` 中所述。
- 不支持递归。
- 有关其他限制，请参阅 ``__global__`` :ref:`函数参数 <global-function-parameters>`。

对 ``__global__`` 和 ``__tile_global__`` 函数的调用是异步的。它们在设备完成执行之前返回到主机线程。

用多个执行空间（例如 ``__host__ __device__``）声明的函数会为每个上下文编译。可以使用 ``__CUDA_ARCH__`` :ref:`宏 <cuda-arch-macro>` 区分主机和设备代码路径：

.. code-block:: c++

   __host__ __device__ void func() {
   #if defined(__CUDA_ARCH__)
       // Device code path
   #else
       // Host code path
   #endif
   }

.. _memory-space-specifiers:

5.4.1.2. 内存空间说明符
^^^^^^^^^^^^^^^^^^^^^^^

内存空间说明符 ``__device__``、``__tile__``、``__managed__``、``__constant__`` 和 ``__shared__`` 指示设备上变量的存储位置。

下表总结了内存空间属性：

.. list-table:: 内存空间说明符
   :header-rows: 1
   :widths: 18 20 22 16 12

   * - 内存空间说明符
     - 位置
     - 可访问者
     - 生存期
     - 唯一实例
   * - ``__device__``
     - 设备全局内存
     - 设备线程（grid）/ CUDA Runtime API
     - 程序/:ref:`CUDA 上下文 <driver-api-context>`
     - 每设备
   * - ``__tile__``
     - 设备全局内存
     - Tile 块 / CUDA Runtime API
     - 程序/:ref:`CUDA 上下文 <driver-api-context>`
     - 每设备
   * - ``__constant__``
     - 设备常量内存
     - 设备线程（grid）/ CUDA Runtime API
     - 程序/:ref:`CUDA 上下文 <driver-api-context>`
     - 每设备
   * - ``__managed__``
     - 主机和设备（自动）
     - 主机/设备线程
     - 程序
     - 每程序
   * - ``__shared__``
     - 设备（流多处理器）
     - 块线程
     - 块
     - 块
   * - 无说明符
     - 设备（寄存器）
     - 单个线程
     - 单个线程
     - 单个线程

- ``__device__``、``__tile__`` 和 ``__constant__`` 变量可以通过 `CUDA Runtime API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>`__ 函数 ``cudaGetSymbolAddress()``、``cudaGetSymbolSize()``、``cudaMemcpyToSymbol()`` 和 ``cudaMemcpyFromSymbol()`` 从主机访问。

- ``__constant__`` 变量在设备代码中是只读的，只能使用 `CUDA Runtime API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>`__ 从主机修改。

以下示例说明如何使用这些 API：

.. code-block:: c++

   __device__   float device_var       = 4.0f; // Variable in device memory
   __constant__ float constant_mem_var = 4.0f; // Variable in constant memory
                                               // For readability, the following example focuses on a device variable.
   int main() {
       float* device_ptr;
       cudaGetSymbolAddress((void**) &device_ptr, device_var);        // Gets address of device_var

       size_t symbol_size;
       cudaGetSymbolSize(&symbol_size, device_var);                   // Retrieves the size of the symbol (4 bytes).

       float host_var;
       cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // Copies from device to host.

       host_var = 3.0f;
       cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // Copies from host to device.
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/vYjP8GGv3>`__ 上的示例。

.. _shared-memory:

5.4.1.2.1. ``__shared__`` 内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``__shared__`` 内存变量可以具有静态大小（在编译时确定）或动态大小（在内核启动时确定）。有关在运行时指定共享内存大小的详细信息，请参阅 :ref:`内核配置 <kernel-configuration>` 部分。

共享内存限制：

- 具有动态大小的变量必须声明为外部数组或指针。
- 具有静态大小的变量不能在其声明中初始化。

以下示例说明如何声明和确定 ``__shared__`` 变量的大小：

.. code-block:: c++

   extern __shared__ char dynamic_smem_pointer[];
   // extern __shared__ char* dynamic_smem_pointer; alternative syntax

   __global__ void kernel() { // or a __device__ function
       __shared__ int smem_var1[4];                  // static size
       auto smem_var2 = (int*) dynamic_smem_pointer; // dynamic size
   }

   int main() {
       size_t shared_memory_size = 16;
       kernel<<<1, 1, shared_memory_size>>>();
       cudaDeviceSynchronize();
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/nPjvd1frb>`__ 上的示例。

.. _managed-memory:

5.4.1.2.2. ``__managed__`` 内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``__managed__`` 变量具有以下限制：

- ``__managed__`` 变量的地址不是常量表达式。
- ``__managed__`` 变量不应具有引用类型 ``T&`` 。
- 当 CUDA 运行时可能未处于有效状态时，不应使用 ``__managed__`` 变量的地址或值，包括以下情况：
  
  - 在具有 ``static`` 或 ``thread_local`` 存储期的对象的静态/动态初始化或销毁中。
  - 在调用 ``exit()`` 后执行的代码中。例如，标记为 ``__attribute__((destructor))`` 的函数。
  - 在 CUDA 运行时可能未初始化时执行的代码中。例如，标记为 ``__attribute__((constructor))`` 的函数。

- ``__managed__`` 变量不能用作 ``decltype()`` 表达式的未括号化 id 表达式参数。
- ``__managed__`` 变量具有与 :ref:`动态分配的托管内存 <memory-unified-memory>` 指定的相同的一致性和一致性行为。
- 另请参阅 :ref:`局部变量 <local-variables>` 的限制。

以下是 ``__managed__`` 变量合法与非法用法的示例：

.. code-block:: c++

   #include <cassert>

   __device__ __managed__ int global_var = 10; // OK

   int* ptr = &global_var;                     // ERROR: use of a managed variable in static initialization

   struct MyStruct1 {
       int field;
       MyStruct1() : field(global_var) {};
   };

   struct MyStruct2 {
       ~MyStruct2() { global_var = 10; }
   };

   MyStruct1 temp1; // ERROR: use of managed variable in dynamic initialization

   MyStruct2 temp2; // ERROR: use of managed variable in the destructor of
                    //        object with static storage duration

   __device__ __managed__ const int const_var = 10;         // ERROR: const-qualified type

   __device__ __managed__ int&      reference = global_var; // ERROR: reference type

   template <int* Addr>
   struct MyStruct3 {};

   MyStruct3<&global_var> temp;     // ERROR: address of managed variable is not a constant expression

   __global__ void kernel(int* ptr) {
       assert(ptr == &global_var);  // OK
       global_var = 20;             // OK
   }

   int main() {
       int* ptr = &global_var;      // OK
       kernel<<<1, 1>>>(ptr);
       cudaDeviceSynchronize();
       global_var++;                // OK
       decltype(global_var) var1;   // ERROR: managed variable used as unparenthesized argument to decltype

       decltype((global_var)) var2; // OK
   }

.. _tile-variables:

5.4.1.2.3. ``__tile__`` 变量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``__tile__`` 内存空间说明符类似于 ``__device__``。用 ``__tile__`` 标记的变量分配在设备全局内存中，可以由 CUDA Runtime API 函数访问。与 ``__device__`` 变量不同，``__tile__`` 变量可以直接从 tile 代码访问。

通常，``__device__`` 变量不能从 ``__tile__`` 或 ``__tile_global__`` 代码按名直接访问，``__tile__`` 变量也不能从 ``__device__`` 或 ``__global__`` 代码按名直接访问。但是，这两种变量都分配在设备全局内存中，指向该内存的指针可以在任一上下文中访问。

以下示例展示了 tile 变量的合法和非法用法：

.. code-block:: c++

   __device__ int x_device = 0;
   __tile__   int x_tile = 0;

   // ERROR: Variable cannot be both device and tile
   __device__ __tile__ int x_device_tile = 0;

   __global__ void simt_function(int* in) {
     x_device = 1; // OK: __device__ variable accessed from SIMT code
     x_tile = 1;   // ERROR: __tile__ variable accessed from SIMT code

     *in = 1; // OK: May point to any device global memory, including x_tile.
   }

   __tile_global__ void tile_function(int* in) {
     x_device = 1; // ERROR: __device__ variable accessed from tile code
     x_tile = 1;   // OK: __tile__ variable accessed from tile code

     *in = 1; // OK: May point to any device global memory, including x_device.
   }

   int main() {
     int* x_device_ptr;
     cudaGetSymbolAddress((void**) &x_device_ptr, x_device);    // Gets address of x_device

     int* x_tile_ptr;
     cudaGetSymbolAddress((void**) &x_tile_ptr, x_tile);        // Gets address of x_tile

     // OK: Passing pointer to tile variable into SIMT kernel
     simt_function<<<1,1>>>(x_tile_ptr);
     cudaDeviceSynchronize();

     // OK: Passing pointer to device variable into tile kernel
     tile_function<<<1,1>>>(x_device_ptr);
     cudaDeviceSynchronize();
   }

``__tile__`` 变量不能包含指针或引用类型的子对象。例如：

.. code-block:: c++

   __tile__ int* ptr; // ERROR

   struct S1 { int* ptr; };
   __tile__ S1 val; // ERROR

.. _inline-specifiers:

5.4.1.3. 内联说明符
^^^^^^^^^^^^^^^^^^^

以下说明符可用于控制 ``__host__`` 和 ``__device__`` 函数的内联：

- ``__noinline__`` ：指示 ``nvcc`` 不要内联该函数。
- ``__forceinline__`` ：强制 ``nvcc`` 在单个翻译单元内内联该函数。
- ``__inline_hint__`` ：使用 :ref:`链接时优化 <nvcc-link-time-optimization>` 时启用跨翻译单元的积极内联。

这些说明符是互斥的。这些说明符应用于 ``__tile__`` 函数时被忽略。

.. _restrict-pointers:

5.4.1.4. ``__restrict__`` 指针
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nvcc`` 通过 ``__restrict__`` 关键字支持受限指针。

当两个或多个指针引用重叠的内存区域时，就会发生指针别名。这可能会抑制代码重排序和公共子表达式消除等优化。

受限限定指针是程序员的一个承诺，即在该指针的生存期内，它指向的内存只能通过该指针访问。这允许编译器执行更积极的优化。

- 访问设备函数的所有线程只从中读取；或者
- 最多有一个线程写入其中，没有其他线程从中读取。

以下示例说明别名问题，并演示如何使用受限指针帮助编译器减少指令数量：

.. code-block:: c++

   __device__
   void device_function(const float* a, const float* b, float* c) {
       c[0] = a[0] * b[0];
       c[1] = a[0] * b[0];
       c[2] = a[0] * b[0] * a[1];
       c[3] = a[0] * a[1];
       c[4] = a[0] * b[0];
       c[5] = b[0];
       ...
   }

由于指针 ``a`` 、 ``b`` 和 ``c`` 可能别名化，通过 ``c`` 的任何写入都可能修改 ``a`` 或 ``b`` 的元素。为保证功能正确性，编译器无法将 ``a[0]`` 和 ``b[0]`` 加载到寄存器中、相乘并将结果存储在 ``c[0]`` 和 ``c[1]`` 中。这是因为如果 ``a[0]`` 和 ``c[0]`` 位于同一位置，结果将与抽象执行模型不同。编译器无法利用公共子表达式。类似地，编译器无法重排 ``c[4]`` 的计算与 ``c[0]`` 和 ``c[1]`` 的计算，因为前面的 ``c[3]`` 写入可能会改变 ``c[4]`` 计算的输入。

通过将 ``a`` 、 ``b`` 和 ``c`` 声明为受限指针，程序员通知编译器这些指针没有别名。这意味着写入 ``c`` 永远不会覆盖 ``a`` 或 ``b`` 的元素。这会将函数原型更改如下：

.. code-block:: c++

   __device__
   void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);

请注意，所有指针参数都必须是受限的，编译器优化器才能有效。添加 ``__restrict__`` 关键字后，编译器可以随意重排并执行公共子表达式消除，同时保持与抽象执行模型相同的功能。

.. code-block:: c++

   __device__
   void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
       float t0 = a[0];
       float t1 = b[0];
       float t2 = t0 * t1;
       float t3 = a[1];
       c[0]     = t2;
       c[1]     = t2;
       c[4]     = t2;
       c[2]     = t2 * t3;
       c[3]     = t0 * t3;
       c[5]     = t1;
       ...
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/6KeTqarnW>`__ 上的示例。

结果是减少了内存访问和计算次数，但由于在寄存器中缓存加载和公共子表达式，寄存器压力增加。

由于寄存器压力是许多 CUDA 代码中的关键问题，使用受限指针可能会通过降低占用率对性能产生负面影响。

访问标记为 ``__restrict__`` 的 ``__global__`` 函数 ``const`` 指针被编译为只读缓存加载，类似于 `PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`__ ``ld.global.nc`` 或 ``__ldg()`` :ref:`低级加载和存储函数 <low-level-load-store-functions>` 指令。

.. code-block:: c++

   __global__
   void kernel1(const float* in, float* out) {
       *out = *in; // PTX: ld.global
   }

   __global__
   void kernel2(const float* __restrict__ in, float* out) {
       *out = *in;  // PTX: ld.global.nc
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/drsTEPa8s>`__ 上的示例。

.. _grid-constant-parameters:

5.4.1.5. ``__grid_constant__`` 参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``__grid_constant__`` 注释 ``__global__`` 函数参数可防止编译器创建该参数的每线程副本。相反，grid 中的所有线程将通过单个地址访问该参数，这可以提高性能。

``__grid_constant__`` 参数具有以下属性：

- 它具有内核的生存期。
- 它对于单个内核是私有的，意味着该对象不可被来自其他 grid 的线程访问，包括子 grid。
- 内核中的所有线程看到相同的地址。
- 它是只读的。修改 ``__grid_constant__`` 对象或其任何子对象（包括 ``mutable`` 成员）是未定义行为。

要求：

- 用 ``__grid_constant__`` 注释的内核参数必须具有 ``const`` 限定的非引用类型。
- 所有函数声明必须与任何 ``__grid_constant__`` 参数一致。
- 函数模板特化必须与主模板声明匹配，关于任何 ``__grid_constant__`` 参数。
- 函数模板实例化也必须与主模板声明匹配，关于任何 ``__grid_constant__`` 参数。
- ``__grid_constant__`` 注释在 ``__tile_global__`` 函数参数中被忽略。

示例：

.. code-block:: c++

   struct MyStruct {
       int         x;
       mutable int y;
   };

   __device__ void external_function(const MyStruct&);

   __global__ void kernel(const __grid_constant__ MyStruct s) {
       // s.x++; // Compile error: tried to modify read-only memory
       // s.y++; // Undefined Behavior: tried to modify read-only memory

       // Compiler will NOT create a per-thread local copy of "s":
       external_function(s);
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/Goq9jrEeo>`__ 上的示例。

.. _annotation-summary:

5.4.1.6. 注释摘要
^^^^^^^^^^^^^^^^^

下表总结了 CUDA 注释，并报告每个注释适用于哪个执行空间以及它在何处有效。

.. list-table:: 注释摘要
   :header-rows: 1
   :widths: 35 35 30

   * - 注释
     - ``__host__`` / ``__device__`` / ``__host__ __device__``
     - ``__global__``
   * - :ref:`__noinline__ <inline-specifiers>`、:ref:`__forceinline__ <inline-specifiers>`、:ref:`__inline_hint__ <inline-specifiers>`
     - 函数
     - ❌
   * - :ref:`__restrict__ <restrict-pointers>`
     - 指针参数
     - 指针参数
   * - :ref:`__grid_constant__ <grid-constant-parameters>`
     - ❌
     - 参数
   * - :ref:`__launch_bounds__ <launch-bounds>`
     - ❌
     - 函数
   * - :ref:`__maxnreg__ <maximum-number-of-registers-per-thread>`
     - ❌
     - 函数
   * - :ref:`__cluster_dims__ <thread-block-cluster>`
     - ❌
     - 函数

.. _built-in-types-and-variables:

5.4.2. 内建类型和变量
---------------------

.. _host-compiler-type-extensions:

5.4.2.1. 主机编译器类型扩展
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 允许使用非标准算术类型，只要主机编译器支持。支持以下类型：

- 128 位整数类型 ``__int128`` 。
  
  - 当主机编译器定义 ``__SIZEOF_INT128__`` 宏时，在 Linux 上支持。

- 128 位浮点类型 ``__float128`` 和 ``_Float128`` 在计算能力 10.0 及更高版本的 GPU 设备上可用。 ``__float128`` 类型的常量表达式可能由编译器以较低精度的浮点表示形式处理。
  
  - 当主机编译器定义 ``__SIZEOF_FLOAT128__`` 或 ``__FLOAT128__`` 宏时，在 Linux x86 上支持。

- ``_Complex`` `类型 <https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Complex-Data-Types.html>`__ 仅在主机代码中支持。

128 位整数和浮点类型在 tile 代码中不受支持。

.. _built-in-variables:

5.4.2.2. 内建变量
^^^^^^^^^^^^^^^^^

用于指定和检索沿 x、y 和 z 维度的 grid 和块的内核配置的值是 ``dim3`` 类型。用于获取块和线程索引的变量是 ``uint3`` 类型。 ``dim3`` 和 ``uint3`` 都是由三个名为 ``x`` 、 ``y`` 和 ``z`` 的无符号值组成的简单结构。在 C++11 及更高版本中， ``dim3`` 的所有组件的默认值为 1。

仅限设备的内建变量：

- ``dim3 gridDim`` ：包含 grid 的维度，即沿 x、y 和 z 维度的线程块数量。

- ``dim3 blockDim`` ：包含线程块的维度，即沿 x、y 和 z 维度的线程数量。

- ``uint3 blockIdx`` ：包含 grid 内的块索引，沿 x、y 和 z 维度。

- ``uint3 threadIdx`` ：包含块内的线程索引，沿 x、y 和 z 维度。

- ``int warpSize`` ：运行时值，定义为 warp 中的线程数，通常为 ``32`` 。另请参阅 :ref:`Warp 和 SIMT <warps-and-simt>` 了解 warp 的定义。

这些变量在 tile 代码中不受支持。要在 tile 代码中检索块 ID 或 grid 大小，请使用 ``cuda::tiles::bid()`` 和 ``cuda::tiles::num_blocks()`` API。

.. _built-in-types:

5.4.2.3. 内建类型
^^^^^^^^^^^^^^^^^

CUDA 提供从基本整数和浮点类型派生的向量类型，这些类型在主机和设备上都受支持。

.. list-table:: 向量类型
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - C++ 基本类型
     - 向量 X1
     - 向量 X2
     - 向量 X3
     - 向量 X4
   * - ``signed char``
     - ``char1``
     - ``char2``
     - ``char3``
     - ``char4``
   * - ``unsigned char``
     - ``uchar1``
     - ``uchar2``
     - ``uchar3``
     - ``uchar4``
   * - ``signed short``
     - ``short1``
     - ``short2``
     - ``short3``
     - ``short4``
   * - ``unsigned short``
     - ``ushort1``
     - ``ushort2``
     - ``ushort3``
     - ``ushort4``
   * - ``signed int``
     - ``int1``
     - ``int2``
     - ``int3``
     - ``int4``
   * - ``unsigned``
     - ``uint1``
     - ``uint2``
     - ``uint3``
     - ``uint4``
   * - ``signed long``
     - ``long1``
     - ``long2``
     - ``long3``
     - ``long4_16a/long4_32a``
   * - ``unsigned long``
     - ``ulong1``
     - ``ulong2``
     - ``ulong3``
     - ``ulong4_16a/ulong4_32a``
   * - ``signed long long``
     - ``longlong1``
     - ``longlong2``
     - ``longlong3``
     - ``longlong4_16a/longlong4_32a``
   * - ``unsigned long long``
     - ``ulonglong1``
     - ``ulonglong2``
     - ``ulonglong3``
     - ``ulonglong4_16a/ulonglong4_32a``
   * - ``float``
     - ``float1``
     - ``float2``
     - ``float3``
     - ``float4``
   * - ``double``
     - ``double1``
     - ``double2``
     - ``double3``
     - ``double4_16a/double4_32a``

请注意，``long4``、``ulong4``、``longlong4``、``ulonglong4`` 和 ``double4`` 已在 CUDA 13 中弃用，并可能在未来版本中移除。

下表详细说明了向量类型的字节大小和对齐要求：

.. list-table:: 对齐要求
   :header-rows: 1
   :widths: 50 25 25

   * - 类型
     - 大小
     - 对齐
   * - ``char1``、``uchar1``
     - 1
     - 1
   * - ``char2``、``uchar2``
     - 2
     - 2
   * - ``char3``、``uchar3``
     - 3
     - 1
   * - ``char4``、``uchar4``
     - 4
     - 4
   * - ``short1``、``ushort1``
     - 2
     - 2
   * - ``short2``、``ushort2``
     - 4
     - 4
   * - ``short3``、``ushort3``
     - 6
     - 2
   * - ``short4``、``ushort4``
     - 8
     - 8
   * - ``int1``、``uint1``
     - 4
     - 4
   * - ``int2``、``uint2``
     - 8
     - 8
   * - ``int3``、``uint3``
     - 12
     - 4
   * - ``int4``、``uint4``
     - 16
     - 16
   * - ``long1``、``ulong1``
     - 4/8 \*
     - 4/8 \*
   * - ``long2``、``ulong2``
     - 8/16 \*
     - 8/16 \*
   * - ``long3``、``ulong3``
     - 12/24 \*
     - 4/8 \*
   * - ``long4``、``ulong4`` （已弃用）
     - 16/32 \*
     - 16 \*
   * - ``long4_16a``、``ulong4_16a``
     - 16/32 \*
     - 16
   * - ``long4_32a``、``ulong4_32a``
     - 16/32 \*
     - 32
   * - ``longlong1``、``ulonglong1``
     - 8
     - 8
   * - ``longlong2``、``ulonglong2``
     - 16
     - 16
   * - ``longlong3``、``ulonglong3``
     - 24
     - 8
   * - ``longlong4``、``ulonglong4`` （已弃用）
     - 32
     - 16
   * - ``longlong4_16a``、``ulonglong4_16a``
     - 32
     - 16
   * - ``longlong4_32a``、``ulonglong4_32a``
     - 32
     - 32
   * - ``float1``
     - 4
     - 4
   * - ``float2``
     - 8
     - 8
   * - ``float3``
     - 12
     - 4
   * - ``float4``
     - 16
     - 16
   * - ``double1``
     - 8
     - 8
   * - ``double2``
     - 16
     - 16
   * - ``double3``
     - 24
     - 8
   * - ``double4`` （已弃用）
     - 32
     - 16
   * - ``double4_16a``
     - 32
     - 16
   * - ``double4_32a``
     - 32
     - 32

\* ``long`` 在 C++ LLP64 数据模型（Windows 64 位）上为 4 字节，而在 C++ LP64 数据模型（Linux 64 位）上为 8 字节。

向量类型是结构体。它们的第一个、第二个、第三个和第四个组件分别可以通过 ``x`` 、 ``y`` 、 ``z`` 和 ``w`` 字段访问。

.. code-block:: c++

   int sum(int4 value) {
       return value.x + value.y + value.z + value.w;
   }

它们都有一个 ``make_<type_name>()`` 形式的工厂函数；例如：

.. code-block:: c++

   int4 add_one(int x, int y, int z, int w) {
       return make_int4(x + 1, y + 1, z + 1, w + 1);
   }

如果主机代码不是用 ``nvcc`` 编译的，可以通过包含 CUDA toolkit 中提供的 ``cuda_runtime.h`` 头文件导入向量类型和相关函数。

.. _kernel-configuration:

5.4.3. 内核配置
---------------

对 ``__global__`` 或 ``__tile_global__`` 函数的任何调用都必须为该调用指定*执行配置*。此执行配置定义将用于在设备上执行函数的 grid 和块的维度，以及关联的 :ref:`流 <cuda-streams>` 。

执行配置通过在函数名和括号参数列表之间插入 ``<<<grid_dim, block_dim, dynamic_smem_bytes, stream>>>`` 形式的表达式来指定，其中：

- ``grid_dim`` 是 :ref:`dim3 <built-in-variables>` 类型，指定 grid 的维度和大小，使得 ``grid_dim.x * grid_dim.y * grid_dim.z`` 等于正在启动的块数；

- ``block_dim`` 是 :ref:`dim3 <built-in-variables>` 类型，指定每个块的维度和大小，使得 ``block_dim.x * block_dim.y * block_dim.z`` 等于每个块的线程数；对于 ``__tile_global__`` 内核， ``block_dim`` 必须为 ``1`` ，因为编译器在启动 tile 内核时将决定调度多少线程。

- ``dynamic_smem_bytes`` 是可选的 ``size_t`` 参数，默认为零。它指定此次调用每个块动态分配的共享内存字节数，除了静态分配的内存。此内存由 ``extern __shared__`` 数组使用（参见 :ref:`__shared__ 内存 <shared-memory>` ）。

- ``stream`` 是 ``cudaStream_t`` （指针）类型，指定关联的流。 ``stream`` 是可选参数，默认为 ``NULL`` 。

以下示例显示内核函数声明和调用：

.. code-block:: c++

   __global__ void kernel(float* parameter);

   kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>(parameter);

执行配置的参数在实际函数参数之前计算。

如果 ``grid_dim`` 或 ``block_dim`` 超过设备允许的最大大小（如 :ref:`计算能力 <compute-capabilities>` 中指定），或者 ``dynamic_smem_bytes`` 大于考虑静态分配内存后的可用共享内存，函数调用将失败。

.. _thread-block-cluster:

5.4.3.1. 线程块集群
^^^^^^^^^^^^^^^^^^^

计算能力 9.0 及更高版本允许用户指定编译时线程块集群维度，以便内核可以使用 CUDA 中的 :ref:`集群层次结构 <intro-cuda-cpp-thread-block-cluster>` 。可以使用 ``__cluster_dims__`` 属性指定编译时集群维度，语法如下： ``__cluster_dims__([x, [y, [z]]])`` 。以下示例显示 X 维度为 2、Y 和 Z 维度为 1 的编译时集群大小。

.. code-block:: c++

   __global__ void __cluster_dims__(2, 1, 1) kernel(float* parameter);

``__cluster_dims__()`` 的默认形式指定内核将作为 grid 集群启动。如果未指定集群维度，用户可以在启动时指定。如果在启动时未能指定维度，将导致启动时错误。 ``__cluster_dims__`` 属性不能在 ``__tile_global__`` 内核上指定，因为编译器决定如何跨集群调度 tile 内核。

线程块集群的维度也可以在运行时指定，可以使用 ``cudaLaunchKernelEx`` API 启动带有集群的内核。此 API 接受 ``cudaLaunchConfig_t`` 类型的配置参数、内核函数指针和内核参数。以下示例显示运行时内核配置。

.. code-block:: c++

   __global__ void kernel(float parameter1, int parameter2) {}

   int main() {
       cudaLaunchConfig_t config = {0};
       // The grid dimension is not affected by cluster launch, and is still enumerated
       // using the number of blocks.
       // The grid dimension should be a multiple of cluster size.
       config.gridDim          = dim3{4};  // 4 blocks
       config.blockDim         = dim3{32}; // 32 threads per block
       config.dynamicSmemBytes = 1024;     // 1 KB

       cudaLaunchAttribute attribute[1];
       attribute[0].id               = cudaLaunchAttributeClusterDimension;
       attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
       attribute[0].val.clusterDim.y = 1;
       attribute[0].val.clusterDim.z = 1;
       config.attrs    = attribute;
       config.numAttrs = 1;

       float parameter1 = 3.0f;
       int   parameter2 = 4;
       cudaLaunchKernelEx(&config, kernel, parameter1, parameter2);
   }

另请参阅 `Compiler Explorer <https://cuda.godbolt.org/z/M67r3a5zM>`__ 上的示例。

.. _launch-bounds:

5.4.3.2. 启动边界
^^^^^^^^^^^^^^^^^

如 :ref:`内核启动和占用率 <writing-cuda-kernels-kernel-launch-and-occupancy>` 部分所述，使用更少的寄存器允许更多线程和线程块驻留在多处理器上，从而提高性能。

因此，编译器使用启发式方法最小化寄存器使用，同时将 :ref:`寄存器溢出 <writing-cuda-kernels-registers>` 和指令计数保持在最低限度。应用程序可以选择以启动边界（launch bounds）的形式向编译器提供额外信息，来辅助这些启发式方法。启动边界使用 ``__launch_bounds__()`` 限定符在 ``__global__`` 函数的定义中指定：

.. code-block:: c++

   __global__ void
   __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
   MyKernel(...) {
       ...
   }

- ``maxThreadsPerBlock`` 指定应用程序将启动 ``MyKernel()`` 的每个块的最大线程数；它编译为 ``.maxntid`` PTX 指令。

- ``minBlocksPerMultiprocessor`` 是可选的，指定每个多处理器所需的最小驻留块数；它编译为 ``.minnctapersm`` PTX 指令。

- ``maxBlocksPerCluster`` 是可选的，指定应用程序将启动 ``MyKernel()`` 的每个集群的最大线程块数；它编译为 ``.maxclusterrank`` PTX 指令。

如果指定了启动边界，编译器首先推导内核应使用的寄存器数量的上限 ``L`` 。这确保 ``minBlocksPerMultiprocessor`` 个 ``maxThreadsPerBlock`` 线程块可以驻留在多处理器上（如果未指定 ``minBlocksPerMultiprocessor`` ，则为单个块）。有关内核使用的寄存器数量与每个块分配的寄存器数量之间的关系，请参见 :ref:`占用率 <writing-cuda-kernels-kernel-launch-and-occupancy>` 部分。然后编译器按如下方式优化寄存器使用：

- 如果初始寄存器使用超过 ``L`` ，编译器会减少它，直到它小于或等于 ``L`` 。这通常会导致本地内存使用增加和/或指令数量增加。

- 如果初始寄存器使用低于 ``L``
  
  - 如果指定了 ``maxThreadsPerBlock`` 但未指定 ``minBlocksPerMultiprocessor`` ，编译器使用 ``maxThreadsPerBlock`` 确定从 ``n`` 到 ``n + 1`` 个驻留块过渡的寄存器使用阈值。当少使用一个寄存器就能为额外的驻留块腾出空间时，就会发生这种情况。然后，编译器应用与未指定启动边界时类似的启发式方法。
  
  - 如果同时指定了 ``minBlocksPerMultiprocessor`` 和 ``maxThreadsPerBlock`` ，编译器可能会将寄存器使用增加到 ``L`` ，以减少指令数量并更好地隐藏单线程指令的延迟。

如果使用以下方式执行，内核将无法启动：

- 每块线程数超过其启动边界 ``maxThreadsPerBlock`` 。
- 每集群线程块数超过其启动边界 ``maxBlocksPerCluster`` 。

CUDA 内核所需的每线程资源可能以不理想的方式限制最大块大小。为保持与未来硬件和工具包的前向兼容性，并确保至少有一个线程块可以在流多处理器上运行，开发者应包含单参数形式 ``__launch_bounds__(maxThreadsPerBlock)`` ，它指定内核将启动的最大块大小。未这样做可能导致「too many resources requested for launch」错误。在某些情况下，提供双参数形式 ``__launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)`` 可以提升性能。``minBlocksPerMultiprocessor`` 的最佳值应通过对每个内核的详细分析来确定。

内核的最优启动边界通常在主要架构修订之间有所不同。以下代码示例说明如何在设备代码中使用 ``__CUDA_ARCH__`` :ref:`宏 <cuda-arch-macro>` 来管理这一点。

.. code-block:: c++

   #define THREADS_PER_BLOCK  256

   #if __CUDA_ARCH__ >= 900
       #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
       #define MY_KERNEL_MIN_BLOCKS   3
   #else
       #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
       #define MY_KERNEL_MIN_BLOCKS   2
   #endif

   __global__ void
   __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
   MyKernel(...) {
       ...
   }

当 ``MyKernel`` 以每个块的最大线程数被调用时——即作为 ``__launch_bounds__()`` 的第一个参数指定——人们很容易想把 ``MY_KERNEL_MAX_THREADS`` 用作执行配置中每块的线程数：

.. code-block:: c++

   // Host code
   MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...);

然而，这行不通，因为如 :ref:`执行空间说明符 <execution-space-specifiers>` 部分所述， ``__CUDA_ARCH__`` 在主机代码中未定义。因此， ``MyKernel`` 将以每块 256 个线程启动。每块的线程数应改为通过以下方式确定：

- 在编译时使用不依赖 ``__CUDA_ARCH__`` 的宏或常量确定，例如：

  .. code-block:: c++

     // Host code
     MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);

- 或在运行时基于计算能力确定：

  .. code-block:: c++

     // Host code
     cudaGetDeviceProperties(&deviceProp, device);
     int threadsPerBlock = (deviceProp.major >= 9) ? 2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK;
     MyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);

``--resource-usage`` 编译器选项会报告寄存器使用量。`CUDA profiler <https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator>`__ 会报告占用率，可用于推导驻留块数量。

.. _maximum-number-of-registers-per-thread:

5.4.3.3. 每线程最大寄存器数
^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了启用低级性能调优，CUDA C++ 提供 ``__maxnreg__()`` 函数限定符，它将性能调优信息传递给后端优化编译器。 ``__maxnreg__()`` 限定符指定可以分配给线程块中单个线程的最大寄存器数。在 ``__global__`` 函数定义中：

.. code-block:: c++

   __global__ void
   __maxnreg__(maxNumberRegistersPerThread)
   MyKernel(...) {
       ...
   }

``maxNumberRegistersPerThread`` 变量指定要分配给内核 ``MyKernel()`` 的线程块中单个线程的最大寄存器数；它编译为 ``.maxnreg`` PTX 指令。

``__launch_bounds__()`` 和 ``__maxnreg__()`` 限定符不能一起应用于同一内核。

可以使用 ``--maxrregcount <N>`` 编译器选项控制文件中所有 ``__global__`` 函数的寄存器使用。对于具有 ``__maxnreg__`` 限定符的内核函数，此选项将被忽略。

.. _synchronization-primitives:

5.4.4. 同步原语
---------------

.. _thread-block-synchronization-functions:

5.4.4.1. 线程块同步函数
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __syncthreads();
   int  __syncthreads_count(int predicate);
   int  __syncthreads_and(int predicate);
   int  __syncthreads_or(int predicate);

这些内建函数协调同一块内线程之间的通信。当块中的线程访问共享或全局内存中的相同地址时，可能会发生读后写、写后读或写后写冲突。可以通过在这些访问之间同步线程来避免这些冲突。

这些内建函数具有以下语义：

- ``__syncthreads*()`` 等待线程块中所有未退出的线程同时到达程序中相同的 ``__syncthreads*()`` 内建调用或退出。

- ``__syncthreads*()`` 在参与线程之间提供内存排序：对 ``__syncthreads*()`` 内建的调用强烈发生在（参见 `C++ 规范 [intro.races] <https://eel.is/c++draft/intro.races>`__）任何参与线程从等待中解除阻塞或退出之前。

以下示例显示如何使用 ``__syncthreads()`` 同步线程块内的线程并安全地对线程之间共享的数组元素求和：

.. code-block:: c++

   #include <cuda_runtime_api.h>
   #include <memory.h>
   #include <cstdlib>
   #include <ctime>
   #include <stdio.h>

   // assuming blockDim.x is 128
   __global__ void example_syncthreads(int* input_data, int* output_data)
   {
       __shared__ int shared_data[128];
       shared_data[threadIdx.x] = input_data[blockDim.x*blockIdx.x + threadIdx.x];

       // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered
       // before any thread is unblocked from '__syncthreads()':
       __syncthreads();

       // A single thread safely reads 'shared_data':
       if (threadIdx.x == 0) {
           float sum = 0;
           for (int i = 0; i < blockDim.x; ++i) {
               sum += shared_data[i];
           }
           output_data[blockIdx.x] = sum;
       }
   }

   void initArray(int* A, int length)
   {
        std::srand(std::time({}));
       for(int i=0; i<length; i++)
       {
           A[i] = int(10.f * (rand() / (float)RAND_MAX));
       }
   }

   int main(int argc, char** arg)
   {
       constexpr int block_size = 128;
       constexpr int input_length = 1024;
       constexpr int output_length = input_length / block_size;

       // Pointers to memory vectors
       int* input = nullptr;
       int* output = nullptr;
       int* comparisonResult = (int*)malloc(output_length*sizeof(int));

       // Use unified memory to allocate buffers
       cudaMallocManaged(&input, input_length*sizeof(int));
       cudaMallocManaged(&output, output_length*sizeof(int));

       initArray(input, input_length);
       int grid_size = input_length / block_size;

       example_syncthreads<<<grid_size, block_size>>>(input, output);
       cudaDeviceSynchronize();

       for(int i=0; i<output_length; i++)
       {
           comparisonResult[i] = 0;
           for(int j=0; j < block_size; j++)
           {
               comparisonResult[i] += input[i*block_size + j];
           }
       }

       for(int i=0; i< output_length; i++)
       {
           if(output[i] != comparisonResult[i])
           {
               printf("Results do not match at index %d: %d != %d\n", i, output[i], comparisonResult[i]);
               exit(-1);
           }
       }
       printf("Test passed\n");

       return 0;
   }

``__syncthreads*()`` 内建函数允许在条件代码中使用，但仅当条件在整个线程块中统一求值时。否则，执行可能会挂起或产生意外的副作用。

以下示例演示了有效的行为：

.. code-block:: c++

   // assuming blockDim.x is 128
   __global__ void syncthreads_valid_behavior(int* input_data, int* output_data) {
       __shared__ int shared_data[128];
       shared_data[threadIdx.x] = input_data[threadIdx.x];
       if (blockIdx.x > 0) { // CORRECT, uniform condition across all block threads
           __syncthreads();
           output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
       }
   }

而以下示例表现出无效行为，例如内核挂起或未定义行为：

.. code-block:: c++

   // assuming blockDim.x is 128
   __global__ void syncthreads_invalid_behavior1(int* input_data, int* output_data) {
       __shared__ int shared_data[256];
       shared_data[threadIdx.x] = input_data[threadIdx.x];
       if (threadIdx.x > 0) { // WRONG, non-uniform condition
           __syncthreads();   // Undefined Behavior
           output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
       }
   }

.. code-block:: c++

   // assuming blockDim.x is 128
   __global__ void syncthreads_invalid_behavior2(int* input_data, int* output_data) {
       __shared__ int shared_data[256];
       shared_data[threadIdx.x] = input_data[threadIdx.x];
       for (int i = 0; i < blockDim.x; ++i) {
           if (i == threadIdx.x) { // WRONG, non-uniform condition
               __syncthreads();    // Undefined Behavior
           }
       }
       output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
   }

**带谓词的 ``__syncthreads()`` 变体：**

.. code-block:: c++

   int __syncthreads_count(int predicate);

与 ``__syncthreads()`` 相同，只是它为块中所有未退出的线程计算谓词，并返回谓词计算为非零值的线程数。

.. code-block:: c++

   int __syncthreads_and(int predicate);

与 ``__syncthreads()`` 相同，只是它为块中所有未退出的线程计算谓词。当且仅当谓词对所有线程都计算为非零值时，它返回非零值。

.. code-block:: c++

   int __syncthreads_or(int predicate);

与 ``__syncthreads()`` 相同，只是它为块中所有未退出的线程计算谓词。当且仅当谓词对一个或多个线程计算为非零值时，它返回非零值。

.. _warp-synchronization-function:

5.4.4.2. Warp 同步函数
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   void __syncwarp(unsigned mask = 0xFFFFFFFF);

内建函数 ``__syncwarp()`` 协调同一 warp 内线程之间的通信。当 warp 内的一些线程访问共享或全局内存中的相同地址时，可能会发生潜在的读后写、写后读或写后写冲突。可以通过在这些访问之间同步线程来避免这些数据冲突。

调用 ``__syncwarp(mask)`` 在 ``mask`` 中命名的 warp 内参与线程之间提供内存排序：对 ``__syncwarp(mask)`` 的调用强烈发生在（参见 `C++ 规范 [intro.races] <https://eel.is/c++draft/intro.races>`__） ``mask`` 中命名的任何 warp 线程从等待中解除阻塞或退出之前。

这些函数受 :ref:`Warp __sync 内建约束 <warp-sync-intrinsic-constraints>` 限制。

以下示例演示如何使用 ``__syncwarp()`` 同步 warp 内的线程以安全访问共享内存数组：

.. code-block:: c++

   __global__ void example_syncwarp(int* input_data, int* output_data) {
       if (threadIdx.x < warpSize) {
           __shared__ int shared_data[warpSize];
           shared_data[threadIdx.x] = input_data[threadIdx.x];

           __syncwarp(); // equivalent to __syncwarp(0xFFFFFFFF)
           if (threadIdx.x == 0)
               output_data[0] = shared_data[1];
       }
   }

.. _memory-fence-functions:

5.4.4.3. 内存栅栏函数
^^^^^^^^^^^^^^^^^^^^^

CUDA 编程模型采用弱排序内存模型。换句话说，CUDA 线程将数据写入共享内存、全局内存、页锁定主机内存或对等设备内存的顺序不一定是另一个 CUDA 或主机线程观察数据写入的顺序。在没有内存栅栏或同步的情况下从同一内存位置读取或写入会导致未定义行为。

在以下示例中，线程 1 执行 ``writeXY()`` ，线程 2 执行 ``readXY()`` 。

.. code-block:: c++

   __device__ int X = 1, Y = 2;

   __device__ void writeXY() {
       X = 10;
       Y = 20;
   }

   __device__ void readXY() {
       int B = Y;
       int A = X;
   }

两个线程同时读写相同的内存位置 ``X`` 和 ``Y`` 。任何数据竞争都会导致未定义行为并且没有定义的语义。因此，``A`` 和 ``B`` 的结果值可以是任何值。

内存栅栏和同步函数强制执行内存访问的 `顺序一致性排序 <https://en.cppreference.com/w/cpp/atomic/memory_order>`__ 。这些函数在强制执行排序的 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes>`__ 方面有所不同，但独立于访问的内存空间，包括共享内存、全局内存、页锁定主机内存和对等设备的内存。

.. hint::

   建议尽可能使用 `libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html>`__ 提供的 ``cuda::atomic_thread_fence`` 以确保安全和可移植性。

**块级内存栅栏**

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         // <cuda/atomic> header
         cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);

      确保：

      - 调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的所有写入，被调用线程所在块中的所有线程观察到发生在调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有写入之前；

      - 调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的所有读取，排序在调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有读取之前。

   .. tab-item:: 内建函数

      .. code-block:: c++

         void __threadfence_block();

      确保：

      - 调用线程在调用 ``__threadfence_block()`` 之前对所有内存的所有写入，被调用线程所在块中的所有线程观察到发生在调用线程在调用 ``__threadfence_block()`` 之后对所有内存的所有写入之前；

      - 调用线程在调用 ``__threadfence_block()`` 之前对所有内存的所有读取，排序在调用线程在调用 ``__threadfence_block()`` 之后对所有内存的所有读取之前。

**设备级内存栅栏**

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

      确保：

      - 调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有写入，不会被设备中的任何线程观察到发生在调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的任何写入之前。

   .. tab-item:: 内建函数

      .. code-block:: c++

         void __threadfence();

      确保：

      - 调用线程在调用 ``__threadfence()`` 之后对所有内存的所有写入，不会被设备中的任何线程观察到发生在调用线程在调用 ``__threadfence()`` 之前对所有内存的任何写入之前。

**系统级内存栅栏**

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);

      确保：

      - 调用线程在调用 ``cuda::atomic_thread_fence()`` 之前对所有内存的所有写入，被设备中的所有线程、主机线程和对等设备中的所有线程观察到发生在调用线程在调用 ``cuda::atomic_thread_fence()`` 之后对所有内存的所有写入之前。

   .. tab-item:: 内建函数

      .. code-block:: c++

         void __threadfence_system();

      确保：

      - 调用线程在调用 ``__threadfence_system()`` 之前对所有内存的所有写入，被设备中的所有线程、主机线程和对等设备中的所有线程观察到发生在调用线程在调用 ``__threadfence_system()`` 之后对所有内存的所有写入之前。

在上面的代码示例中，可以按如下方式在代码中插入内存栅栏：

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         #include <cuda/atomic>

         __device__ int X = 1, Y = 2;

         __device__ void writeXY() {
             X = 10;
             cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
             Y = 20;
         }

         __device__ void readXY() {
             int B = Y;
             cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
             int A = X;
         }

   .. tab-item:: 内建函数

      .. code-block:: c++

         __device__ int X = 1, Y = 2;

         __device__ void writeXY() {
             X = 10;
             __threadfence();
             Y = 20;
         }

         __device__ void readXY() {
             int B = Y;
             __threadfence();
             int A = X;
         }

对于此代码，可以观察到以下结果：

- ``A`` 等于 1 且 ``B`` 等于 2，即 ``readXY()`` 在 ``writeXY()`` 之前执行；

- ``A`` 等于 10 且 ``B`` 等于 20，即 ``writeXY()`` 在 ``readXY()`` 之前执行；

- ``A`` 等于 10 且 ``B`` 等于 2；

- ``A`` 为 1 且 ``B`` 为 20 的情况不可能出现，因为内存栅栏确保对 ``X`` 的写入在对 ``Y`` 的写入之前可见。

如果线程 1 和线程 2 属于同一个块，使用块级栅栏就足够了。如果线程 1 和线程 2 不属于同一个块，当它们是同一设备的 CUDA 线程时必须使用设备级栅栏，当它们来自两个不同设备的 CUDA 线程时必须使用系统级栅栏。

一个常见用例由以下代码示例说明，其中线程消费由其他线程生产的数据。此内核在单次调用中计算 N 个数字数组的和。

- 每个块首先对数组的一个子集求和，并将结果存储到全局内存。

- 当所有块都完成后，最后一个块从全局内存读取每个部分和并将它们相加以获得最终结果。

- 为确定哪个块最后完成，每个块原子地递增一个计数器，以表示其部分和的计算与存储已经完成（更多细节参见 :ref:`原子函数 <atomic-functions>` 部分）。最后一个块收到的计数器值等于 ``gridDim.x - 1`` 。

如果在存储部分和与递增计数器之间没有栅栏，计数器可能在部分和存储之前递增。这可能导致计数器达到 ``gridDim.x - 1`` ，并使最后一个块在部分和更新到内存之前就开始读取它们。

.. note::

   内存栅栏只影响内存操作的执行顺序；它不保证这些操作对其他线程的可见性。

在下面的代码示例中，通过将 ``result`` 变量声明为 ``volatile`` 来确保对其内存操作的可见性。更多细节请参见 :ref:`volatile 限定变量 <volatile-qualified-variables>` 部分。

.. code-block:: c++

   #include <cuda/atomic>

   __device__ int count = 0;

   __global__ void sum(const float*    array,
                       int             N,
                       volatile float* result) {
       __shared__ bool isLastBlockDone;
       // Each block sums a subset of the input array.
       float partialSum = calculatePartialSum(array, N);

       if (threadIdx.x == 0) {
           // Thread 0 of each block stores the partial sum to global memory.
           // The compiler will use a store operation that bypasses the L1 cache
           // since the "result" variable is declared as volatile.
           // This ensures that the threads of the last block will read the correct
           // partial sums computed by all other blocks.
           result[blockIdx.x] = partialSum;

           // Thread 0 makes sure that the increment of the "count" variable is
           // only performed after the partial sum has been written to global memory.
           cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

           // Thread 0 signals that it is done.
           int count_old = atomicInc(&count, gridDim.x);

           // Thread 0 determines if its block is the last block to be done.
           isLastBlockDone = (count_old == (gridDim.x - 1));
       }
       // Synchronize to make sure that each thread reads the correct value of
       // isLastBlockDone.
       __syncthreads();

       if (isLastBlockDone) {
           // The last block sums the partial sums stored in result[0 .. gridDim.x-1]
           float totalSum = calculateTotalSum(result);

           if (threadIdx.x == 0) {
               // Thread 0 of last block stores the total sum to global memory and
               // resets the count variable, so that the next kernel call works
               // properly.
               result[0] = totalSum;
               count     = 0;
           }
       }
   }

.. _atomic-functions:

5.4.5. 原子函数
---------------

原子函数对共享数据执行读-修改-写操作，使其看起来像是在单个步骤中执行。原子性确保每个操作要么完全完成，要么根本不发生，为所有参与线程提供数据的一致视图。

CUDA 以五种方式提供原子函数：

扩展 CUDA C++ 原子函数，`cuda::atomic <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html>`__ 和 `cuda::atomic_ref <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html>`__ 。

- 允许在主机和设备代码中使用。

- 遵循 `C++ 标准原子操作 <https://en.cppreference.com/w/cpp/atomic/atomic.html>`__ 语义。

- 允许指定原子操作的 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes>`__。

标准 C++ 原子函数，`cuda::std::atomic <https://en.cppreference.com/w/cpp/atomic/atomic.html>`__ 和 `cuda::std::atomic_ref <https://en.cppreference.com/w/cpp/atomic/atomic_ref.html>`__。

- 允许在主机和设备代码中使用。

- 遵循 `C++ 标准原子操作 <https://en.cppreference.com/w/cpp/atomic/atomic.html>`__ 语义。

- 不允许指定原子操作的 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes>`__。

编译器 :ref:`内置原子函数 <built-in-atomic-functions>` ， ``__nv_atomic_<op>()`` 。

- 自 CUDA 12.8 可用。

- 仅允许在设备代码中使用。

- 遵循 `C++ 标准原子内存序 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__ 语义。

- 允许指定原子操作的 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes>`__。

- 与 `C++ 标准原子操作 <https://en.cppreference.com/w/cpp/atomic/atomic.html>`__ 具有相同的内存排序语义。

- 支持 `cuda::std::atomic <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html>`__ 和 `cuda::std::atomic_ref <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html>`__ 允许的数据类型子集，但不支持 128 位数据类型。

- 在 tile 代码中不受支持。

CUDA Tile C++ 原子函数（例如 ``cuda::tiles::atomic_load`` ）：

- 仅允许在 tile 代码中使用。

- 遵循 `C++ 标准原子内存序 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__ 语义。

- 允许通过 ``cuda::tiles::thread_scope`` 指定线程作用域。

:ref:`遗留原子函数 <legacy-atomic-functions>` ， ``atomic<Op>()`` 。

- 仅允许在设备代码中使用。

- 仅支持 ``memory_order_relaxed`` `C++ 原子内存语义 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__。

- 允许在函数名中指定原子操作的 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes>`__。

- 与 :ref:`内置原子函数 <built-in-atomic-functions>` 不同，遗留原子函数仅确保原子性，不引入同步点（栅栏）。

- 支持 :ref:`内置原子函数 <built-in-atomic-functions>` 允许的数据类型子集。原子 ``add`` 操作支持额外的数据类型。

.. hint::

   建议使用 ``libcu++`` 提供的 `扩展 CUDA C++ 原子函数 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html>`__ 以提高效率、安全性和可移植性。

.. _legacy-atomic-functions:

5.4.5.1. 遗留原子函数
^^^^^^^^^^^^^^^^^^^^^

遗留原子函数对存储在全局或共享内存中的 32、64 或 128 位字执行原子读-修改-写操作。例如， ``atomicAdd()`` 函数读取全局或共享内存中特定地址的字，向其添加一个数字，并将结果写回同一地址。

- 原子函数只能在设备函数中使用。

- 对于向量类型（如 ``__half2`` 、 ``__nv_bfloat162`` 、 ``float2`` 和 ``float4`` ），读-修改-写操作对向量的每个元素执行。不保证整个向量在单个访问中是原子的。

本节描述的原子函数具有 `内存排序 <https://en.cppreference.com/w/cpp/atomic/memory_order>`__ ``cuda::std::memory_order_relaxed`` ，并且仅在特定的 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes>`__ 内是原子的：

- 无后缀的原子 API，例如 ``atomicAdd`` ，在作用域 ``cuda::thread_scope_device`` 内是原子的。

- 带 ``_block`` 后缀的原子 API，例如 ``atomicAdd_block`` ，在作用域 ``cuda::thread_scope_block`` 内是原子的。

- 带 ``_system`` 后缀的原子 API，例如 ``atomicAdd_system`` ，如果满足特定 `条件 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity>`__ ，则在作用域 ``cuda::thread_scope_system`` 内是原子的。

以下示例显示 CPU 和 GPU 原子更新地址 ``addr`` 处的整数值：

.. code-block:: c++

   #include <cuda_runtime.h>

   __global__ void atomicAdd_kernel(int* addr) {
       atomicAdd_system(addr, 10);
   }

   void test_atomicAdd(int device_id) {
       int* addr;
       cudaMallocManaged(&addr, 4);
       *addr = 0;

       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device_id);
       if (deviceProp.concurrentManagedAccess != 1) {
           return; // the device does not coherently access managed memory concurrently with the CPU
       }

       atomicAdd_kernel<<<...>>>(addr);
       __sync_fetch_and_add(addr, 10);  // CPU atomic operation
   }

---

请注意，任何原子操作都可以基于 ``atomicCAS()`` （比较并交换）实现。例如，单精度浮点数的 ``atomicAdd()`` 可以如下实现：

.. code-block:: c++

   #include <cuda/memory>
   #include <cuda/std/bit>

   __device__ float customAtomicAdd(float* d_ptr, float value) {
       volatile unsigned* d_ptr_unsigned = reinterpret_cast<unsigned*>(d_ptr);
       unsigned  old_value      = *d_ptr_unsigned;
       unsigned  assumed;
       do {
           assumed                          = old_value;
           float    assumed_float           = cuda::std::bit_cast<float>(assumed);
           float    expected_value          = assumed_float + value;
           unsigned expected_value_unsigned = cuda::std::bit_cast<unsigned>(expected_value);
           old_value                        = atomicCAS(d_ptr_unsigned, assumed, expected_value_unsigned);
       // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
       } while (assumed != old_value);
       return cuda::std::bit_cast<float>(old_value);
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/676e5bc7a>`__ 上的示例。

.. _atomicadd:

5.4.5.1.1. ``atomicAdd()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicAdd(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old + val`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicAdd()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``float`` 、 ``double`` 、 ``__half2`` 、 ``__half`` 。

- 计算能力 8.x 及更高版本设备上的 ``__nv_bfloat16`` 、 ``__nv_bfloat162`` 。

- 计算能力 9.x 及更高版本设备上的 ``float2`` 、 ``float4`` ，且仅支持全局内存地址。

对向量类型（例如 ``__half2`` 或 ``float4`` ）应用 ``atomicAdd()`` 的原子性对每个组件分别保证；不保证整个向量作为单个访问是原子的。

.. _atomicsub:

5.4.5.1.2. ``atomicSub()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicSub(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old - val`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicSub()`` 支持以下数据类型：

- ``int`` 、 ``unsigned``

.. _atomicinc:

5.4.5.1.3. ``atomicInc()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   unsigned atomicInc(unsigned* address, unsigned val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old >= val ? 0 : (old + 1)`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

.. _atomicdec:

5.4.5.1.4. ``atomicDec()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   unsigned atomicDec(unsigned* address, unsigned val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``(old == 0 || old > val) ? val : (old - 1)`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

.. _atomicand:

5.4.5.1.5. ``atomicAnd()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicAnd(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old & val`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicAnd()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 。

.. _atomicor:

5.4.5.1.6. ``atomicOr()``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicOr(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old | val`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicOr()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 。

.. _atomicxor:

5.4.5.1.7. ``atomicXor()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicXor(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old ^ val`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicXor()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 。

.. _atomicmin:

5.4.5.1.8. ``atomicMin()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicMin(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old`` 和 ``val`` 的最小值。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicMin()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``long long`` 。

.. _atomicmax:

5.4.5.1.9. ``atomicMax()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicMax(T* address, T val);

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old`` 和 ``val`` 的最大值。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicMax()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``long long`` 。

.. _atomicexch:

5.4.5.1.10. ``atomicExch()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicExch(T* address, T val);

.. code-block:: c++

   template<typename T>
   T atomicExch(T* address, T val); // only 128-bit types, compute capability 9.x and higher

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 将 ``val`` 存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicExch()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``float`` 。

C++ 模板函数 ``atomicExch()`` 支持 128 位类型，具有以下要求：

- 计算能力 9.x 及更高版本。

- ``T`` 必须对齐到 16 字节，即 ``alignof(T) >= 16`` 。

- ``T`` 必须是平凡可复制的，即 ``std::is_trivially_copyable_v<T>`` 。

- 对于 C++03 及更早版本： ``T`` 必须是平凡可构造的，即 ``std::is_default_constructible_v<T>`` 。

.. _atomiccas:

5.4.5.1.11. ``atomicCAS()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   T atomicCAS(T* address, T compare, T val);

.. code-block:: c++

   template<typename T>
   T atomicCAS(T* address, T compare, T val);  // only 128-bit types, compute capability 9.x and higher

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old == compare ? val : old`` 。

3. 将结果存储回同一地址的内存。

该函数返回 ``old`` 值。

``atomicCAS()`` 支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``unsigned short`` 。

C++ 模板函数 ``atomicCAS()`` 支持 128 位类型，具有以下要求：

- 计算能力 9.x 及更高版本。

- ``T`` 必须对齐到 16 字节，即 ``alignof(T) >= 16`` 。

- ``T`` 必须是平凡可复制的，即 ``std::is_trivially_copyable_v<T>`` 。

- 对于 C++03 及更早版本： ``T`` 必须是平凡可构造的，即 ``std::is_default_constructible_v<T>`` 。

.. _built-in-atomic-functions:

5.4.5.2. 内置原子函数
^^^^^^^^^^^^^^^^^^^^^

CUDA 12.8 及更高版本支持 CUDA 编译器原子操作的内置函数，遵循与 `C++ 标准原子操作 <https://en.cppreference.com/w/cpp/atomic/atomic.html>`__ 相同的内存排序语义和 CUDA `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes>`__。这些函数遵循 `GNU 的原子内置函数签名 <https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html>`__，并额外添加了一个线程作用域参数。

当支持内置原子函数时， ``nvcc`` 定义宏 ``__CUDACC_DEVICE_ATOMIC_BUILTINS__`` 。

以下是用于内置原子函数的 ``order`` 和 ``scope`` 参数的原始枚举器，对应 `内存序 <https://en.cppreference.com/w/cpp/atomic/atomic.html>`__ 和 `线程作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes>`__：

.. code-block:: c++

   // atomic memory orders
   enum {
      __NV_ATOMIC_RELAXED,
      __NV_ATOMIC_CONSUME,
      __NV_ATOMIC_ACQUIRE,
      __NV_ATOMIC_RELEASE,
      __NV_ATOMIC_ACQ_REL,
      __NV_ATOMIC_SEQ_CST
   };

.. code-block:: c++

   // thread scopes
   enum {
      __NV_THREAD_SCOPE_THREAD,
      __NV_THREAD_SCOPE_BLOCK,
      __NV_THREAD_SCOPE_CLUSTER,
      __NV_THREAD_SCOPE_DEVICE,
      __NV_THREAD_SCOPE_SYSTEM
   };

- 内存序对应 `C++ 标准原子操作的内存序 <https://en.cppreference.com/w/cpp/atomic/memory_order>`__。

- 线程作用域遵循 ``cuda::thread_scope`` `定义 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes>`__。

- ``__NV_ATOMIC_CONSUME`` 内存序目前使用更强的 ``__NV_ATOMIC_ACQUIRE`` 内存序实现。

- ``__NV_THREAD_SCOPE_THREAD`` 线程作用域目前使用更宽的 ``__NV_THREAD_SCOPE_BLOCK`` 线程作用域实现。

示例：

.. code-block:: c++

   __device__ T __nv_atomic_load_n(T*  pointer,
                                   int memory_order,
                                   int thread_scope = __NV_THREAD_SCOPE_SYSTEM);

内置原子函数有以下限制：

- 只能在设备函数中使用。

- 不能操作本地内存。

- 不能获取这些函数的地址。

- ``order`` 和 ``scope`` 参数必须是整型字面值；不能是变量。

- 线程作用域 ``__NV_THREAD_SCOPE_CLUSTER`` 在架构 ``sm_90`` 及更高版本上支持。

不支持的情况示例：

.. code-block:: c++

   // Not permitted in a host function
   __host__ void bar() {
       unsigned u1 = 1, u2 = 2;
       __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
   }

   // Not permitted to be applied to local memory
   __device__ void foo() {
     unsigned a = 1, b;
     __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
   }

   // Not permitted as a template default argument.
   // The function address cannot be taken.
   template<void *F = __nv_atomic_load_n>
   class X {
       void *f = F; // The function address cannot be taken.
   };

   // Not permitted to be called in a constructor initialization list.
   class Y {
       int a;
     public:
       __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
   };

.. _nv-atomic-fetch-add-nv-atomic-add:

5.4.5.2.1. ``__nv_atomic_fetch_add()`` 、 ``__nv_atomic_add()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old + val`` 。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_add`` 返回 ``old`` 值。

- ``__nv_atomic_add`` 无返回值。

这些函数支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``float`` 、 ``double`` 。

.. _nv-atomic-fetch-sub-nv-atomic-sub:

5.4.5.2.2. ``__nv_atomic_fetch_sub()`` 、 ``__nv_atomic_sub()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old - val`` 。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_sub`` 返回 ``old`` 值。

- ``__nv_atomic_sub`` 无返回值。

这些函数支持以下数据类型：

- ``int`` 、 ``unsigned`` 、 ``unsigned long long`` 、 ``float`` 、 ``double`` 。

.. _nv-atomic-fetch-and-nv-atomic-and:

5.4.5.2.3. ``__nv_atomic_fetch_and()`` 、 ``__nv_atomic_and()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_and(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_and      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old & val`` 。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_and`` 返回 ``old`` 值。

- ``__nv_atomic_and`` 无返回值。

这些函数支持以下数据类型：

- 任何大小为 4 或 8 字节的整数类型。

.. _nv-atomic-fetch-or-nv-atomic-or:

5.4.5.2.4. ``__nv_atomic_fetch_or()`` 、 ``__nv_atomic_or()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_or(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_or      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old | val`` 。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_or`` 返回 ``old`` 值。

- ``__nv_atomic_or`` 无返回值。

这些函数支持以下数据类型：

- 任何大小为 4 或 8 字节的整数类型。

.. _nv-atomic-fetch-xor-nv-atomic-xor:

5.4.5.2.5. ``__nv_atomic_fetch_xor()`` 、 ``__nv_atomic_xor()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old ^ val`` 。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_xor`` 返回 ``old`` 值。

- ``__nv_atomic_xor`` 无返回值。

这些函数支持以下数据类型：

- 任何大小为 4 或 8 字节的整数类型。

.. _nv-atomic-fetch-min-nv-atomic-min:

5.4.5.2.6. ``__nv_atomic_fetch_min()`` 、 ``__nv_atomic_min()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old`` 和 ``val`` 的最小值。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_min`` 返回 ``old`` 值。

- ``__nv_atomic_min`` 无返回值。

这些函数支持以下数据类型：

- ``unsigned`` 、 ``int`` 、 ``unsigned long long`` 、 ``long long`` 。

.. _nv-atomic-fetch-max-nv-atomic-max:

5.4.5.2.7. ``__nv_atomic_fetch_max()`` 、 ``__nv_atomic_max()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_fetch_max(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_max      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 计算 ``old`` 和 ``val`` 的最大值。

3. 将结果存储回同一地址的内存。

- ``__nv_atomic_fetch_max`` 返回 ``old`` 值。

- ``__nv_atomic_max`` 无返回值。

这些函数支持以下数据类型：

- ``unsigned`` 、 ``int`` 、 ``unsigned long long`` 、 ``long long`` 。

.. _nv-atomic-exchange-nv-atomic-exchange-n:

5.4.5.2.8. ``__nv_atomic_exchange()`` 、 ``__nv_atomic_exchange_n()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. ``__nv_atomic_exchange_n`` 将 ``val`` 存储到 ``address`` 指向的位置。

   ``__nv_atomic_exchange`` 将 ``old`` 存储到 ``ret`` 指向的位置，并将 ``val`` 地址处的值存储到 ``address`` 指向的位置。

- ``__nv_atomic_exchange_n`` 返回 ``old`` 值。

- ``__nv_atomic_exchange`` 无返回值。

这些函数支持以下数据类型：

- 任何大小为 4、8 或 16 字节的数据类型。

- 16 字节数据类型在计算能力 9.x 及更高版本的设备上支持。

.. _nv-atomic-compare-exchange-nv-atomic-compare-exchange-n:

5.4.5.2.9. ``__nv_atomic_compare_exchange()`` 、 ``__nv_atomic_compare_exchange_n()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                                  int scope = __NV_THREAD_SCOPE_SYSTEM);

   __device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                                  int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. 将 ``old`` 与 ``expected`` 指向的值进行比较。

3. 如果它们相等，返回值为 ``true`` ，并将 ``desired`` 存储到 ``address`` 指向的位置。否则，返回 ``false`` ，并将 ``old`` 存储到 ``expected`` 指向的位置。

参数 ``weak`` 被忽略，它会在 ``success_order`` 和 ``failure_order`` 之间选择更强的内存序来执行比较交换操作。

这些函数支持以下数据类型：

- 任何大小为 2、4、8 或 16 字节的数据类型。

- 16 字节数据类型在计算能力 9.x 及更高版本的设备上支持。

.. _nv-atomic-load-nv-atomic-load-n:

5.4.5.2.10. ``__nv_atomic_load()`` 、 ``__nv_atomic_load_n()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. ``__nv_atomic_load`` 将 ``old`` 存储到 ``ret`` 指向的位置。

   ``__nv_atomic_load_n`` 返回 ``old`` 。

这些函数支持以下数据类型：

- 任何大小为 1、2、4、8 或 16 字节的数据类型。

``order`` 不能是 ``__NV_ATOMIC_RELEASE`` 或 ``__NV_ATOMIC_ACQ_REL`` 。

.. _nv-atomic-store-nv-atomic-store-n:

5.4.5.2.11. ``__nv_atomic_store()`` 、 ``__nv_atomic_store_n()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
   __device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 ``address`` 处的 ``old`` 值。

2. ``__nv_atomic_store`` 读取 ``val`` 指向位置的值并存储到 ``address`` 指向的位置。

   ``__nv_atomic_store_n`` 将 ``val`` 存储到 ``address`` 指向的位置。

``order`` 不能是 ``__NV_ATOMIC_CONSUME`` 、 ``__NV_ATOMIC_ACQUIRE`` 或 ``__NV_ATOMIC_ACQ_REL`` 。

.. _nv-atomic-thread-fence:

5.4.5.2.12. ``__nv_atomic_thread_fence()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

   __device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

此原子函数基于指定的内存序在此线程请求的内存访问之间建立排序。线程作用域参数指定可能观察到此操作排序效果的线程集合。

.. _warp-functions:

5.4.6. Warp 函数
----------------

以下章节描述了 warp 函数，这些函数允许 warp 内的线程相互通信并执行计算。

.. hint::

   建议尽可能使用 ``CUB`` `Warp 级「集合」原语 <https://nvidia.github.io/cccl/unstable/cub/api_docs/warp_wide.html#warp-wide-collective-primitives>`__ 来执行 warp 操作，以保证效率、安全性和可移植性。

.. _warp-active-mask:

5.4.6.1. Warp 活动掩码
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cuda

   unsigned __activemask();

该函数返回一个 32 位整数掩码，表示调用 warp 中所有当前活动的线程。如果 warp 中的第 N 个通道在调用 ``__activemask()`` 时处于活动状态，则第 N 位被设置为 1。返回掩码中的 0 位表示非活动线程。已退出程序的线程始终标记为非活动。

.. warning::

   ``__activemask()`` 不能用于确定哪些 warp 通道执行给定的分支。此函数旨在用于机会性的 warp 级编程，仅提供 warp 内活动线程的瞬时快照。

.. code-block:: cuda

   // Check whether at least one thread's predicate evaluates to true
   if (pred) {
       // Invalid: the value of 'at_least_one' is non-deterministic
       // and could vary between executions.
       at_least_one = __activemask() > 0;
   }

请注意，在 ``__activemask()`` 调用处汇聚的线程不能保证在后续指令处保持汇聚，除非这些指令是 warp 同步内联函数 (``__sync``)。

例如，编译器可能会重新排序指令，活动线程集可能不会被保留：

.. code-block:: cuda

   unsigned mask      = __activemask();              // Assume mask == 0xFFFFFFFF (all bits set, all threads active)
   int      predicate = threadIdx.x % 2 == 0;        // 1 for even threads, 0 for odd threads
   int      result    = __any_sync(mask, predicate); // Active threads might not be preserved

.. _warp-vote-functions:

5.4.6.2. Warp 投票函数
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cuda

   int      __all_sync   (unsigned mask, int predicate);
   int      __any_sync   (unsigned mask, int predicate);
   unsigned __ballot_sync(unsigned mask, int predicate);

Warp 投票函数使给定 warp 的线程能够执行归约和广播操作。这些函数从 warp 中每个未退出的线程获取整数 ``predicate`` 作为输入，并将这些值与零进行比较。然后将比较结果在 warp 的活动线程中以以下方式之一进行组合（归约），向每个参与线程广播单个返回值：

``__all_sync(unsigned mask, predicate)`` ：

对 ``mask`` 中所有未退出的线程评估 ``predicate`` ，如果所有线程的 ``predicate`` 都评估为非零值，则返回非零值。

``__any_sync(unsigned mask, predicate)`` ：

对 ``mask`` 中所有未退出的线程评估 ``predicate`` ，如果一个或多个线程的 ``predicate`` 评估为非零值，则返回非零值。

``__ballot_sync(unsigned mask, predicate)`` ：

对 ``mask`` 中所有未退出的线程评估 ``predicate`` ，返回一个整数，如果第 N 个线程的 ``predicate`` 评估为非零值且第 N 个线程处于活动状态，则第 N 位被设置为 1。否则，第 N 位为零。

这些函数受 :ref:`Warp __sync 内建约束 <warp-sync-intrinsic-constraints>` 限制。

.. warning::

   这些内联函数不提供任何内存排序。

.. _warp-match-functions:

5.4.6.3. Warp 匹配函数
^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   建议使用 `libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp/warp_match_all.html>`__ 的 ``cuda::device::warp_match_all()`` 函数作为 ``__match_all_sync`` 函数的通用且更安全的替代。

.. code-block:: cuda

   unsigned __match_any_sync(unsigned mask, T value);
   unsigned __match_all_sync(unsigned mask, T value, int *pred);

Warp 匹配函数在 warp 内的未退出线程之间执行变量的广播和比较操作。

``__match_any_sync``

返回 ``mask`` 中具有相同按位 ``value`` 的未退出线程的掩码。

``__match_all_sync``

如果 ``mask`` 中所有未退出的线程都具有相同的按位 ``value`` ，则返回 ``mask`` ；否则返回 0。如果 ``mask`` 中所有未退出的线程都具有相同的按位 ``value`` ，则 predicate ``pred`` 设置为 ``true`` ；否则 predicate 设置为 false。

``T`` 可以是 ``int`` 、 ``unsigned`` 、 ``long`` 、 ``unsigned long`` 、 ``long long`` 、 ``unsigned long long`` 、 ``float`` 或 ``double`` 。

这些函数受 :ref:`Warp __sync 内建约束 <warp-sync-intrinsic-constraints>` 限制。

.. warning::

   这些内联函数不提供任何内存排序。

.. _warp-reduce-functions:

5.4.6.4. Warp 归约函数
^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   建议尽可能使用 ``CUB`` `Warp 级「集合」原语 <https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html#_CPPv4I0_iEN3cub10WarpReduceE>`__ 来执行 Warp 归约，以保证效率、安全性和可移植性。

计算能力 8.x 或更高的设备支持。

.. code-block:: cuda

   T        __reduce_add_sync(unsigned mask, T value);
   T        __reduce_min_sync(unsigned mask, T value);
   T        __reduce_max_sync(unsigned mask, T value);

   unsigned __reduce_and_sync(unsigned mask, unsigned value);
   unsigned __reduce_or_sync (unsigned mask, unsigned value);
   unsigned __reduce_xor_sync(unsigned mask, unsigned value);

``__reduce_<op>_sync`` 内联函数在同步 ``mask`` 中指定的所有未退出线程后，对 ``value`` 中提供的数据执行归约操作。

``__reduce_add_sync`` 、 ``__reduce_min_sync`` 、 ``__reduce_max_sync``

对 ``mask`` 中命名的每个未退出线程在 ``value`` 中提供的值执行算术加法、最小值或最大值归约操作后返回结果。 ``T`` 可以是 ``unsigned`` 或 ``signed`` 整数。

``__reduce_and_sync`` 、 ``__reduce_or_sync`` 、 ``__reduce_xor_sync``

对 ``mask`` 中命名的每个未退出线程在 ``value`` 中提供的值执行按位 AND、OR 或 XOR 归约操作后返回结果。

这些函数受 :ref:`Warp __sync 内建约束 <warp-sync-intrinsic-constraints>` 限制。

.. warning::

   这些内联函数不提供任何内存排序。

.. _warp-shuffle-functions:

5.4.6.5. Warp 洗牌函数
^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   建议使用 `libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp/warp_shuffle.html#libcudacxx-extended-api-warp-warp-shuffle>`__ 的 ``cuda::device::warp_shuffle()`` 函数作为 ``__shfl_sync()`` 和 ``__shfl_<op>_sync()`` 内联函数的通用且更安全的替代。

.. code-block:: cuda

   T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
   T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
   T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
   T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);

Warp 洗牌函数在 warp 内的未退出线程之间交换值，无需使用共享内存。

``__shfl_sync()`` ：从索引通道直接复制。

内联函数返回由 ``srcLane`` 给出的 ID 的线程所持有的 ``value`` 的值。

- 如果 ``width`` 小于 ``warpSize`` ，则 warp 的每个子分区作为一个独立实体，起始逻辑通道 ID 为 0。

- 如果 ``srcLane`` 超出范围 ``[0, width - 1]`` ，结果对应于 ``srcLane % width`` 持有的值，该值在同一子分区内。

---

``__shfl_up_sync()`` ：从 ID 比调用者低的通道复制。

内联函数通过从调用者的通道 ID 减去 ``delta`` 来计算源通道 ID。返回结果通道 ID 所持有的 ``value`` 的值：实际上， ``value`` 在 warp 内向上移动 ``delta`` 个通道。

- 如果 ``width`` 小于 ``warpSize`` ，则 warp 的每个子分区作为一个独立实体，起始逻辑通道 ID 为 0。

- 源通道索引不会环绕 ``width`` 的值，因此较低的 ``delta`` 个通道将保持不变。

---

``__shfl_down_sync()`` ：从 ID 比调用者高的通道复制。

内联函数通过向调用者的通道 ID 加上 ``delta`` 来计算源通道 ID。返回结果通道 ID 所持有的 ``value`` 的值：这具有将 ``value`` 在 warp 内向下移动 ``delta`` 个通道的效果。

- 如果 ``width`` 小于 ``warpSize`` ，则 warp 的每个子分区作为一个独立实体，起始逻辑通道 ID 为 0。

- 与 ``__shfl_up_sync()`` 类似，源通道的 ID 号不会环绕 ``width`` 的值，因此较高的 ``delta`` 个通道将实际上保持不变。

---

``__shfl_xor_sync()`` ：基于自身通道 ID 的按位 XOR 从通道复制。

内联函数通过对调用者的通道 ID 和 ``laneMask`` 执行按位 XOR 来计算源通道 ID：返回结果通道 ID 所持有的 ``value`` 的值。此模式实现了蝶形寻址模式，用于树归约和广播。

- 如果 ``width`` 小于 ``warpSize`` ，则每组 ``width`` 个连续线程能够访问来自较早组的元素。但是，如果它们尝试访问来自后续线程组的元素，将返回其自己的 ``value`` 值。

---

``T`` 可以是：

- ``int`` 、 ``unsigned`` 、 ``long`` 、 ``unsigned long`` 、 ``long long`` 、 ``unsigned long long`` 、 ``float`` 或 ``double`` 。

- 包含 ``cuda_fp16.h`` 头文件的 ``__half`` 和 ``__half2`` 。

- 包含 ``cuda_bf16.h`` 头文件的 ``__nv_bfloat16`` 和 ``__nv_bfloat162`` 。

线程只能从积极参与内联函数的另一个线程读取数据。如果目标线程是非活动的，检索的值是未定义的。

``width`` 必须是 ``[1, warpSize]`` 范围内的 2 的幂，即 1、2、4、8、16 或 32。其他值将产生未定义的结果。

这些函数受 :ref:`Warp __sync 内建约束 <warp-sync-intrinsic-constraints>` 限制。

**有效的 warp 洗牌使用示例：**

.. code-block:: cuda

   int laneId = threadIdx.x % warpSize;
   int data   = ...

   // all warp threads get 'data' from lane 0
   int result1 = __shfl_sync(0xFFFFFFFF, data, 0);

   if (laneId < 4) {
       // lanes 0, 1, 2, 3 get 'data' from lane 1
       int result2 = __shfl_sync(0xb1111, data, 1);
   }

   // lanes [0 - 15] get 'data' from lane 0
   // lanes [16 - 31] get 'data' from lane 16
   int result3 = __shfl_sync(0xFFFFFFFF, value, warpSize / 2);

   // each lane gets 'data' from the lane two positions above
   // lanes 30, 31 get their original value
   int result4 = __shfl_down_sync(0xFFFFFFFF, data, 2);

**无效的 warp 洗牌使用示例：**

.. code-block:: cuda

   int laneId = threadIdx.x % warpSize;
   int value  = ...
    // undefined behavior: lane 0 does not participate in the call
   int result = (laneId > 0) ? __shfl_sync(0xFFFFFFFF, value, 0) : 0;

   if (laneId <= 4) {
       // undefined behavior: destination lanes 5, 6 are not active for lanes 3, 4
       result = __shfl_down_sync(0b11111, value, 2);
   }

   // undefined behavior: width is not a power of 2
   __shfl_sync(0xFFFFFFFF, value, 0, /*width=*/31);

.. warning::

   这些内联函数不暗示内存屏障。它们不保证任何内存排序。

**示例 1：在 warp 内广播单个值**

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         #include <cassert>
         #include <cuda/warp>

         __global__ void warp_broadcast_kernel(int input) {
             int laneId = threadIdx.x % 32;
             int value;
             if (laneId == 0) { // unused variable for all threads except lane 0
                 value = input;
             }
             value = cuda::device::warp_shuffle_idx(value, 0); // Synchronize all threads in warp, and get "value" from lane 0
             assert(value == input);
         }

         int main() {
             warp_broadcast_kernel<<<1, 32>>>(1234);
             cudaDeviceSynchronize();
             return 0;
         }

   .. tab-item:: 内建函数

      .. code-block:: c++

         #include <assert.h>

         __global__ void warp_broadcast_kernel(int input) {
             int laneId = threadIdx.x % 32;
             int value;
             if (laneId == 0) { // unused variable for all threads except lane 0
                 value = input;
             }
             value = __shfl_sync(0xFFFFFFFF, value, 0); // Synchronize all threads in warp, and get "value" from lane 0
             assert(value == input);
         }

         int main() {
             warp_broadcast_kernel<<<1, 32>>>(1234);
             cudaDeviceSynchronize();
             return 0;
         }

另请参阅 `Compiler Explorer <https://cuda.godbolt.org/z/E3E3Y5e4e>`__ 上的示例。

**示例 2：跨 8 线程子分区的包含式加法扫描**

.. hint::

   建议使用 `cub::WarpScan <https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpScan.html>`__ 函数以获得高效且通用的 warp 扫描功能。

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         #include <cstdio>
         #include <cub/cub.cuh>

         __global__ void scan_sub_partition_with_8_threads_kernel() {
             using WarpScan    = cub::WarpScan<int, 8>;
             using TempStorage = typename WarpScan::TempStorage;
             __shared__ TempStorage temp_storage;

             int laneId = threadIdx.x % 32;
             int value  = 31 - laneId; // starting value to accumulate
             int partial_sum;
             WarpScan(temp_storage).InclusiveSum(value, partial_sum);
             printf("Thread %d final value = %d\n", threadIdx.x, partial_sum);
         }

         int main() {
             scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
             cudaDeviceSynchronize();
             return 0;
         }

   .. tab-item:: 内建函数

      .. code-block:: c++

         #include <stdio.h>

         __global__ void scan_sub_partition_with_8_threads_kernel() {
             int laneId = threadIdx.x % 32;
             int value  = 31 - laneId; // starting value to accumulate
             // Loop to accumulate scan within my partition.
             // Scan requires log2(8) == 3 steps for 8 threads
             for (int delta = 1; delta <= 4; delta *= 2) {
                 int tmp         = __shfl_up_sync(0xFFFFFFFF, value, delta, /*width=*/8); // read from laneId - delta
                 int source_lane = laneId % 8 - delta;
                 if (source_lane >= 0) // lanes with 'source_lane < 0' have their value unchanged
                     value += tmp;
             }
             printf("Thread %d final value = %d\n", threadIdx.x, value);
         }

         int main() {
             scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
             cudaDeviceSynchronize();
             return 0;
         }

另请参阅 `Compiler Explorer <https://cuda.godbolt.org/z/Tohd38edc>`__ 上的示例。

**示例 3：跨 warp 的归约**

.. hint::

   建议使用 `cub::WarpReduce <https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html>`__ 函数以获得高效且通用的 warp 归约功能。

.. tab-set::

   .. tab-item:: CUDA C++

      .. code-block:: c++

         #include <cstdio>
         #include <cub/cub.cuh>
         #include <cuda/warp>

         __global__ void warp_reduce_kernel() {
             using WarpReduce  = cub::WarpReduce<int>;
             using TempStorage = typename WarpReduce::TempStorage;
             __shared__ TempStorage temp_storage;

             int laneId     = threadIdx.x % 32;
             int value      = 31 - laneId; // starting value to accumulate
             auto aggregate = WarpReduce(temp_storage).Sum(value);
             aggregate      = cuda::device::warp_shuffle_idx(aggregate, 0);
             printf("Thread %d final value = %d\n", threadIdx.x, aggregate);
         }

         int main() {
             warp_reduce_kernel<<<1, 32>>>();
             cudaDeviceSynchronize();
             return 0;
         }

   .. tab-item:: 内建函数

      .. code-block:: c++

         #include <stdio.h>

         __global__ void warp_reduce_kernel() {
             int laneId = threadIdx.x % 32;
             int value  = 31 - laneId; // starting value to accumulate
             // Use XOR mode to perform butterfly reduction
             // A full-warp reduction requires log2(32) == 5 steps
             for (int i = 1; i <= 16; i *= 2)
                 value += __shfl_xor_sync(0xFFFFFFFF, value, i);
             // "value" now contains the sum across all threads
             printf("Thread %d final value = %d\n", threadIdx.x, value);
         }

         int main() {
             warp_reduce_kernel<<<1, 32>>>();
             cudaDeviceSynchronize();
             return 0;
         }

另请参阅 `Compiler Explorer <https://cuda.godbolt.org/z/T94nfGMzG>`__ 上的示例。

.. _warp-sync-intrinsic-constraints:

5.4.6.6. Warp ``__sync`` 内联函数约束
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有 warp ``__sync`` 内联函数，如：

- ``__shfl_sync`` 、 ``__shfl_up_sync`` 、 ``__shfl_down_sync`` 、 ``__shfl_xor_sync``

- ``__match_any_sync`` 、 ``__match_all_sync``

- ``__reduce_add_sync`` 、 ``__reduce_min_sync`` 、 ``__reduce_max_sync`` 、 ``__reduce_and_sync`` 、 ``__reduce_or_sync`` 、 ``__reduce_xor_sync``

- ``__syncwarp``

使用 ``mask`` 参数指示哪些 warp 线程参与调用。此参数确保在硬件执行内联函数之前正确汇聚。

``mask`` 中的每个位对应于线程的通道 ID（ ``threadIdx.x % warpSize`` ）。内联函数等待直到 ``mask`` 中指定的所有未退出的 warp 线程到达调用。

以下约束必须满足才能正确执行：

- 每个调用线程必须在 ``mask`` 中设置其对应的位。

- 每个未调用线程必须在 ``mask`` 中将其对应的位设置为零。已退出的线程被忽略。

- ``mask`` 中指定的所有未退出线程必须使用相同的 ``mask`` 值执行内联函数。

- Warp 线程可以同时使用不同的 ``mask`` 值调用内联函数，只要掩码是不相交的。即使在分歧控制流中，此条件也是有效的。

如果以下情况，warp ``__sync`` 函数的行为无效，可能导致内核挂起或未定义行为：

- 调用线程未在 ``mask`` 中指定。

- ``mask`` 中指定的未退出线程未能最终退出或在同一程序点使用相同的 ``mask`` 值调用内联函数。

- 在条件代码中，所有条件必须在 ``mask`` 中指定的所有未退出线程之间评估相同。

.. note::

   当所有 warp 线程参与调用时，内联函数达到最佳效率，即当 ``mask`` 设置为 ``0xFFFFFFFF`` 时。

**有效的 warp 内联函数使用示例：**

.. code-block:: cuda

   __global__ void valid_examples() {
       if (threadIdx.x < 4) {        // threads 0, 1, 2, 3 are active
           __all_sync(0b1111, pred); // CORRECT, threads 0, 1, 2, 3 participate in the call
       }

       if (threadIdx.x == 0)
           return; // exit
       // CORRECT, all non-exited threads participate in the call
       __all_sync(0xFFFFFFFF, pred);
   }

**不相交 ``mask`` 示例：**

.. code-block:: cuda

   __global__ void example_syncwarp_with_mask(int* input_data, int* output_data) {
       if (threadIdx.x < warpSize) {
           __shared__ int shared_data[warpSize];
           shared_data[threadIdx.x] = input_data[threadIdx.x];

           unsigned mask = threadIdx.x < 16 ? 0xFFFF : 0xFFFF0000; // CORRECT
           __syncwarp(mask);
           if (threadIdx.x == 0 || threadIdx.x == 16)
               output_data[threadIdx.x] = shared_data[threadIdx.x + 1];
       }
   }

.. code-block:: cuda

   __global__ void example_syncwarp_with_mask_branches(int* input_data, int* output_data) {
       if (threadIdx.x < warpSize) {
           __shared__ int shared_data[warpSize];
           shared_data[threadIdx.x] = input_data[threadIdx.x];

           if (threadIdx.x < 16) {
               unsigned mask = 0xFFFF; // CORRECT
               __syncwarp(mask);
               output_data[threadIdx.x] = shared_data[15 - threadIdx.x];
           }
           else {
               unsigned mask = 0xFFFF0000; // CORRECT
               __syncwarp(mask);
               output_data[threadIdx.x] = shared_data[31 - threadIdx.x];
           }
       }
   }

**无效的 warp 内联函数使用示例：**

.. code-block:: cuda

   if (threadIdx.x < 4) {           // threads 0, 1, 2, 3 are active
       __all_sync(0b0000011, pred); // WRONG, threads 2, 3 are active but not set in mask
       __all_sync(0b1111111, pred); // WRONG, threads 4, 5, 6 are not active but set in mask
   }

   // WRONG, participating threads have a different and overlapping mask
   __all_sync(threadIdx.x == 0 ? 1 : 0xFFFFFFFF, pred);

.. _cuda-specific-macros:

5.4.7. CUDA 特定宏
------------------

.. _cuda-arch-macro:

5.4.7.1. ``__CUDA_ARCH__``
^^^^^^^^^^^^^^^^^^^^^^^^^^

宏 ``__CUDA_ARCH__`` 表示正在为其编译代码的 NVIDIA GPU 的虚拟架构。其值可能与设备的实际计算能力不同。此宏允许编写针对特定 GPU 架构优化的代码路径，这对于获得最佳性能或使用架构特定的功能和指令可能是必要的。该宏还可以用于区分主机和设备代码。

``__CUDA_ARCH__`` 仅在设备代码中定义，即在 ``__device__`` 、 ``__host__ __device__`` 和 ``__global__`` 函数中。宏的值与 ``nvcc`` 选项 ``compute_<version>`` 相关联，关系为 ``__CUDA_ARCH__ = <version> * 10`` 。

示例：

.. code-block:: bash

   nvcc --generate-code arch=compute_80,code=sm_90 prog.cu

将 ``__CUDA_ARCH__`` 定义为 ``800`` 。

---

``__CUDA_ARCH__`` 约束

**1.** 以下实体的类型签名不应依赖于 ``__CUDA_ARCH__`` 是否定义或其值。

- ``__global__`` 函数和函数模板。

- ``__device__`` 和 ``__constant__`` 变量。

- 纹理和表面。

示例：

.. code-block:: cuda

   #if !defined(__CUDA_ARCH__)
       typedef int my_type;
   #else
       typedef double my_type;
   #endif

   __device__ my_type my_var;           // ERROR: my_var's type depends on __CUDA_ARCH__

   __global__ void kernel(my_type in) { // ERROR: kernel's type depends on __CUDA_ARCH__
       ...
   }

**2.** ``__global__`` 函数模板的实例化不应依赖于 ``__CUDA_ARCH__`` 是否定义或其值。也就是说，相同的实例化（具有等价的模板参数）必须在*所有*设备程序和主机程序中都存在（无论这些实例化是否在运行时曾被启动）。

示例：

.. code-block:: cuda

   __device__ int result;

   template <typename T>
   __global__ void kernel(T in) {
       result = in;
   }

   __host__ __device__ void host_device_function(void) {
   #if !defined(__CUDA_ARCH__)
       kernel<<<1, 1>>>(1); // ERROR: "kernel<int>" instantiation only
                            //        when __CUDA_ARCH__ is undefined!
   #endif
   }

   int main(void) {
       host_device_function();
       cudaDeviceSynchronize();
       return 0;
   }

通过将计算本身从 ``__global__`` 函数移入由前者调用的 ``__device__`` 函数模板，可以避免此问题。在 ``__global__`` 函数内，随后可以使用 ``__CUDA_ARCH__`` 以不同参数有条件地实例化该 ``__device__`` 函数模板。

**3.** 在单独编译模式下，具有外部链接的函数或变量定义的存在或缺少不应依赖于 ``__CUDA_ARCH__`` 的定义或其值。

示例：

.. code-block:: cuda

   #if !defined(__CUDA_ARCH__)
       void host_function(void) {} // ERROR: The definition of host_function()
                                   //        is only present when __CUDA_ARCH__
                                   //        is undefined
   #endif

**4.** 在单独编译中，预处理器宏 ``__CUDA_ARCH__`` 不应在头文件中使用，以防止对象具有不同的行为。或者，所有对象必须为相同的虚拟架构编译。如果在头文件中定义了一个弱函数或模板函数且其行为依赖于 ``__CUDA_ARCH__`` ，那么当这些对象为不同的计算架构编译时，该函数在不同对象中的实例可能会发生冲突。

例如，如果头文件 ``a.h`` 包含：

.. code-block:: cuda

   template<typename T>
   __device__ T* get_ptr() {
   #if __CUDA_ARCH__ == 900
       return nullptr; /* no address */
   #else
       __shared__ T arr[256];
       return arr;
   #endif
   }

然后如果 ``a.cu`` 和 ``b.cu`` 都包含 ``a.h`` 并为同一类型实例化 ``get_ptr()`` ，且 ``b.cu`` 期望一个非 ``NULL`` 地址，并使用以下命令编译：

.. code-block:: bash

   nvcc -arch=compute_70 -dc a.cu
   nvcc -arch=compute_80 -dc b.cu
   nvcc -arch=sm_80 a.o b.o

链接时只会使用 ``get_ptr()`` 函数的一个版本，因此行为取决于选择了哪个版本。要避免此问题，要么 ``a.cu`` 和 ``b.cu`` 必须为相同的计算架构编译，要么不应在共享的头文件函数中使用 ``__CUDA_ARCH__`` 。

编译器不保证为上述 ``__CUDA_ARCH__`` 的不支持使用生成诊断。

.. _cuda-arch-specific-macro:

5.4.7.2. ``__CUDA_ARCH_SPECIFIC__`` 和 ``__CUDA_ARCH_FAMILY_SPECIFIC__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

宏 ``__CUDA_ARCH_SPECIFIC__`` 和 ``__CUDA_ARCH_FAMILY_SPECIFIC__`` 定义用于分别识别具有 :ref:`架构 <compute-capabilities-architecture-specific-features>` 特定功能和 :ref:`家族 <compute-capabilities-family-specific-features>` 特定功能的 GPU 设备。更多信息请参见 :ref:`特性集编译目标 <compute-capabilities-feature-set-compiler-targets>` 部分。

与 ``__CUDA_ARCH__`` 类似， ``__CUDA_ARCH_SPECIFIC__`` 和 ``__CUDA_ARCH_FAMILY_SPECIFIC__`` 仅在设备代码中定义，即在 ``__device__`` 、 ``__host__ __device__`` 和 ``__global__`` 函数中。这些宏与 ``nvcc`` 选项 ``compute_<version>a`` 和 ``compute_<version>f`` 相关联。

.. code-block:: bash

   nvcc --generate-code arch=compute_100a,code=sm_100a prog.cu

- ``__CUDA_ARCH__ == 1000`` 。

- ``__CUDA_ARCH_SPECIFIC__ == 1000`` 。

- ``__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`` 。

.. code-block:: bash

   nvcc --generate-code arch=compute_100f,code=sm_103f prog.cu

- ``__CUDA_ARCH__ == 1000`` 。

- ``__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`` 。

- ``__CUDA_ARCH_SPECIFIC__`` 未定义。

.. code-block:: bash

   nvcc -arch=sm_100 prog.cu

- ``__CUDA_ARCH__ == 1000`` 。

- ``__CUDA_ARCH_FAMILY_SPECIFIC__`` 未定义。

- ``__CUDA_ARCH_SPECIFIC__`` 未定义。

.. code-block:: bash

   nvcc -arch=sm_100a prog.cu
   # equivalent to:
   nvcc --generate-code arch=sm_100a,compute_100,compute_100a prog.cu

- ``__CUDA_ARCH__ == 1000`` 。

- ``__CUDA_ARCH_FAMILY_SPECIFIC__`` 未定义。

- ``__CUDA_ARCH_SPECIFIC__ == 1000`` 和 ``__CUDA_ARCH_SPECIFIC__`` 未定义两种情况均会生成。

.. _cuda-feature-testing-macros:

5.4.7.3. CUDA 特性测试宏
^^^^^^^^^^^^^^^^^^^^^^^^

``nvcc`` 提供以下预处理器宏用于特性测试。当 CUDA 前端编译器支持特定特性时，这些宏被定义。

- ``__CUDACC_DEVICE_ATOMIC_BUILTINS__`` ：支持设备原子编译器内建函数。

- ``__NVCC_DIAG_PRAGMA_SUPPORT__`` ：支持诊断控制 pragma。

- ``__CUDACC_EXTENDED_LAMBDA__`` ：支持扩展 lambda。由 ``--expt-extended-lambda`` 或 ``--extended-lambda`` 标志启用。

- ``__CUDACC_RELAXED_CONSTEXPR__`` ：支持宽松 constexpr 函数。由 ``--expt-relaxed-constexpr`` 标志启用。

.. _nv-pure-attribute:

5.4.7.4. ``__nv_pure__`` 属性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 C/C++ 中，纯函数对其参数没有副作用，可以访问全局变量，但不会修改它们。

CUDA 提供主机和设备函数支持的 ``__nv_pure__`` 属性。编译器将 ``__nv_pure__`` 转换为 ``pure`` GNU 属性或 Microsoft Visual Studio ``noalias`` 属性。

.. code-block:: cuda

   __device__ __nv_pure__
   int add(int a, int b) {
       return a + b;
   }

.. _cuda-specific-functions:

5.4.8. CUDA 特定函数
--------------------

.. _address-space-predicate-functions:

5.4.8.1. 地址空间谓词函数
^^^^^^^^^^^^^^^^^^^^^^^^^

地址空间谓词函数用于确定指针的地址空间。

.. hint::

   建议使用 `libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory/is_address_from.html>`__ 提供的 ``cuda::device::is_address_from()`` 和 ``cuda::device::is_object_from()`` 函数作为地址空间谓词内联函数的可移植且更安全的替代。

.. code-block:: cuda

   __device__ unsigned __isGlobal      (const void* ptr);
   __device__ unsigned __isShared      (const void* ptr);
   __device__ unsigned __isConstant    (const void* ptr);
   __device__ unsigned __isGridConstant(const void* ptr);
   __device__ unsigned __isLocal       (const void* ptr);

如果 ``ptr`` 包含指定地址空间中对象的通用地址，这些函数返回 ``1`` ，否则返回 ``0`` 。如果参数是 ``NULL`` 指针，其行为未指定。

- ``__isGlobal()`` ：全局内存空间。

- ``__isShared()`` ：共享内存空间。

- ``__isConstant()`` ：常量内存空间。

- ``__isGridConstant()`` ：使用 ``__grid_constant__`` 注释的内核参数。

- ``__isLocal()`` ：本地内存空间。

.. _address-space-conversion-functions:

5.4.8.2. 地址空间转换函数
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 指针（ ``T*`` ）可以访问对象，无论对象存储在哪里。例如， ``int*`` 可以访问 ``int`` 对象，无论它们驻留在全局还是共享内存中。

地址空间转换函数用于在通用地址和特定地址空间的地址之间转换。当编译器无法确定指针的地址空间时，这些函数很有用，例如在跨翻译单元或与 PTX 指令交互时。

.. code-block:: cuda

   __device__ size_t __cvta_generic_to_global  (const void* ptr); // PTX: cvta.to.global
   __device__ size_t __cvta_generic_to_shared  (const void* ptr); // PTX: cvta.to.shared
   __device__ size_t __cvta_generic_to_constant(const void* ptr); // PTX: cvta.to.const
   __device__ size_t __cvta_generic_to_local   (const void* ptr); // PTX: cvta.to.local

.. code-block:: cuda

   __device__ void* __cvta_global_to_generic  (size_t raw_ptr); // PTX: cvta.global
   __device__ void* __cvta_shared_to_generic  (size_t raw_ptr); // PTX: cvta.shared
   __device__ void* __cvta_constant_to_generic(size_t raw_ptr); // PTX: cvta.const
   __device__ void* __cvta_local_to_generic   (size_t raw_ptr); // PTX: cvta.local

作为与 PTX 指令交互操作的示例， ``ld.shared.s32 r0, [ptr];`` PTX 指令期望 ``ptr`` 引用共享内存地址空间。具有指向 ``__shared__`` 内存中对象的 ``int*`` 指针的 CUDA 程序需要在将此指针传递给 PTX 指令之前通过调用 ``__cvta_generic_to_shared`` 将此指针转换为共享地址空间，如下所示：

.. code-block:: cuda

   __shared__ int smem_var;
   smem_var        = 42;
   size_t smem_ptr = __cvta_generic_to_shared(&smem_var);
   int    output;
   asm volatile("ld.shared.s32 %0, [%1];" : "=r"(output) : "l"(smem_ptr) : "memory");
   assert(output == 42);

利用这些地址表示的一种常见优化是：共享、本地和常量空间的地址范围小于 32 位，因此可以存储 32 位地址而不是 64 位指针以缩小数据结构并节省寄存器。此外，32 位算术运算比 64 位算术运算更快。要获得这些地址的 32 位整数表示，请将 64 位值截断为 32 位，即从无符号 64 位整数强制转换为无符号 32 位整数：

.. code-block:: cuda

   __shared__ int smem_var;
   uint32_t       smem_ptr_32bit = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_var));

要从这样的 32 位表示恢复泛型地址，请将地址零扩展回无符号 64 位整数，然后调用相应的地址空间转换函数：

.. code-block:: cuda

   size_t smem_ptr_64bit = static_cast<size_t>(smem_ptr_32bit); // zero-extend to 64 bits
   void*  generic_ptr    = __cvta_shared_to_generic(smem_ptr_64bit);
   assert(generic_ptr == &smem_var);

.. _low-level-load-store-functions:

5.4.8.3. 低级加载和存储函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cuda

   T __ldg(const T* address);

函数 ``__ldg()`` 执行只读 L1/Tex 缓存加载。它支持所有 C++ 基本类型、CUDA 向量类型（除 x3 分量外）和扩展浮点类型，如 ``__half`` 、 ``__half2`` 、 ``__nv_bfloat16`` 和 ``__nv_bfloat162`` 。

---

.. code-block:: cuda

   T __ldcg(const T* address);
   T __ldca(const T* address);
   T __ldcs(const T* address);
   T __ldlu(const T* address);
   T __ldcv(const T* address);

这些函数使用 `PTX ISA 指南 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`__ 中指定的缓存操作符执行加载。它们支持所有 C++ 基本类型、CUDA 向量类型（除 x3 分量外）和扩展浮点类型，如 ``__half`` 、 ``__half2`` 、 ``__nv_bfloat16`` 和 ``__nv_bfloat162`` 。

---

.. code-block:: cuda

   void __stwb(T* address, T value);
   void __stcg(T* address, T value);
   void __stcs(T* address, T value);
   void __stwt(T* address, T value);

这些函数使用 `PTX ISA 指南 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`__ 中指定的缓存操作符执行存储。它们支持所有 C++ 基本类型、CUDA 向量类型（除 x3 分量外）和扩展浮点类型，如 ``__half`` 、 ``__half2`` 、 ``__nv_bfloat16`` 和 ``__nv_bfloat162`` 。

.. _trap-function:

5.4.8.4. ``__trap()``
^^^^^^^^^^^^^^^^^^^^^

.. hint::

   建议使用 `libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api.html>`__ 提供的 ``cuda::std::terminate()`` 函数（ `C++ 参考 <https://en.cppreference.com/w/cpp/error/terminate.html>`__ ）作为 ``__trap()`` 的可移植替代。

可以通过从任何设备线程调用 ``__trap()`` 函数来启动陷阱操作。

.. code-block:: cuda

   void __trap();

内核执行被中止，在主机程序中引发中断。调用 ``__trap()`` 会导致 CUDA 上下文损坏，使后续 CUDA 调用和内核调用失败。

.. _nanosleep-function:

5.4.8.5. ``__nanosleep()``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cuda

   __device__ void __nanosleep(unsigned nanoseconds);

函数 ``__nanosleep(ns)`` 将线程暂停大约 ``ns`` 纳秒的睡眠持续时间。最大睡眠持续时间大约为一毫秒。

示例：

以下代码实现了具有指数退避的互斥锁。

.. code-block:: cuda

   __device__ void mutex_lock(unsigned* mutex) {
       unsigned ns = 8;
       while (atomicCAS(mutex, 0, 1) == 1) {
           __nanosleep(ns);
           if (ns < 256) {
               ns *= 2;
           }
       }
   }

   __device__ void mutex_unlock(unsigned *mutex) {
       atomicExch(mutex, 0);
   }

.. _dpx-instructions:

5.4.8.6. 动态编程扩展 (DPX) 指令
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DPX 函数集支持查找最小值和最大值，以及最多对三个 16 位或 32 位有符号或无符号整数参数执行融合加法与最小/最大值运算。还具有可选的 ReLU（即钳位到零）特性。

比较函数：

- 三个参数。语义：``max(a, b, c)`` 、 ``min(a, b, c)`` 。

   .. code-block:: c++

      int      __vimax3_s32  (int,      int,      int);
      unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
      unsigned __vimax3_u32  (unsigned, unsigned, unsigned);
      unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);

      int      __vimin3_s32  (int,      int,      int);
      unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
      unsigned __vimin3_u32  (unsigned, unsigned, unsigned);
      unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);

- 两个参数，带 ReLU。语义：``max(a, b, 0)`` 、 ``max(min(a, b), 0)`` 。

   .. code-block:: c++

      int      __vimax_s32_relu  (int,      int);
      unsigned __vimax_s16x2_relu(unsigned, unsigned);

      int      __vimin_s32_relu  (int,      int);
      unsigned __vimin_s16x2_relu(unsigned, unsigned);

- 三个参数，带 ReLU。语义：``max(a, b, c, 0)`` 、 ``max(min(a, b, c), 0)`` 。

   .. code-block:: c++

      int      __vimax3_s32_relu  (int,      int,      int);
      unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);

      int      __vimin3_s32_relu  (int,      int,      int);
      unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);

- 两个参数，同时返回哪个参数更小/更大：

   .. code-block:: c++

      int      __vibmax_s32  (int,      int,      bool* pred);
      unsigned __vibmax_u32  (unsigned, unsigned, bool* pred);
      unsigned __vibmax_s16x2(unsigned, unsigned, bool* pred);
      unsigned __vibmax_u16x2(unsigned, unsigned, bool* pred);

      int      __vibmin_s32  (int,      int,      bool* pred);
      unsigned __vibmin_u32  (unsigned, unsigned, bool* pred);
      unsigned __vibmin_s16x2(unsigned, unsigned, bool* pred);
      unsigned __vibmin_u16x2(unsigned, unsigned, bool* pred);

融合加法与最小/最大值：

- 三个参数，将（第一个 + 第二个）与第三个比较。语义：``max(a + b, c)`` 、 ``min(a + b, c)`` 。

   .. code-block:: c++

      int      __viaddmax_s32  (int,      int,      int);
      unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
      unsigned __viaddmax_u32  (unsigned, unsigned, unsigned);
      unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);

      int      __viaddmin_s32  (int,      int,      int);
      unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
      unsigned __viaddmin_u32  (unsigned, unsigned, unsigned);
      unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);

- 三个参数，带 ReLU，将（第一个 + 第二个）与第三个以及零比较。语义：``max(a + b, c, 0)`` 、 ``max(min(a + b, c), 0)`` 。

   .. code-block:: c++

      int      __viaddmax_s32_relu  (int,      int,      int);
      unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);

      int      __viaddmin_s32_relu  (int,      int,      int);
      unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);

根据计算能力的不同，这些指令或由硬件加速，或由软件模拟。计算能力要求参见 `算术指令 <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions>`__ 部分。

完整 API 可在 `CUDA Math API 文档 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html>`__ 中找到。

---

对于实现动态编程算法，DPX 是一款极为有用的工具，例如基因组学中的 Smith-Waterman 和 Needleman-Wunsch 算法，以及路径优化中的 Floyd-Warshall 算法。

三个有符号 32 位整数的最大值，带 ReLU：

.. code-block:: c++

   int a           = -15;
   int b           = 8;
   int c           = 5;
   int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8
   int d           = -2;
   int e           = -4;
   int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0

两个 32 位有符号整数之和、另一个 32 位有符号整数与零（ReLU）的最小值：

.. code-block:: c++

   int a           = -5;
   int b           = 6;
   int c           = -2;
   int max_value_0 = __viaddmax_s32_relu(a, b, c); // max(-5 + 6, -2, 0) = max(1, -2, 0) = 1
   int d           = 4;
   int max_value_1 = __viaddmax_s32_relu(a, d, c); // max(-5 + 4, -2, 0) = max(-1, -2, 0) = 0

两个无符号 32 位整数的最小值，并确定哪个值更小：

.. code-block:: c++

   unsigned a = 9;
   unsigned b = 6;
   bool     smaller_value;
   unsigned min_value = __vibmin_u32(a, b, &smaller_value); // min_value is 6, smaller_value is true

三对无符号 16 位整数的最大值：

.. code-block:: c++

   unsigned a         = 0x00050002;
   unsigned b         = 0x00070004;
   unsigned c         = 0x00020006;
   unsigned max_value = __vimax3_u16x2(a, b, c); // max(5, 7, 2) and max(2, 4, 6), so max_value is 0x00070006

.. _compiler-optimization-hints:

5.4.9. 编译器优化提示
----------------------

编译器优化提示通过为代码添加额外信息来帮助编译器优化生成的代码。

- 内置函数在设备代码中始终可用。
- 主机代码的支持取决于主机编译器。

.. _pragma-unroll:

5.4.9.1. ``#pragma unroll``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

编译器默认会展开具有已知迭代次数的小循环。然而， ``#pragma unroll`` 指令可用于控制任何给定循环的展开。该指令必须紧挨着放在循环之前，且仅适用于该循环。

可选地，可以跟随一个整数常量表达式。以下是关于整数常量表达式的各种情况：

- 如果未提供，当循环的迭代次数为常数时，循环将被完全展开。
- 如果其值为 ``0`` 或 ``1`` ，循环将不会被展开。
- 如果其为非正整数或大于 ``INT_MAX`` ，该 pragma 将被忽略，并发出警告。

示例：

.. code-block:: cuda

   struct MyStruct {
       static constexpr int value = 4;
   };

   inline constexpr int Count = 4;

   __device__ void foo(int* p1, int* p2) {
       // no argument specified, the loop will be completely unrolled
       #pragma unroll
       for (int i = 0; i < 12; ++i)
           p1[i] += p2[i] * 2;

       // unroll value = 5
       #pragma unroll (Count + 1)
       for (int i = 0; i < 12; ++i)
           p1[i] += p2[i] * 4;

       // unroll value = 1, loop unrolling disabled
       #pragma unroll 1
       for (int i = 0; i < 12; ++i)
           p1[i] += p2[i] * 8;

       // unroll value = 4
       #pragma unroll (MyStruct::value)
       for (int i = 0; i < 12; ++i)
           p1[i] += p2[i] * 16;

       // negative value, pragma unroll ignored
       #pragma unroll -1
       for (int i = 0; i < 12; ++i)
           p1[i] += p2[i] * 2;
   }

另请参阅 `Compiler Explorer <https://godbolt.org/z/fPMK55PxE>`__ 上的示例。

.. _builtin-assume-aligned:

5.4.9.2. ``__builtin_assume_aligned()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   建议使用 `libcu++ <https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api.html>`__ 提供的 ``cuda::std::assume_aligned()`` 函数（ `C++ 参考 <https://en.cppreference.com/w/cpp/memory/assume_aligned.html>`__ ）作为内置函数的可移植且更安全的替代方案。

.. code-block:: cpp

   void* __builtin_assume_aligned(const void* ptr, size_t align)
   void* __builtin_assume_aligned(const void* ptr, size_t align, <integral type> offset)

内置函数使编译器能够假定返回的指针至少按 ``align`` 字节对齐。

- 三参数版本使编译器能够假定 ``(char*) ptr - offset`` 至少按 ``align`` 字节对齐。

``align`` 必须是 2 的幂且为整数字面量。

示例：

.. code-block:: cpp

   void* res1 = __builtin_assume_aligned(ptr, 32);    // compiler can assume 'res1' is at least 32-byte aligned
   void* res2 = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume 'res2 = (char*) ptr - 8' is at least 32-byte aligned

.. _builtin-assume:

5.4.9.3. ``__builtin_assume()`` 和 ``__assume()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   void __builtin_assume(bool predicate)
   void __assume        (bool predicate) // only with Microsoft Compiler

内置函数使编译器能够假定布尔参数为真。如果参数在运行时为假，行为是未定义的。请注意，如果参数有副作用，行为是未指定的。

示例：

.. code-block:: cuda

   __device__ bool is_greater_than_zero(int value) {
       return value > 0;
   }

   __device__ bool f(int value) {
       __builtin_assume(value > 0);
       return is_greater_than_zero(value); // returns true, without evaluating the condition
   }

.. _builtin-expect:

5.4.9.4. ``__builtin_expect()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   long __builtin_expect(long input, long expected)

内置函数告知编译器 ``input`` 预期等于 ``expected`` ，并返回 ``input`` 的值。它通常用于向编译器提供分支预测信息。其行为类似于 C++20 的 ``[[likely]]`` 和 ``[[unlikely]]`` 属性。

示例：

.. code-block:: cpp

   // indicate to the compiler that likely "var == 0"
   if (__builtin_expect(var, 0))
       doit();

.. _builtin-unreachable:

5.4.9.5. ``__builtin_unreachable()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   void __builtin_unreachable(void)

内置函数告知编译器控制流永远不会到达调用该函数的位置。如果控制流在运行时确实到达了该位置，程序具有未定义行为。

此函数对于避免生成不可达分支的代码以及禁用编译器对不可达代码的警告很有用。

示例：

.. code-block:: cpp

   // indicates to the compiler that the default case label is never reached.
   switch (in) {
       case 1:  return 4;
       case 2:  return 10;
       default: __builtin_unreachable();
   }

.. _custom-abi-pragmas:

5.4.9.6. 自定义 ABI Pragma
^^^^^^^^^^^^^^^^^^^^^^^^^^

``#pragma nv_abi`` 指令使在单独编译模式下编译的应用程序能够实现与全程序编译相似的性能，通过保持函数使用的寄存器数量。

使用此 pragma 的语法如下，其中 ``EXPR`` 指代任何整数常量表达式：

.. code-block:: cpp

   #pragma nv_abi preserve_n_data(EXPR) preserve_n_control(EXPR)

- ``#pragma nv_abi`` 后的参数是可选的，可以按任何顺序提供；然而，至少需要一个参数。
- ``preserve_n`` 参数限制了函数调用期间保留的寄存器数量：
  
  - ``preserve_n_data(EXPR)`` 限制了数据寄存器的数量。
  
  - ``preserve_n_control(EXPR)`` 限制了控制寄存器的数量。

``#pragma nv_abi`` 指令可以紧挨着放在设备函数声明或定义之前。

.. code-block:: cuda

   #pragma nv_abi preserve_n_data(16)
   __device__ void dev_func();

   #pragma nv_abi preserve_n_data(16) preserve_n_control(8)
   __device__ int dev_func() {
       return 0;
   }

或者，它可以直接放在设备函数内 C++ 表达式语句中的间接函数调用之前。请注意，虽然支持对自由函数的间接调用，但不支持对函数引用或类成员函数的间接调用。

.. code-block:: cuda

   __device__ int dev_func1();

   struct MyStruct {
       __device__ int member_func2();
   };

   __device__ void test() {
       auto* dev_func_ptr = &dev_func1; // type: int (*)(void)
       #pragma nv_abi preserve_n_control(8)
       int v1 = dev_func_ptr();         // CORRECT, indirect call

       #pragma nv_abi preserve_n_control(8)
       int v2 = dev_func1();            // WRONG, direct call; the pragma has no effect
                                        // dev_func1 has type: int(void)

       auto& dev_func_ref = &dev_func1; // type: int (&)(void)
       #pragma nv_abi preserve_n_control(8)
       int v3 = dev_func_ref();         // WRONG, call to a reference
                                        // the pragma has no effect

       auto member_function_ptr = &MyStruct::member_func2; // type: int (MyStruct::*)(void)
       #pragma nv_abi preserve_n_control(8)
       int v4 = member_function_ptr();  // WRONG, indirect call to member function
                                        // the pragma has no effect
   }

当应用于设备函数的声明或定义时，该 pragma 会修改对该函数的任何调用的自定义 ABI 属性。当放置在间接函数调用位置时，它仅影响该特定调用的 ABI 属性。请注意，pragma 仅在放置在调用位置时影响间接函数调用；它对直接函数调用没有影响。

.. code-block:: cuda

   #pragma nv_abi preserve_n_control(8)
   __device__ int dev_func3();

   __device__ int dev_func4();

   __device__ void test() {
       int v1 = dev_func3();            // CORRECT, the pragma affects the direct call

       auto* dev_func_ptr = &dev_func4; // type: int (*)(void)
       #pragma nv_abi preserve_n_control(8)
       int v2 = dev_func_ptr();         // CORRECT, the pragma affects the indirect call

       int v3 = dev_func_ptr();         // WRONG, the pragma has no effect
   }

请注意，如果函数声明及其对应定义的 pragma 参数不匹配，程序是格式错误的。

.. _mma-throughput-pragma:

5.4.9.7. MMA 吞吐量 Pragma
^^^^^^^^^^^^^^^^^^^^^^^^^^

``nv_mma_throughput`` pragma 是一个指令，用于启用专门针对矩阵乘加运算调优的编译器优化。它仅在入口函数作用域（即 ``__global__`` 函数上）有效。其效果可能通过沿调用路径传播的优化，扩展到从该入口函数调用的 ``__device__`` 函数。

示例：

.. code-block:: cuda

   #pragma nv_mma_throughput
   __global__  void kernel(){
   ...
   }

.. note::

   ``nv_mma_throughput`` pragma 仍处于实验阶段。它启用的设置是在有限的 NVIDIA 内部工作负载上调优的。不保证每个内核都能提升性能。

.. _debugging-and-diagnostics:

5.4.10. 调试和诊断
------------------

.. _assertion:

5.4.10.1. 断言
^^^^^^^^^^^^^^

.. code-block:: cpp

   #define assert(expression) /* unspecified */

``assert()`` 宏在 ``expression`` 等于零时停止内核执行。如果程序在调试器中运行，会触发断点，允许使用调试器检查设备的当前状态。否则，对于每个 ``expression`` 等于零的线程，在通过 ``cudaDeviceSynchronize()`` 、 ``cudaStreamSynchronize()`` 或 ``cudaEventSynchronize()`` 与主机同步后，会向 stderr 打印一条消息。此消息的格式如下：

.. code-block:: text

   <filename>:<line number>:<function>:
   block: [blockIdx.x,blockIdx.y,blockIdx.z],
   thread: [threadIdx.x,threadIdx.y,threadIdx.z]
   Assertion `<expression>` failed.

内核执行被中止，并在主机程序中引发中断。 ``assert()`` 宏会导致 CUDA 上下文损坏，使任何后续 CUDA 调用或内核调用失败，返回 ``cudaErrorAssert`` 。

如果 ``expression`` 不为零，内核执行不受影响。

例如，来自源文件 ``test.cu`` 的以下程序：

.. code-block:: cuda

   #include <assert.h>

   __global__ void testAssert(void) {
       int is_one        = 1;
       int should_be_one = 0;

       // This will have no effect
       assert(is_one);

       // This will halt kernel execution
       assert(should_be_one);
   }

   int main(void) {
       testAssert<<<1,1>>>();
       cudaDeviceSynchronize();
       return 0;
   }

将输出：

.. code-block:: text

   test.cu:11: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.

断言用于调试目的。由于它们可能影响性能，建议在生产代码中禁用它们。可以在编译时通过在包含 ``assert.h`` 或 ``<cassert>`` 之前定义 ``NDEBUG`` 预处理器宏，或使用编译器标志 ``-DNDEBUG`` 来禁用它们。请注意，表达式不应有副作用；否则，禁用断言会影响代码的功能。

``assert()`` 宏在 ``__device__`` 和 ``__tile__`` 代码中均可用。

.. _breakpoint-function:

5.4.10.2. 断点函数
^^^^^^^^^^^^^^^^^^

可以通过从任何设备线程调用 ``__brkpt()`` 函数来暂停内核函数的执行。

.. code-block:: cpp

   void __brkpt();

.. _diagnostic-pragmas:

5.4.10.3. 诊断 Pragma
^^^^^^^^^^^^^^^^^^^^^

以下 pragma 可用于管理触发特定诊断消息时的错误严重程度。

.. code-block:: cpp

   #pragma nv_diag_suppress
   #pragma nv_diag_warning
   #pragma nv_diag_error
   #pragma nv_diag_default
   #pragma nv_diag_once

这些 pragma 的用法如下：

.. code-block:: cpp

   #pragma nv_diag_xxx <error_number1>, <error_number2> ...

受影响的诊断通过警告消息中显示的错误号指定。任何诊断都可以改为错误，但只有警告在被改为错误后才能被抑制或恢复其严重程度。 ``nv_diag_default`` pragma 将诊断的严重程度恢复到发出任何其他 pragma 之前有效的严重程度，即消息的正常严重程度（可能已被任何命令行选项修改）。以下示例抑制了 ``foo()`` 的「已声明但从未引用」警告：

.. code-block:: cuda

   #pragma nv_diag_suppress 177 // "declared but never referenced"
   void foo() {
       int i = 0;
   }

   #pragma nv_diag_default 177
   void bar() {
       int i = 0;
   }

以下 pragma 可用于保存和恢复当前诊断 pragma 状态：

.. code-block:: cpp

   #pragma nv_diagnostic push
   #pragma nv_diagnostic pop

示例：

.. code-block:: cuda

   #pragma nv_diagnostic push
   #pragma nv_diag_suppress 177 // "declared but never referenced"
   void foo() {
       int i = 0;
   }

   #pragma nv_diagnostic pop
   void bar() {
       int i = 0; // raise a warning
   }

请注意，这些指令仅影响 ``nvcc`` CUDA 前端编译器。它们对主机编译器没有影响。

当支持诊断 pragma 时， ``nvcc`` 定义宏 ``__NVCC_DIAG_PRAGMA_SUPPORT__`` 。

.. _warp-matrix-functions:

5.4.11. Warp 矩阵函数
---------------------

C++ warp 矩阵操作利用 Tensor Core 来加速形式为 ``D=A*B+C`` 的矩阵问题。这些操作在计算能力 7.0 或更高的设备上支持混合精度浮点数据。这需要 warp 中所有线程的协作。此外，这些操作仅允许在条件代码中使用，前提是条件在整个 warp 中评估一致，否则代码执行很可能会挂起。

.. _wmma-description:

5.4.11.1. 描述
^^^^^^^^^^^^^^

以下所有函数和类型都定义在命名空间 ``nvcuda::wmma`` 中。子字节操作被视为预览功能，即它们的数据结构和 API 可能会发生变化，可能不兼容未来的版本。这些额外功能定义在 ``nvcuda::wmma::experimental`` 命名空间中。

.. code-block:: cpp

   template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

   void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
   void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
   void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
   void fill_fragment(fragment<...> &a, const T& v);
   void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);

**fragment**

一个重载类，包含分布在 warp 中所有线程上的矩阵片段。矩阵元素到 ``fragment`` 内部存储的映射是未指定的，可能在未来的架构中发生变化。

只允许特定的模板参数组合。第一个模板参数指定片段如何参与矩阵操作。 ``Use`` 的可接受值为：

- ``matrix_a`` ：当片段用作第一个乘数 ``A`` 时，
- ``matrix_b`` ：当片段用作第二个乘数 ``B`` 时，或
- ``accumulator`` ：当片段用作源或目标累加器（分别为 ``C`` 或 ``D`` ）时。

``m`` 、 ``n`` 和 ``k`` 大小描述参与乘加操作的 warp 级矩阵块的形状。每个块的维度取决于其角色。对于 ``matrix_a`` ，块的维度为 ``m x k`` ；对于 ``matrix_b`` ，维度为 ``k x n`` ； ``accumulator`` 块为 ``m x n`` 。

数据类型 ``T`` 对于乘数可以是 ``double`` 、 ``float`` 、 ``__half`` 、 ``__nv_bfloat16`` 、 ``char`` 或 ``unsigned char`` ，对于累加器可以是 ``double`` 、 ``float`` 、 ``int`` 或 ``__half`` 。如 :ref:`元素类型和矩阵大小 <wmma-element-types-matrix-sizes>` 部分所述，只支持有限的累加器和乘数类型组合。对于 ``matrix_a`` 和 ``matrix_b`` 片段必须指定 Layout 参数。 ``row_major`` 或 ``col_major`` 分别表示矩阵行或列中的元素在内存中是连续的。 ``accumulator`` 矩阵的 ``Layout`` 参数应保留默认值 ``void`` 。行或列布局仅在如下所述加载或存储累加器时指定。

**load_matrix_sync**

等待所有 warp 线程到达 load_matrix_sync，然后从内存加载矩阵片段 a。 ``mptr`` 必须是 256 位对齐的指针，指向内存中矩阵的第一个元素。 ``ldm`` 描述连续行（对于行主序布局）或连续列（对于列主序布局）之间的元素步长，对于 ``__half`` 元素类型必须是 8 的倍数，对于 ``float`` 元素类型必须是 4 的倍数（两种情况下都是 16 字节的倍数）。如果片段是 ``accumulator`` ，必须将 ``layout`` 参数指定为 ``mem_row_major`` 或 ``mem_col_major`` 。对于 ``matrix_a`` 和 ``matrix_b`` 片段，布局从片段的 ``layout`` 参数推断。 ``mptr`` 、 ``ldm`` 、 ``layout`` 和 ``a`` 的所有模板参数的值对于 warp 中的所有线程必须相同。此函数必须由 warp 中的所有线程调用，否则结果未定义。

**store_matrix_sync**

等待所有 warp 线程到达 store_matrix_sync，然后将矩阵片段 a 存储到内存。 ``mptr`` 必须是 256 位对齐的指针，指向内存中矩阵的第一个元素。 ``ldm`` 描述连续行（对于行主序布局）或连续列（对于列主序布局）之间的元素步长，对于 ``__half`` 元素类型必须是 8 的倍数，对于 ``float`` 元素类型必须是 4 的倍数（两种情况下都是 16 字节的倍数）。输出矩阵的布局必须指定为 ``mem_row_major`` 或 ``mem_col_major`` 。 ``mptr`` 、 ``ldm`` 、 ``layout`` 和 a 的所有模板参数的值对于 warp 中的所有线程必须相同。

**fill_fragment**

用常量值 ``v`` 填充矩阵片段。由于矩阵元素到每个片段的映射是未指定的，此函数通常由 warp 中的所有线程调用，并使用 ``v`` 的共同值。

**mma_sync**

等待所有 warp 线程到达 mma_sync，然后执行 warp 同步矩阵乘加操作 ``D=A*B+C`` 。也支持原地操作 ``C=A*B+C`` 。 ``satf`` 的值和每个矩阵片段的模板参数对于 warp 中的所有线程必须相同。此外，片段 ``A`` 、 ``B`` 、 ``C`` 和 ``D`` 之间的模板参数 ``m`` 、 ``n`` 和 ``k`` 必须匹配。此函数必须由 warp 中的所有线程调用，否则结果未定义。

如果 ``satf`` （饱和到有限值）模式为 ``true`` ，以下额外的数值属性适用于目标累加器：

- 如果元素结果为 +Infinity，相应的累加器将包含 ``+MAX_NORM``
- 如果元素结果为 -Infinity，相应的累加器将包含 ``-MAX_NORM``
- 如果元素结果为 NaN，相应的累加器将包含 ``+0``

由于矩阵元素到每个线程的 ``fragment`` 的映射是未指定的，在调用 ``store_matrix_sync`` 后，必须从内存（共享或全局）访问单个矩阵元素。在 warp 中所有线程统一对所有片段元素应用逐元素操作的特殊情况下，可以使用以下 ``fragment`` 类成员实现直接元素访问。

.. code-block:: cpp

   enum fragment<Use, m, n, k, T, Layout>::num_elements;
   T fragment<Use, m, n, k, T, Layout>::x[num_elements];

作为示例，以下代码将 ``accumulator`` 矩阵块缩放一半。

.. code-block:: cuda

   wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
   float alpha = 0.5f; // Same value for all threads in warp
   /*...*/
   for(int t=0; t<frag.num_elements; t++)
       frag.x[t] *= alpha;

.. _wmma-alternate-floating-point:

5.4.11.2. 备选浮点
^^^^^^^^^^^^^^^^^^

Tensor Core 在计算能力 8.0 及更高的设备上支持备选类型的浮点运算。

``__nv_bfloat16``

此数据格式是一种备选 fp16 格式，具有与 f32 相同的范围，但精度降低（7 位）。你可以直接使用 ``cuda_bf16.h`` 中提供的 ``__nv_bfloat16`` 类型来使用此数据格式。具有 ``__nv_bfloat16`` 数据类型的矩阵片段需要与 ``float`` 类型的累加器组合使用。支持的形状和操作与 ``__half`` 相同。

``tf32``

此数据格式是 Tensor Core 支持的一种特殊浮点格式，具有与 f32 相同的范围和降低的精度（≥10 位）。此格式的内部布局是实现定义的。要将此浮点格式与 WMMA 操作一起使用，输入矩阵必须手动转换为 tf32 精度。

为便于转换，提供了新的内建函数 ``__float_to_tf32``。虽然该内建函数的输入和输出参数都是 ``float`` 类型，但输出在数值上将是 ``tf32``。此新精度仅用于 Tensor Core，如果与其他 ``float`` 类型操作混合使用，结果的精度和范围将是未定义的。

一旦输入矩阵（``matrix_a`` 或 ``matrix_b``）被转换为 tf32 精度，具有 ``precision::tf32`` 精度和 ``float`` 数据类型的 ``fragment`` 与 ``load_matrix_sync`` 的组合将利用这一新功能。两个累加器片段都必须具有 ``float`` 数据类型。唯一支持的矩阵大小是 16x16x8（m-n-k）。

片段的元素表示为 ``float``，因此从 ``element_type<T>`` 到 ``storage_element_type<T>`` 的映射为：

.. code-block:: text

   precision::tf32 -> float

.. _wmma-double-precision:

5.4.11.3. 双精度
^^^^^^^^^^^^^^^^

Tensor Core 在计算能力 8.0 及更高的设备上支持双精度浮点运算。要使用此新功能，必须使用 ``double`` 类型的 ``fragment``。``mma_sync`` 操作将使用 .rn（舍入到最接近的偶数）舍入修饰符执行。

.. _wmma-sub-byte-operations:

5.4.11.4. 子字节操作
^^^^^^^^^^^^^^^^^^^^

子字节 WMMA 操作提供了访问 Tensor Core 低精度能力的方式。它们被视为预览功能，即其数据结构和 API 可能会发生变化，可能不兼容未来版本。此功能通过 ``nvcuda::wmma::experimental`` 命名空间提供：

.. code-block:: cpp

   namespace experimental {
       namespace precision {
           struct u4; // 4-bit unsigned
           struct s4; // 4-bit signed
           struct b1; // 1-bit
      }
       enum bmmaBitOp {
           bmmaBitOpXOR = 1, // compute_75 minimum
           bmmaBitOpAND = 2  // compute_80 minimum
       };
       enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
   }

对于 4 位精度，可用的 API 保持不变，但你必须指定 ``experimental::precision::u4`` 或 ``experimental::precision::s4`` 作为片段数据类型。由于片段的元素被打包在一起，该片段的 ``num_storage_elements`` 将小于 ``num_elements``。因此，子字节片段的 ``num_elements`` 变量返回子字节类型 ``element_type<T>`` 的元素数量。对于单位精度也是如此，在这种情况下，从 ``element_type<T>`` 到 ``storage_element_type<T>`` 的映射如下：

.. code-block:: text

   experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
   experimental::precision::s4 -> int (8 elements in 1 storage element)
   experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
   T -> T  //all other types

子字节片段允许的布局对于 ``matrix_a`` 始终为 ``row_major``，对于 ``matrix_b`` 始终为 ``col_major``。

对于子字节操作，``load_matrix_sync`` 中 ``ldm`` 的值对于元素类型 ``experimental::precision::u4`` 和 ``experimental::precision::s4`` 应为 32 的倍数，或对于元素类型 ``experimental::precision::b1`` 应为 128 的倍数（即两种情况下都是 16 字节的倍数）。

.. note::

   对以下 MMA 指令变体的支持已弃用，并将在 sm_90 中移除：

   - ``experimental::precision::u4``

   - ``experimental::precision::s4``

   - ``bmmaBitOp`` 设置为 ``bmmaBitOpXOR`` 的 ``experimental::precision::b1``

``bmma_sync``

等待所有 warp lane 执行 bmma_sync，然后执行 warp 同步的位矩阵乘加操作 ``D = (A op B) + C``，其中 ``op`` 由逻辑操作 ``bmmaBitOp`` 后跟 ``bmmaAccumulateOp`` 定义的累加组成。可用的操作有：

``bmmaBitOpXOR``，``matrix_a`` 中的一行与 ``matrix_b`` 的 128 位列进行 128 位 XOR

``bmmaBitOpAND``，``matrix_a`` 中的一行与 ``matrix_b`` 的 128 位列进行 128 位 AND，在计算能力 8.0 及更高的设备上可用。

累加操作始终为 ``bmmaAccumulateOpPOPC``，它计算置位位的数量。

.. _wmma-restrictions:

5.4.11.5. 限制
^^^^^^^^^^^^^^

Tensor Core 所需的特殊格式对于每个主要和次要设备架构可能不同。线程仅持有整体矩阵的一个片段（不透明的架构特定 ABI 数据结构），且开发者不允许对各参数如何映射到参与矩阵乘加的寄存器做任何假设，这使情况更加复杂。

由于片段是架构特定的，如果函数是为不同的链接兼容架构编译并链接到同一设备可执行文件中的，则将它们从函数 A 传递到函数 B 是不安全的。在这种情况下，片段的大小和布局将特定于某一架构，在另一架构中使用 WMMA API 将导致错误结果或潜在的数据损坏。

两个链接兼容但片段布局不同的架构示例是 sm_70 和 sm_75。

.. code-block:: text

   fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
   fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }

.. code-block:: text

   // sm_70 fragment layout
   $> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
   // sm_75 fragment layout
   $> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
   // Linking the two together
   $> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o

这种未定义行为在编译时和运行时工具中也可能无法检测，因此需要格外小心以确保片段布局一致。当与既为不同链接兼容架构构建又期望被传递 WMMA 片段的遗留库链接时，最可能出现此链接风险。

请注意，在弱链接的情况下（例如 CUDA C++ 内联函数），链接器可能选择任何可用的函数定义，这可能导致编译单元之间的隐式传递。

为避免此类问题，应始终将矩阵存储到内存中以便通过外部接口传输（例如 ``wmma::store_matrix_sync(dst, …);``），然后可以作为指针类型（例如 ``float *dst``）安全地传递给 ``bar()``。

请注意，由于 sm_70 可以在 sm_75 上运行，上述示例的 sm_75 代码可以改为 sm_70 并在 sm_75 上正确工作。但是，当与其他 sm_75 分离编译的二进制文件链接时，建议在应用程序中使用 sm_75 原生代码。

.. _wmma-element-types-matrix-sizes:

5.4.11.6. 元素类型和矩阵大小
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tensor Core 支持多种元素类型和矩阵大小。下表展示了支持的 ``matrix_a``、``matrix_b`` 和 ``accumulator`` 矩阵的各种组合：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Matrix A
     - Matrix B
     - 累加器
     - 矩阵大小 (m-n-k)
   * - ``__half``
     - ``__half``
     - ``float``
     - 16x16x16
   * - ``__half``
     - ``__half``
     - ``float``
     - 32x8x16
   * - ``__half``
     - ``__half``
     - ``float``
     - 8x32x16
   * - ``__half``
     - ``__half``
     - ``__half``
     - 16x16x16
   * - ``__half``
     - ``__half``
     - ``__half``
     - 32x8x16
   * - ``__half``
     - ``__half``
     - ``__half``
     - 8x32x16
   * - ``unsigned char``
     - ``unsigned char``
     - ``int``
     - 16x16x16
   * - ``unsigned char``
     - ``unsigned char``
     - ``int``
     - 32x8x16
   * - ``unsigned char``
     - ``unsigned char``
     - ``int``
     - 8x32x16
   * - ``signed char``
     - ``signed char``
     - ``int``
     - 16x16x16
   * - ``signed char``
     - ``signed char``
     - ``int``
     - 32x8x16
   * - ``signed char``
     - ``signed char``
     - ``int``
     - 8x32x16

备选浮点支持：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Matrix A
     - Matrix B
     - 累加器
     - 矩阵大小 (m-n-k)
   * - ``__nv_bfloat16``
     - ``__nv_bfloat16``
     - ``float``
     - 16x16x16
   * - ``__nv_bfloat16``
     - ``__nv_bfloat16``
     - ``float``
     - 32x8x16
   * - ``__nv_bfloat16``
     - ``__nv_bfloat16``
     - ``float``
     - 8x32x16
   * - ``precision::tf32``
     - ``precision::tf32``
     - ``float``
     - 16x16x8

双精度支持：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Matrix A
     - Matrix B
     - 累加器
     - 矩阵大小 (m-n-k)
   * - ``double``
     - ``double``
     - ``double``
     - 8x8x4

子字节操作的实验性支持：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Matrix A
     - Matrix B
     - 累加器
     - 矩阵大小 (m-n-k)
   * - ``precision::u4``
     - ``precision::u4``
     - ``int``
     - 8x8x32
   * - ``precision::s4``
     - ``precision::s4``
     - ``int``
     - 8x8x32
   * - ``precision::b1``
     - ``precision::b1``
     - ``int``
     - 8x8x128

.. _wmma-example:

5.4.11.7. 示例
^^^^^^^^^^^^^^

以下代码在单个 warp 中实现了 16x16x16 矩阵乘法。

.. code-block:: cuda

   #include <mma.h>
   using namespace nvcuda;

   __global__ void wmma_ker(half *a, half *b, float *c) {
      // Declare the fragments
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

      // Initialize the output to zero
      wmma::fill_fragment(c_frag, 0.0f);

      // Load the inputs
      wmma::load_matrix_sync(a_frag, a, 16);
      wmma::load_matrix_sync(b_frag, b, 16);

      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

      // Store the output
      wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
   }
