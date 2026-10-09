.. _graphics-interop-details:

4.21. CUDA 与图形 API 互操作
============================

在 CUDA 中直接访问来自 API 的 GPU 数据，允许使用 CUDA 核函数读写这些数据，从而在从其他 API 使用它们的同时提供 CUDA 功能。主要有两个概念：直接方法，即与 OpenGL 和 Direct3D[9-11] 的 :ref:`图形互操作性 <graphics-interoperability>`，它使得可以将资源从 OpenGL 和 Direct3D 映射到 CUDA 地址空间；以及更灵活的 :ref:`外部资源互操作性 <external-resource-interoperability>`，其中可以通过导入和导出操作系统级句柄来访问内存和同步对象。这支持以下 API：Direct3D[11-12]、Vulkan 和 NVIDIA 软件通信接口互操作性。

.. _graphics-interoperability:

4.21.1. 图形互操作性
--------------------

在使用 CUDA 访问 Direct3D 或 OpenGL 资源（例如 VBO（顶点缓冲对象））之前，必须先注册和映射它。使用相应的 CUDA 函数（见下面的示例）进行注册将返回类型为 ``struct cudaGraphicsResource`` 的 CUDA 图形资源，该资源保存 CUDA 设备指针或数组。要访问核函数中的设备数据，必须映射资源。资源注册后，可以根据需要多次映射和取消映射。映射的资源通过使用 ``cudaGraphicsResourceGetMappedPointer()`` 返回的设备内存地址（用于缓冲区）和 ``cudaGraphicsSubResourceGetMappedArray()`` （用于 CUDA 数组）由核函数访问。一旦 CUDA 不再需要该资源，可以取消注册。这些是主要步骤：

1. 使用 CUDA 注册图形缓冲区
2. 映射资源
3. 访问映射资源的设备指针或数组
4. 在 CUDA 核函数中使用设备指针或数组
5. 取消映射资源
6. 取消注册资源

请注意，注册资源代价高昂，因此理想情况下每个资源只调用一次，但是对于打算使用该资源的每个 CUDA 上下文，需要分别注册资源。可以调用 ``cudaGraphicsResourceSetMapFlags()`` 来指定使用提示（只写、只读），CUDA 驱动程序可以使用这些提示来优化资源管理。另外请注意，在资源映射时通过 OpenGL、Direct3D 或不同的 CUDA 上下文访问资源会产生未定义的结果。

.. _opengl-interoperability:

4.21.1.1. OpenGL 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^

可以映射到 CUDA 地址空间的 OpenGL 资源是 OpenGL 缓冲区、纹理和渲染缓冲区对象。使用 ``cudaGraphicsGLRegisterBuffer()`` 注册缓冲区对象，在 CUDA 中，它显示为普通设备指针。使用 ``cudaGraphicsGLRegisterImage()`` 注册纹理或渲染缓冲区对象，在 CUDA 中，它显示为 CUDA 数组。

如果使用 ``cudaGraphicsRegisterFlagsSurfaceLoadStore`` 标志注册了纹理或渲染缓冲区对象，则可以写入它。 ``cudaGraphicsGLRegisterImage()`` 支持具有 1、2 或 4 个分量且内部类型为 float（例如 ``GL_RGBA_FLOAT32`` ）、归一化整数（例如 ``GL_RGBA8, GL_INTENSITY16`` ）和非归一化整数（例如 ``GL_RGBA8UI`` ）的所有纹理格式。

**示例：simpleGL 互操作性**

以下代码示例使用核函数动态修改存储在顶点缓冲对象 (VBO) 中的 2D ``width`` x ``height`` 顶点网格，并经过以下主要步骤：

1. 使用 CUDA 注册 VBO
2. 循环：从 CUDA 映射 VBO 进行写入
3. 循环：运行 CUDA 核函数修改顶点位置
4. 循环：取消映射 VBO
5. 循环：使用 OpenGL 渲染结果
6. 取消注册并删除 VBO

此部分的完整示例 simpleGL 可以在这里找到：`NVIDIA/cuda-samples <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleGL>`__

.. code-block:: cuda

   __global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
   {
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

       // calculate uv coordinates
       float u = x / (float)width;
       float v = y / (float)height;
       u = u * 2.0f - 1.0f;
       v = v * 2.0f - 1.0f;

       // calculate simple sine wave pattern
       float freq = 4.0f;
       float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

       // write output vertex
       pos[y * width + x] = make_float4(u, w, v, 1.0f);
   }

.. code-block:: c++

   int main(int argc, char **argv)
   {
       char *ref_file = NULL;

       pArgc = &argc;
       pArgv = argv;

   #if defined(__linux__)
       setenv("DISPLAY", ":0", 0);
   #endif

       printf("%s starting...\n", sSDKsample);

       if (argc > 1) {
           if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
               // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
               getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
           }
       }

       printf("\n");

       // First initialize OpenGL context
       if (false == initGL(&argc, argv)) {
           return false;
       }

       // register callbacks
       glutDisplayFunc(display);
       glutKeyboardFunc(keyboard);
       glutMouseFunc(mouse);
       glutMotionFunc(motion);
       glutCloseFunc(cleanup);

       // Create an empty vertex buffer object (VBO)
       // 1. Register the VBO with CUDA
       createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

       // start rendering mainloop
       //  5. Render the results using OpenGL
       glutMainLoop();

       printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
       exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);

   }

   void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
   {
       assert(vbo);

       // create buffer object
       glGenBuffers(1, vbo);
       glBindBuffer(GL_ARRAY_BUFFER, *vbo);

       // initialize buffer object
       unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
       glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

       glBindBuffer(GL_ARRAY_BUFFER, 0);

       // register this buffer object with CUDA
       checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

       SDK_CHECK_ERROR_GL();
   }

   void display()
   {
       float4 *dptr;
       // 2. Map the VBO for writing from CUDA
       checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
       size_t num_bytes;
       checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource));

       // 3. Run CUDA kernel to modify the vertex positions
       //call the CUDA kernel
       dim3 block(8, 8, 1);
       dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
       simple_vbo_kernel<<<grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

       //  4. Unmap the VBO
       checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       // set view matrix
       glMatrixMode(GL_MODELVIEW);
       glLoadIdentity();
       glTranslatef(0.0, 0.0, translate_z);
       glRotatef(rotate_x, 1.0, 0.0, 0.0);
       glRotatef(rotate_y, 0.0, 1.0, 0.0);

       // 5. Render the updated  using OpenGL
       glBindBuffer(GL_ARRAY_BUFFER, vbo);
       glVertexPointer(4, GL_FLOAT, 0, 0);

       glEnableClientState(GL_VERTEX_ARRAY);
       glColor3f(1.0, 0.0, 0.0);
       glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
       glDisableClientState(GL_VERTEX_ARRAY);

       glutSwapBuffers();

       g_fAnim += 0.01f;

   }

   void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
   {
       // 6. Unregister and delete VBO
       checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

       glBindBuffer(1, *vbo);
       glDeleteBuffers(1, vbo);

       *vbo = 0;
   }

   void cleanup()
   {

       if (vbo) {
           deleteVBO(&vbo, cuda_vbo_resource);
       }
   }

**限制和注意事项。**

- 正在共享其资源的 OpenGL 上下文必须对进行任何 OpenGL 互操作性 API 调用的主机线程当前有效。
- 当 OpenGL 纹理设置为无绑定（例如，通过使用 ``glGetTextureHandle`` 或 ``glGetImageHandle`` API 请求图像或纹理句柄）时，它不能与 CUDA 注册。应用程序需要在请求图像或纹理句柄之前为互操作注册纹理。

.. _direct3d-interoperability:

4.21.1.2. Direct3D 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct3D 互操作性支持 Direct3D9、Direct3D10 和 Direct3D11，但不支持 Direct3D12，这里我们关注 Direct3D11，对于 Direct3D9 和 Direct3D10 请参阅 CUDA 编程指南 12.9。可以映射到 CUDA 地址空间的 Direct3D 资源是 Direct3D 缓冲区、纹理和表面。这些资源使用 ``cudaGraphicsD3D11RegisterResource()`` 注册。

CUDA 上下文只能与使用 ``DriverType`` 设置为 ``D3D_DRIVER_TYPE_HARDWARE`` 创建的 Direct3D11 设备互操作。

**示例：2D 纹理 Direct3D11 互操作性**

以下代码片段来自 simpleD3D11Texture 示例：`NVIDIA/cuda-samples <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D11Texture>`__。完整示例包括大量 DX11 样板代码，这里我们关注 CUDA 方面。

CUDA 核函数 ``cuda_kernel_texture_2d`` 在频闪蓝色背景上绘制移动的红色/绿色阴影图案的 2D 纹理，它依赖于先前的纹理值。底层数据是 2D CUDA 数组，其中行偏移由 pitch 定义。

.. code-block:: cuda

   /*
    * Paint a 2D texture with a moving red/green hatch pattern on a
    * strobing blue background.  Note that this kernel reads to and
    * writes from the texture, hence why this texture was not mapped
    * as WriteDiscard.
    */
   __global__ void cuda_kernel_texture_2d(unsigned char *surface, int width,
                                          int height, size_t pitch, float t) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     float *pixel;

     // in the case where, due to quantization into grids, we have
     // more threads than pixels, skip the threads which don't
     // correspond to valid pixels
     if (x >= width || y >= height) return;

     // get a pointer to the pixel at (x,y)
     pixel = (float *)(surface + y * pitch) + 4 * x;

     // populate it
     float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width - 1.0f));
     float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));
     pixel[0] = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f);  // red
     pixel[1] = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f);  // green
     pixel[2] = 0.5f + 0.5f * cos(t);                       // blue
     pixel[3] = 1;                                          // alpha
   }

   extern "C" void cuda_texture_2d(void *surface, int width, int height,
                                   size_t pitch, float t) {
     cudaError_t error = cudaSuccess;

     dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
     dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

     cuda_kernel_texture_2d<<<Dg, Db>>>((unsigned char *)surface, width, height,
                                        pitch, t);

     error = cudaGetLastError();

     if (error != cudaSuccess) {
       printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
     }
   }

为了使指针和数据缓冲区保持在一起，使用以下数据结构：

.. code-block:: c++

   // Data structure for 2D texture shared between DX11 and CUDA
   struct {
     ID3D11Texture2D *pTexture;
     ID3D11ShaderResourceView *pSRView;
     cudaGraphicsResource *cudaResource;
     void *cudaLinearMemory;
     size_t pitch;
     int width;
     int height;
     int offsetInShader;
   } g_texture_2d;

在初始化 Direct3D 设备和纹理后，资源向 CUDA 注册一次。为了匹配 Direct3D 像素格式，CUDA 数组分配有相同的宽度和高度，以及与 Direct3D 纹理行 pitch 匹配的 pitch。

.. code-block:: c++

       // register the Direct3D resources that are used in the CUDA kernel
       // we'll read to and write from g_texture_2d, so don't set any special map flags for it
       cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource,
                                         g_texture_2d.pTexture,
                                         cudaGraphicsRegisterFlagsNone);
       getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
       // CUDA cannot write into the texture directly : the texture is seen as a
       // cudaArray and can only be mapped as a texture
       // Create a buffer so that CUDA can write into it
       // the pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
       cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch,
                       g_texture_2d.width * sizeof(float) * 4,
                       g_texture_2d.height);
       getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
       cudaMemset(g_texture_2d.cudaLinearMemory, 1,
                  g_texture_2d.pitch * g_texture_2d.height);

在渲染循环中，映射资源，启动 CUDA 核函数更新纹理数据，然后取消映射资源。在此步骤之后，使用 Direct3D 设备在屏幕上绘制更新的纹理。

.. code-block:: c++

       cudaStream_t stream = 0;
       const int nbResources = 3;
       cudaGraphicsResource *ppResources[nbResources] = {
           g_texture_2d.cudaResource, g_texture_3d.cudaResource,
           g_texture_cube.cudaResource,
       };
       cudaGraphicsMapResources(nbResources, ppResources, stream);
       getLastCudaError("cudaGraphicsMapResources(3) failed");

       // run kernels which will populate the contents of those textures
       RunKernels();

       // unmap the resources
       cudaGraphicsUnmapResources(nbResources, ppResources, stream);
       getLastCudaError("cudaGraphicsUnmapResources(3) failed");

最后，一旦 CUDA 不再需要这些资源，它们将取消注册并释放设备数组。

.. code-block:: c++

     // unregister the Cuda resources
     cudaGraphicsUnregisterResource(g_texture_2d.cudaResource);
     getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
     cudaFree(g_texture_2d.cudaLinearMemory);
     getLastCudaError("cudaFree (g_texture_2d) failed");

.. _sli-interoperability:

4.21.1.3. 可扩展链接接口 (SLI) 配置中的互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在具有多个 GPU 的系统中，所有支持 CUDA 的 GPU 都可以通过 CUDA 驱动程序和运行时作为单独的设备访问。当系统处于 SLI 模式时，这是不同的。SLI 是一种硬件配置的多 GPU 配置，通过跨多个 GPU 划分工作负载来提供更高的渲染性能。隐式 SLI 模式（驱动程序进行假设）不再受支持，但显式 SLI 仍受支持。显式 SLI 意味着应用程序通过 API（例如 Vulkan、DirectX、GL）了解和管理 SLI 组中所有设备的 SLI 状态。

当系统处于 SLI 模式时有特殊考虑：

- 一个 GPU 上的一个 CUDA 设备中的分配将消耗属于 Direct3D 或 OpenGL 设备 SLI 配置的其他 GPU 上的内存。因此，分配可能比预期更早失败。
- 应用程序应创建多个 CUDA 上下文，SLI 配置中的每个 GPU 一个。虽然这不是严格要求，但可以避免设备之间不必要的数据传输。应用程序可以使用 ``cudaD3D[9|10|11]GetDevices()`` （用于 Direct3D）和 ``cudaGLGetDevices()`` （用于 OpenGL）调用来识别当前帧和下一帧中执行渲染的设备的 CUDA 设备句柄。鉴于此信息，应用程序通常会选择合适的设备，并在 ``deviceList`` 参数设置为 ``cudaD3D[9|10|11]DeviceListCurrentFrame`` 或 ``cudaGLDeviceListCurrentFrame`` 时将 Direct3D 或 OpenGL 资源映射到 ``cudaD3D[9|10|11]GetDevices()`` 或 ``cudaGLGetDevices()`` 返回的 CUDA 设备。
- 从 ``cudaGraphicsD3D[9|10|11]RegisterResource`` 和 ``cudaGraphicsGLRegister[Buffer|Image]`` 返回的资源必须仅在发生注册的设备上使用。因此，在 SLI 配置中，当在不同 CUDA 设备上计算不同帧的数据时，需要分别为每个资源注册。

.. _external-resource-interoperability:

4.21.2. 外部资源互操作性
------------------------

外部资源互操作性允许 CUDA 导入由 API 显式导出的某些资源。这些对象通常使用操作系统原生的句柄导出，如 Linux 上的文件描述符或 Windows 上的 NT 句柄。这使得可以在其他 API 和 CUDA 之间高效地共享资源，而无需在两者之间复制或重复资源。它支持以下 API：Direct3D[11-12]、Vulkan 和 NVIDIA 软件通信接口互操作性。有两种类型的资源可以导入：

- **内存对象**

  可以使用 ``cudaImportExternalMemory()`` 导入到 CUDA 中。导入的内存对象随后可以在核函数中访问，使用 ``cudaExternalMemoryGetMappedBuffer()`` 映射到该内存对象上的设备指针，或使用 ``cudaExternalMemoryGetMappedMipmappedArray()`` 映射的 CUDA mipmapped 数组。根据内存对象的类型，可能在单个内存对象上设置多个映射。这些映射必须与导出 API 的映射设置相匹配。任何不匹配的映射都会导致未定义行为。导入的内存对象必须使用 ``cudaDestroyExternalMemory()`` 释放。释放内存对象不会释放对该对象的任何映射。因此，映射到该对象上的任何设备指针必须使用 ``cudaFree()`` 显式释放，映射到该对象上的任何 CUDA mipmapped 数组必须使用 ``cudaFreeMipmappedArray()`` 显式释放。在对象被销毁后访问该对象的映射是非法的。

- **同步对象**

  可以使用 ``cudaImportExternalSemaphore()`` 导入到 CUDA 中。导入的同步对象随后可以使用 ``cudaSignalExternalSemaphoresAsync()`` 发出信号，并使用 ``cudaWaitExternalSemaphoresAsync()`` 等待。在发出相应的信号之前发出等待是非法的。另外，根据导入的同步对象的类型，可能对如何发出信号和等待施加额外的约束，如后续小节所述。导入的信号量对象必须使用 ``cudaDestroyExternalSemaphore()`` 释放。在销毁信号量对象之前，所有未完成的信号和等待必须已完成。

.. _vulkan-interoperability:

4.21.2.1. Vulkan 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^

Vulkan 图形和计算工作负载在同一硬件上的耦合执行可以最大化 GPU 利用率并避免不必要的复制。注意，这不是 Vulkan 指南，我们只关注与 CUDA 的互操作性，有关 Vulkan 指南请参阅 `https://www.vulkan.org/learn#vulkan-tutorials <https://www.vulkan.org/learn#vulkan-tutorials>`_。

使 Vulkan-CUDA 互操作性工作的主要步骤涉及：

1. 初始化 Vulkan，创建并导出外部缓冲区和/或同步对象
2. 使用匹配的设备 UUID 设置 Vulkan 正在运行的 CUDA 设备
3. 获取内存和/或同步句柄
4. 使用这些句柄在 CUDA 中导入内存和/或同步对象
5. 将设备指针或 mipmapped 数组映射到内存对象上
6. 通过对同步对象发出信号和等待来定义执行顺序，从而在 CUDA 和 Vulkan 中交替使用导入的内存对象。

在本节中，将借助 simpleVulkan 示例解释上述步骤：`NVIDIA/cuda-samples <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan>`__。我们将逐步浏览该示例，重点介绍 CUDA 互操作性所需的部分。一些变体将使用独立的代码片段进行说明。

.. note::

   本节使用的代码示例采用直接内存分配和资源创建方式。由于多种原因（包括可创建实例数量的限制），这并不是最先进的做法。但是，要理解互操作性，需要了解底层的 Vulkan 代码和特定的标志。有关使用 `VulkanMemoryAllocator <https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator>`_ 的更先进示例，请参阅 `NVProSamples <https://github.com/nvpro-samples>`_ 仓库中的 *sample_cuda_interop* 。

整个示例中使用以下数据结构：

.. code-block:: c++

   class VulkanCudaSineWave : public VulkanBaseApp {
     typedef struct UniformBufferObject_st {
       mat4x4 modelViewProj;
     } UniformBufferObject;

     VkBuffer m_heightBuffer, m_xyBuffer, m_indexBuffer;
     VkDeviceMemory m_heightMemory, m_xyMemory, m_indexMemory;
     UniformBufferObject m_ubo;
     VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
     SineWaveSimulation m_sim;
     cudaStream_t m_stream;
     cudaExternalSemaphore_t m_cudaWaitSemaphore, m_cudaSignalSemaphore, m_cudaTimelineSemaphore;
     cudaExternalMemory_t m_cudaVertMem;
     float *m_cudaHeightMap;
     // ...

.. _setting-up-a-vulkan-device:

4.21.2.1.1. 设置 Vulkan 设备
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了导出内存对象，必须创建启用了 ``VK_KHR_external_memory_capabilities`` 扩展的 Vulkan 实例，以及启用了 ``VK_KHR_external_memory`` 的设备。此外，必须启用特定于平台的句柄类型，对于 Windows 为 ``VK_KHR_external_memory_win32`` ，对于基于 UNIX 的系统为 ``VK_KHR_external_memory_fd`` 。

类似地，对于导出同步对象，需要在设备级别启用 ``VK_KHR_external_semaphore_capabilities`` ，在实例级别启用 ``VK_KHR_external_semaphore`` 。以及用于句柄的特定于平台的扩展，即 Windows 的 ``VK_KHR_external_semaphore_win32`` 和基于 Unix 的系统的 ``VK_KHR_external_semaphore_fd`` 。

在 simpleVulkan 示例中，这些扩展通过以下枚举启用。

.. code-block:: c++

     std::vector<const char *> getRequiredExtensions() const {
       std::vector<const char *> extensions;
       extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
       extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
       return extensions;
     }

     std::vector<const char *> getRequiredDeviceExtensions() const {
       std::vector<const char *> extensions;
       extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
       extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
       extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
   #ifdef _WIN64
       extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
       extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
   #else
       extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
       extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
   #endif /* _WIN64 */
       return extensions;
     }

然后将这些添加到 Vulkan 实例和设备创建信息中，详情请参阅 simpleVulkan 示例。

.. _initializing-cuda-with-matching-device-uuids:

4.21.2.1.2. 使用匹配的设备 UUID 初始化 CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

导入由 Vulkan 导出的内存和同步对象时，必须在创建它们的同一设备上导入和映射它们。与创建对象的 Vulkan 物理设备对应的 CUDA 设备，可以通过比较 CUDA 设备的 UUID 与 Vulkan 物理设备的 UUID 来确定，如以下来自 simpleVulkan 示例的代码片段所示，其中 ``vkDeviceUUID`` 是 Vulkan API 结构 ``vkPhysicalDeviceIDProperties.deviceUUID`` 的成员，定义了当前 Vulkan 实例的物理设备 id。

.. code-block:: c++

   // from the CUDA example `simpleVulkan`
   int SineWaveSimulation::initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE) {
     int current_device = 0;
     int device_count = 0;
     int devices_prohibited = 0;

     cudaDeviceProp deviceProp;
     checkCudaErrors(cudaGetDeviceCount(&device_count));

     if (device_count == 0) {
       fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
       exit(EXIT_FAILURE);
     }

     // Find the GPU which is selected by Vulkan
     while (current_device < device_count) {
       cudaGetDeviceProperties(&deviceProp, current_device);

       if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
         // Compare the cuda device UUID with vulkan UUID
         int ret = memcmp((void *)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
         if (ret == 0) {
           checkCudaErrors(cudaSetDevice(current_device));
           checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
           printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                  current_device, deviceProp.name, deviceProp.major,
                  deviceProp.minor);

           return current_device;
         }

       } else {
         devices_prohibited++;
       }

       current_device++;
     }

     if (devices_prohibited == device_count) {
       fprintf(stderr,
               "CUDA error:"
               " No Vulkan-CUDA Interop capable GPU found.\n");
       exit(EXIT_FAILURE);
     }

     return -1;
   }

请注意，Vulkan 物理设备不应属于包含多个 Vulkan 物理设备的设备组。也就是说，由 ``vkEnumeratePhysicalDeviceGroups`` 返回的、包含给定 Vulkan 物理设备的设备组，其物理设备计数必须为 1。

.. _exporting-vulkan-memory-objects:

4.21.2.1.3. 导出 Vulkan 内存对象
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了导出 Vulkan 内存对象，必须创建具有相应导出标志的缓冲区。请注意，句柄类型的枚举是特定于平台的。

.. code-block:: c++

   void VulkanBaseApp::createExternalBuffer(
       VkDeviceSize size, VkBufferUsageFlags usage,
       VkMemoryPropertyFlags properties,
       VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer &buffer,
       VkDeviceMemory &bufferMemory) {
     VkBufferCreateInfo bufferInfo = {};
     bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
     bufferInfo.size = size;
     bufferInfo.usage = usage;
     bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

     VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
     externalMemoryBufferInfo.sType =
         VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
     externalMemoryBufferInfo.handleTypes = extMemHandleType;
     bufferInfo.pNext = &externalMemoryBufferInfo;

     if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
       throw std::runtime_error("failed to create buffer!");
     }

     VkMemoryRequirements memRequirements;
     vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

   #ifdef _WIN64
     WindowsSecurityAttributes winSecurityAttributes;

     VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
     vulkanExportMemoryWin32HandleInfoKHR.sType =
         VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
     vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
     vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
     vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
         DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
     vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
   #endif /* _WIN64 */
     VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
     vulkanExportMemoryAllocateInfoKHR.sType =
         VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
   #ifdef _WIN64
     vulkanExportMemoryAllocateInfoKHR.pNext =
         extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
             ? &vulkanExportMemoryWin32HandleInfoKHR
             : NULL;
     vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
   #else
     vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
     vulkanExportMemoryAllocateInfoKHR.handleTypes =
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
   #endif /* _WIN64 */
     VkMemoryAllocateInfo allocInfo = {};
     allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
     allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
     allocInfo.allocationSize = memRequirements.size;
     allocInfo.memoryTypeIndex = findMemoryType(
         m_physicalDevice, memRequirements.memoryTypeBits, properties);

     if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) !=
         VK_SUCCESS) {
       throw std::runtime_error("failed to allocate external buffer memory!");
     }

     vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
   }

.. _exporting-vulkan-synchronization-objects:

4.21.2.1.4. 导出 Vulkan 同步对象
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 GPU 上执行的 Vulkan API 调用是异步的。为了定义执行顺序，Vulkan 中提供了可以与 CUDA 共享的信号量和围栏（fence）。与内存对象类似，信号量可以由 Vulkan 导出，它们需要根据信号量的类型使用导出标志创建。信号量有二进制（binary）信号量和时间轴（timeline）信号量两种。二进制信号量只有 1 位计数器，即已发出信号或未发出信号。时间轴信号量具有 64 位计数器，可用于使用同一信号量定义执行顺序。在 simpleVulkan 示例中，时间轴信号量和二进制信号量都有对应的代码路径。

.. code-block:: c++

   void VulkanBaseApp::createExternalSemaphore(
       VkSemaphore &semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType) {
     VkSemaphoreCreateInfo semaphoreInfo = {};
     semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
     VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
     exportSemaphoreCreateInfo.sType =
         VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

   #ifdef _VK_TIMELINE_SEMAPHORE
     VkSemaphoreTypeCreateInfo timelineCreateInfo;
     timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
     timelineCreateInfo.pNext = NULL;
     timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
     timelineCreateInfo.initialValue = 0;
     exportSemaphoreCreateInfo.pNext = &timelineCreateInfo;
   #else
     exportSemaphoreCreateInfo.pNext = NULL;
   #endif /* _VK_TIMELINE_SEMAPHORE */
     exportSemaphoreCreateInfo.handleTypes = handleType;
     semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

     if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &semaphore) !=
         VK_SUCCESS) {
       throw std::runtime_error(
           "failed to create synchronization objects for a CUDA-Vulkan!");
     }
   }

.. _importing-memory-objects:

4.21.2.1.5. 导入内存对象
~~~~~~~~~~~~~~~~~~~~~~~~

由 Vulkan 导出的专用（dedicated）和非专用内存对象都可以导入到 CUDA 中。导入 Vulkan 专用内存对象时，必须设置 ``cudaExternalMemoryDedicated`` 标志。在 Windows 中，使用 ``VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`` 导出的 Vulkan 内存对象可以使用与该对象关联的 NT 句柄导入到 CUDA 中，如下所示。请注意，CUDA 不会取得该 NT 句柄的所有权，在不再需要该句柄时关闭它是应用程序的责任。NT 句柄持有对资源的引用，因此必须先显式释放它，然后才能释放底层内存。在 Linux 中，使用 ``VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`` 导出的 Vulkan 内存对象可以使用与该对象关联的文件描述符导入到 CUDA 中，如下所示。请注意，文件描述符一经导入，CUDA 即取得其所有权。在成功导入之后使用该文件描述符会导致未定义行为。

.. code-block:: c++

     // from the CUDA example `simpleVulkan`
     void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t &cudaMem,
                                   VkDeviceMemory &vkMem, VkDeviceSize size,
                                   VkExternalMemoryHandleTypeFlagBits handleType) {
       cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

       if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
         externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
       } else if (handleType &
                  VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
         externalMemoryHandleDesc.type =
             cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
       } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
         externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
       } else {
         throw std::runtime_error("Unknown handle type requested!");
       }

       externalMemoryHandleDesc.size = size;

   #ifdef _WIN64
       externalMemoryHandleDesc.handle.win32.handle =
           (HANDLE)getMemHandle(vkMem, handleType);
   #else
       externalMemoryHandleDesc.handle.fd =
           (int)(uintptr_t)getMemHandle(vkMem, handleType);
   #endif

       checkCudaErrors(
           cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));

如果存在命名句柄，使用 ``VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`` 导出的 Vulkan 内存对象也可以使用命名句柄导入，如以下独立代码片段所示。

.. code-block:: c++

   cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated) {
      cudaExternalMemory_t extMem = NULL;
      cudaExternalMemoryHandleDesc desc = {};

      memset(&desc, 0, sizeof(desc));

      desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
      desc.handle.win32.name = (void *)name;
      desc.size = size;
      if (isDedicated) {
          desc.flags |= cudaExternalMemoryDedicated;
      }

      cudaImportExternalMemory(&extMem, &desc);

      return extMem;
   }

.. _mapping-buffers-onto-imported-memory-objects:

4.21.2.1.6. 将缓冲区映射到导入的内存对象上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

导入内存对象后，必须先进行映射才能使用。可以将设备指针映射到导入的内存对象上，如下所示。映射的偏移量和大小必须与使用相应 Vulkan API 创建映射时指定的值相匹配。所有映射的设备指针都必须使用 ``cudaFree()`` 释放。

.. code-block:: c++

       // from the CUDA example `simpleVulkan`, continuation of function `importCudaExternalMemory`
       cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
       externalMemBufferDesc.offset = 0;
       externalMemBufferDesc.size = size;
       externalMemBufferDesc.flags = 0;

       checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem,
                                                         &externalMemBufferDesc));
     }

.. _mapping-mipmapped-arrays-onto-imported-memory-objects:

4.21.2.1.7. 将 mipmapped 数组映射到导入的内存对象上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以将 CUDA mipmapped 数组映射到导入的内存对象上，如下所示。偏移量、维度、格式和 mip 层级数量必须与使用相应 Vulkan API 创建映射时指定的值相匹配。此外，如果该 mipmapped 数组在 Vulkan 中被绑定为颜色目标（color target），则必须设置 ``cudaArrayColorAttachment`` 标志。所有映射的 mipmapped 数组都必须使用 ``cudaFreeMipmappedArray()`` 释放。以下独立代码片段展示了在将 mipmapped 数组映射到导入的内存对象上时，如何将 Vulkan 参数转换为相应的 CUDA 参数。

.. code-block:: c++

   cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
       cudaMipmappedArray_t mipmap = NULL;
       cudaExternalMemoryMipmappedArrayDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.offset = offset;
       desc.formatDesc = *formatDesc;
       desc.extent = *extent;
       desc.flags = flags;
       desc.numLevels = numLevels;

       // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
       cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

       return mipmap;
   }
   //end mapMipmappedArrayOntoExternalMemory

   //begin getCudaChannelFormatDescForVulkanFormat
   cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
   {
       cudaChannelFormatDesc d;

       memset(&d, 0, sizeof(d));

       switch (format) {
          case VK_FORMAT_R8_UINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R8_SINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R8G8_UINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R8G8_SINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R8G8B8A8_UINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R8G8B8A8_SINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R16_UINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R16_SINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R16G16_UINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R16G16_SINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R16G16B16A16_UINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R16G16B16A16_SINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R32_UINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R32_SINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R32_SFLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
          case VK_FORMAT_R32G32_UINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R32G32_SINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R32G32_SFLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
          case VK_FORMAT_R32G32B32A32_UINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
          case VK_FORMAT_R32G32B32A32_SINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
          case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
          default: assert(0);
       }
       return d;
   }
   //end getCudaChannelFormatDescForVulkanFormat

   //begin getCudaExtentForVulkanExtent
   cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType) {
       cudaExtent e = { 0, 0, 0 };

       switch (vkImageViewType) {
           case VK_IMAGE_VIEW_TYPE_1D:         e.width = vkExt.width; e.height = 0;            e.depth = 0;           break;
           case VK_IMAGE_VIEW_TYPE_2D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = 0;           break;
           case VK_IMAGE_VIEW_TYPE_3D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = vkExt.depth; break;
           case VK_IMAGE_VIEW_TYPE_CUBE:       e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
           case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   e.width = vkExt.width; e.height = 0;            e.depth = arrayLayers; break;
           case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
           case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
           default: assert(0);
       }

       return e;
   }
   //end getCudaExtentForVulkanExtent

   //begin getCudaMipmappedArrayFlagsForVulkanImage
   unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType,
                                                         VkImageUsageFlags vkImageUsageFlags,
                                                         bool allowSurfaceLoadStore) {
       unsigned int flags = 0;

       switch (vkImageViewType) {
           case VK_IMAGE_VIEW_TYPE_CUBE:       flags |= cudaArrayCubemap;                    break;
           case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
           case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   flags |= cudaArrayLayered;                    break;
           case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   flags |= cudaArrayLayered;                    break;
           default: break;
       }
       if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
           flags |= cudaArrayColorAttachment;
       }

       if (allowSurfaceLoadStore) {
           flags |= cudaArraySurfaceLoadStore;
       }

       return flags;
   }

.. _importing-synchronization-objects:

4.21.2.1.8. 导入同步对象
~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`` 导出的 Vulkan 信号量对象可以使用与该对象关联的文件描述符导入到 CUDA 中，如下所示。请注意，文件描述符一经导入，CUDA 即取得其所有权。在成功导入之后使用该文件描述符会导致未定义行为。而使用 ``VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT`` 导出的 Vulkan 信号量对象可以使用与该对象关联的 NT 句柄导入到 CUDA 中，如下所示。请注意，CUDA 不会取得该 NT 句柄的所有权，在不再需要该句柄时关闭它是应用程序的责任。NT 句柄持有对资源的引用，因此必须先显式释放它，然后才能释放底层信号量。另外，使用 ``VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT`` 导出的 Vulkan 信号量对象可以使用与该对象关联的全局共享 D3DKMT 句柄导入到 CUDA 中，如下所示。由于全局共享的 D3DKMT 句柄不持有对底层信号量的引用，当该资源的所有其他引用都被销毁时，它会被自动销毁。

.. code-block:: c++

     void importCudaExternalSemaphore(
         cudaExternalSemaphore_t &cudaSem, VkSemaphore &vkSem,
         VkExternalSemaphoreHandleTypeFlagBits handleType) {
       cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

   #ifdef _VK_TIMELINE_SEMAPHORE
       if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
         externalSemaphoreHandleDesc.type =
             cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
       } else if (handleType &
                  VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
         externalSemaphoreHandleDesc.type =
             cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
       } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
         externalSemaphoreHandleDesc.type =
             cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
       }
   #else
       if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
         externalSemaphoreHandleDesc.type =
             cudaExternalSemaphoreHandleTypeOpaqueWin32;
       } else if (handleType &
                  VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
         externalSemaphoreHandleDesc.type =
             cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
       } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
         externalSemaphoreHandleDesc.type =
             cudaExternalSemaphoreHandleTypeOpaqueFd;
       }
   #endif /* _VK_TIMELINE_SEMAPHORE */
       else {
         throw std::runtime_error("Unknown handle type requested!");
       }

   #ifdef _WIN64
       externalSemaphoreHandleDesc.handle.win32.handle =
           (HANDLE)getSemaphoreHandle(vkSem, handleType);
   #else
       externalSemaphoreHandleDesc.handle.fd =
           (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
   #endif

       externalSemaphoreHandleDesc.flags = 0;

       checkCudaErrors(
           cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
     }

.. _signaling-waiting-on-imported-synchronization-objects:

4.21.2.1.9. 对导入的同步对象发出信号/等待
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以对导入的 Vulkan 信号量发出信号和进行等待，如下所示。对信号量发出信号会将其置为已发信号状态；对于时间轴信号量，还会将计数器设置为信号调用中指定的值。等待该信号的相应等待操作必须在 Vulkan 中发出。此外，对于二进制信号量，等待该信号的等待操作必须在此信号发出之后才能发出。等待信号量会一直等到其达到已发信号状态或指定的等待值。已发信号的二进制信号量随后会被重置回未发信号状态。该等待所等待的相应信号必须在 Vulkan 中发出。此外，对于二进制信号量，必须先发出信号，然后才能发出此等待。在以下摘自 simpleVulkan 示例的代码中，只有当围绕顶点缓冲区的信号量由 Vulkan 发出信号后，才会调用模拟步骤/CUDA 核函数。在模拟步骤之后，会对另一个信号量发出信号；对于时间轴信号量，则由 CUDA 递增同一信号量，使得等待该信号量的 Vulkan 部分可以使用更新后的顶点缓冲区继续渲染。

.. code-block:: c++

   #ifdef _VK_TIMELINE_SEMAPHORE
       static uint64_t waitValue = 1;
       static uint64_t signalValue = 2;

       cudaExternalSemaphoreWaitParams waitParams = {};
       waitParams.flags = 0;
       waitParams.params.fence.value = waitValue;

       cudaExternalSemaphoreSignalParams signalParams = {};
       signalParams.flags = 0;
       signalParams.params.fence.value = signalValue;
       // Wait for vulkan to complete it's work
       checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaTimelineSemaphore,
                                                       &waitParams, 1, m_stream));
       // Now step the simulation, call CUDA kernel
       m_sim.stepSimulation(time, m_stream);
       // Signal vulkan to continue with the updated buffers
       checkCudaErrors(cudaSignalExternalSemaphoresAsync(
           &m_cudaTimelineSemaphore, &signalParams, 1, m_stream));

       waitValue += 2;
       signalValue += 2;
   #else
       cudaExternalSemaphoreWaitParams waitParams = {};
       waitParams.flags = 0;
       waitParams.params.fence.value = 0;

       cudaExternalSemaphoreSignalParams signalParams = {};
       signalParams.flags = 0;
       signalParams.params.fence.value = 0;

       // Wait for vulkan to complete it's work
       checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore,
                                                       &waitParams, 1, m_stream));
       // Now step the simulation, call CUDA kernel
       m_sim.stepSimulation(time, m_stream);
       // Signal vulkan to continue with the updated buffers
       checkCudaErrors(cudaSignalExternalSemaphoresAsync(
           &m_cudaSignalSemaphore, &signalParams, 1, m_stream));
   #endif /* _VK_TIMELINE_SEMAPHORE */

.. _opengl-interoperability-ext-res-int:

4.21.2.1.10. OpenGL 互操作性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如 :ref:`OpenGL 互操作性 <opengl-interoperability>` 中所述，传统的 OpenGL-CUDA 互操作通过 CUDA 直接消费在 OpenGL 中创建的句柄来工作。但是，由于 OpenGL 也可以消费在 Vulkan 中创建的内存和同步对象，因此存在另一种实现 OpenGL-CUDA 互操作的方法。本质上，由 Vulkan 导出的内存和同步对象可以同时导入到 OpenGL 和 CUDA 中，然后用于协调 OpenGL 和 CUDA 之间的内存访问。有关如何导入由 Vulkan 导出的内存和同步对象的更多详细信息，请参阅以下 OpenGL 扩展：

- ``GL_EXT_memory_object``
- ``GL_EXT_memory_object_fd``
- ``GL_EXT_memory_object_win32``
- ``GL_EXT_semaphore``
- ``GL_EXT_semaphore_fd``
- ``GL_EXT_semaphore_win32``

.. _direct3d-interoperability-ext-res-int:

4.21.2.2. Direct3D 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^

将 Direct3D[11|12] 资源导入到 CUDA 支持 Direct3D11 和 Direct3D12。这里我们只关注 Direct3D12，有关 Direct3D11，请参阅 CUDA 编程指南 12.9。

.. _matching-device-luids:

4.21.2.2.1. 匹配设备 LUID
~~~~~~~~~~~~~~~~~~~~~~~~~

导入由 Direct3D12 导出的内存和同步对象时，必须在创建它们的同一设备上导入和映射它们。与创建对象的 Direct3D12 设备对应的 CUDA 设备，可以通过比较 CUDA 设备的 LUID 与 Direct3D12 设备的 LUID 来确定，如以下代码示例所示。请注意，Direct3D12 设备不得在链接节点适配器（linked node adapter）上创建，即 ``ID3D12Device::GetNodeCount`` 返回的节点计数必须为 1。

.. code-block:: c++

   int getCudaDeviceForD3D12Device(ID3D12Device *d3d12Device) {
       LUID d3d12Luid = d3d12Device->GetAdapterLuid();

       int cudaDeviceCount;
       cudaGetDeviceCount(&cudaDeviceCount);

       for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
           cudaDeviceProp deviceProp;
           cudaGetDeviceProperties(&deviceProp, cudaDevice);
           char *cudaLuid = deviceProp.luid;

           if (!memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
               !memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart), sizeof(d3d12Luid.HighPart))) {
               return cudaDevice;
           }
       }
       return cudaInvalidDeviceId;
   }

.. _importing-memory-objects-dir3d-12-int:

4.21.2.2.2. 导入内存对象
~~~~~~~~~~~~~~~~~~~~~~~~

从 NT 句柄导入内存对象有几种不同的方式。请注意，在不再需要 NT 句柄时关闭它是应用程序的责任。NT 句柄持有对资源的引用，因此必须先显式释放它，然后才能释放底层内存。导入 Direct3D 资源时，必须像下面的代码片段那样设置 ``cudaExternalMemoryDedicated`` 标志。可共享的 Direct3D12 堆（heap）内存对象（通过在调用 ``ID3D12Device::CreateHeap`` 时设置 ``D3D12_HEAP_FLAG_SHARED`` 标志创建）可以使用与该对象关联的 NT 句柄导入到 CUDA 中，如下所示。

.. code-block:: c++

   cudaExternalMemory_t importD3D12HeapFromNTHandle(HANDLE handle, unsigned long long size) {
       cudaExternalMemory_t extMem = NULL;
       cudaExternalMemoryHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
       desc.handle.win32.handle = (void *)handle;
       desc.size = size;

       cudaImportExternalMemory(&extMem, &desc);

       // Input parameter 'handle' should be closed if it's not needed anymore
       CloseHandle(handle);

       return extMem;
   }

如果存在命名句柄，可共享的 Direct3D12 堆内存对象也可以使用命名句柄导入：

.. code-block:: c++

   cudaExternalMemory_t importD3D12HeapFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
       cudaExternalMemory_t extMem = NULL;
       cudaExternalMemoryHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
       desc.handle.win32.name = (void *)name;
       desc.size = size;

       cudaImportExternalMemory(&extMem, &desc);

       return extMem;
   }

可共享的 Direct3D12 已提交资源（committed resource，通过在调用 ``D3D12Device::CreateCommittedResource`` 时设置 ``D3D12_HEAP_FLAG_SHARED`` 标志创建）可以使用与该对象关联的 NT 句柄导入到 CUDA 中，如下所示。导入 Direct3D12 已提交资源时，必须设置 ``cudaExternalMemoryDedicated`` 标志。

.. code-block:: c++

   cudaExternalMemory_t importD3D12CommittedResourceFromNTHandle(HANDLE handle, unsigned long long size) {
       cudaExternalMemory_t extMem = NULL;
       cudaExternalMemoryHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
       desc.handle.win32.handle = (void *)handle;
       desc.size = size;
       desc.flags |= cudaExternalMemoryDedicated;

       cudaImportExternalMemory(&extMem, &desc);

       // Input parameter 'handle' should be closed if it's not needed anymore
       CloseHandle(handle);

       return extMem;
   }

如果存在命名句柄，可共享的 Direct3D12 已提交资源也可以使用命名句柄导入，如下所示。

.. code-block:: c++

   cudaExternalMemory_t importD3D12CommittedResourceFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
       cudaExternalMemory_t extMem = NULL;
       cudaExternalMemoryHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
       desc.handle.win32.name = (void *)name;
       desc.size = size;
       desc.flags |= cudaExternalMemoryDedicated;

       cudaImportExternalMemory(&extMem, &desc);

       return extMem;
   }

.. _mapping-buffers-onto-imported-memory-objects-dir3d-12-int:

4.21.2.2.3. 将缓冲区映射到导入的内存对象上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以将设备指针映射到导入的内存对象上，如下所示。映射的偏移量和大小必须与使用相应 Direct3D12 API 创建映射时指定的值相匹配。所有映射的设备指针都必须使用 ``cudaFree()`` 释放。

.. code-block:: c++

   void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
       void *ptr = NULL;
       cudaExternalMemoryBufferDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.offset = offset;
       desc.size = size;

       cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

       // Note: 'ptr' must eventually be freed using cudaFree()
       return ptr;
   }

.. _mapping-mipmapped-arrays-onto-imported-memory-objects-dir3d-12-int:

4.21.2.2.4. 将 mipmapped 数组映射到导入的内存对象上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以将 CUDA mipmapped 数组映射到导入的内存对象上，如下所示。偏移量、维度、格式和 mip 层级数量必须与使用相应 Direct3D12 API 创建映射时指定的值相匹配。此外，如果该 mipmapped 数组可以在 Direct3D12 中被绑定为渲染目标（render target），则必须设置 ``cudaArrayColorAttachment`` 标志。所有映射的 mipmapped 数组都必须使用 ``cudaFreeMipmappedArray()`` 释放。以下代码示例展示了在将 mipmapped 数组映射到导入的内存对象上时，如何将参数转换为相应的 CUDA 参数。

.. code-block:: c++

   cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
       cudaMipmappedArray_t mipmap = NULL;
       cudaExternalMemoryMipmappedArrayDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.offset = offset;
       desc.formatDesc = *formatDesc;
       desc.extent = *extent;
       desc.flags = flags;
       desc.numLevels = numLevels;

       // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
       cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

       return mipmap;
   }

   cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat)
   {
       cudaChannelFormatDesc d;

       memset(&d, 0, sizeof(d));

       switch (dxgiFormat) {
           case DXGI_FORMAT_R8_UINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R8_SINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R8G8_UINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R8G8_SINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R8G8B8A8_UINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R8G8B8A8_SINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R16_UINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R16_SINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R16G16_UINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R16G16_SINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R16G16B16A16_UINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R16G16B16A16_SINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R32_UINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R32_SINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R32_FLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
           case DXGI_FORMAT_R32G32_UINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R32G32_SINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R32G32_FLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
           case DXGI_FORMAT_R32G32B32A32_UINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
           case DXGI_FORMAT_R32G32B32A32_SINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
           case DXGI_FORMAT_R32G32B32A32_FLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
           default: assert(0);
       }
       return d;
   }

   cudaExtent getCudaExtentForD3D12Extent(UINT64 width, UINT height, UINT16 depthOrArraySize, D3D12_SRV_DIMENSION d3d12SRVDimension) {
       cudaExtent e = { 0, 0, 0 };

       switch (d3d12SRVDimension) {
           case D3D12_SRV_DIMENSION_TEXTURE1D:        e.width = width; e.height = 0;      e.depth = 0;                break;
           case D3D12_SRV_DIMENSION_TEXTURE2D:        e.width = width; e.height = height; e.depth = 0;                break;
           case D3D12_SRV_DIMENSION_TEXTURE3D:        e.width = width; e.height = height; e.depth = depthOrArraySize; break;
           case D3D12_SRV_DIMENSION_TEXTURECUBE:      e.width = width; e.height = height; e.depth = depthOrArraySize; break;
           case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:   e.width = width; e.height = 0;      e.depth = depthOrArraySize; break;
           case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:   e.width = width; e.height = height; e.depth = depthOrArraySize; break;
           case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
           default: assert(0);
       }

       return e;
   }

   unsigned int getCudaMipmappedArrayFlagsForD3D12Resource(D3D12_SRV_DIMENSION d3d12SRVDimension, D3D12_RESOURCE_FLAGS d3d12ResourceFlags, bool allowSurfaceLoadStore) {
       unsigned int flags = 0;

       switch (d3d12SRVDimension) {
           case D3D12_SRV_DIMENSION_TEXTURECUBE:      flags |= cudaArrayCubemap;                    break;
           case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
           case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:   flags |= cudaArrayLayered;                    break;
           case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:   flags |= cudaArrayLayered;                    break;
           default: break;
       }

       if (d3d12ResourceFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
           flags |= cudaArrayColorAttachment;
       }
       if (allowSurfaceLoadStore) {
           flags |= cudaArraySurfaceLoadStore;
       }

       return flags;
   }

.. _importing-synchronization-objects-dir3d-12-int:

4.21.2.2.5. 导入同步对象
~~~~~~~~~~~~~~~~~~~~~~~~

可共享的 Direct3D12 围栏（fence）对象（通过在调用 ``ID3D12Device::CreateFence`` 时设置 ``D3D12_FENCE_FLAG_SHARED`` 标志创建）可以使用与该对象关联的 NT 句柄导入到 CUDA 中，如下所示。请注意，在不再需要该句柄时关闭它是应用程序的责任。NT 句柄持有对资源的引用，因此必须先显式释放它，然后才能释放底层信号量。

.. code-block:: c++

   cudaExternalSemaphore_t importD3D12FenceFromNTHandle(HANDLE handle) {
       cudaExternalSemaphore_t extSem = NULL;
       cudaExternalSemaphoreHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
       desc.handle.win32.handle = handle;

       cudaImportExternalSemaphore(&extSem, &desc);

       // Input parameter 'handle' should be closed if it's not needed anymore
       CloseHandle(handle);

       return extSem;
   }

如果存在命名句柄，可共享的 Direct3D12 围栏对象也可以使用命名句柄导入，如下所示。

.. code-block:: c++

   cudaExternalSemaphore_t importD3D12FenceFromNamedNTHandle(LPCWSTR name) {
       cudaExternalSemaphore_t extSem = NULL;
       cudaExternalSemaphoreHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
       desc.handle.win32.name = (void *)name;

       cudaImportExternalSemaphore(&extSem, &desc);

       return extSem;
   }

.. _signaling-waiting-on-imported-synchronization-objects-dir3d-12-int:

4.21.2.2.6. 对导入的同步对象发出信号/等待
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一旦从 Direct3D12 导入了带有围栏的信号量，就可以对它们发出信号和进行等待。对围栏对象发出信号会设置其值。等待该信号的相应等待操作必须在 Direct3D12 中发出。请注意，等待该信号的等待操作必须在此信号发出之后才能发出。

.. code-block:: c++

   void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
       cudaExternalSemaphoreSignalParams params = {};

       memset(&params, 0, sizeof(params));

       params.params.fence.value = value;

       cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
   }

围栏对象会一直等待，直到其值大于或等于指定的值。它所等待的相应信号必须在 Direct3D12 中发出。请注意，必须先发出信号，然后才能发出此等待。

.. code-block:: c++

   void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
       cudaExternalSemaphoreWaitParams params = {};

       memset(&params, 0, sizeof(params));

       params.params.fence.value = value;

       cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
   }

.. _nvidia-software-communication-interface-interoperability-nvsci:

4.21.2.3. NVIDIA 软件通信接口互操作性 (NVSCI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NvSciBuf 和 NvSciSync 是为以下目的而开发的接口：

- NvSciBuf：允许应用程序在内存中分配和交换缓冲区
- NvSciSync：允许应用程序在操作边界处管理同步对象

有关这些接口的更多详细信息，请访问：`https://docs.nvidia.com/drive <https://docs.nvidia.com/drive>`_。

.. _importing-memory-objects-nvsci:

4.21.2.3.1. 导入内存对象
~~~~~~~~~~~~~~~~~~~~~~~~

要分配与给定 CUDA 设备兼容的 NvSciBuf 对象，必须如下所示在 NvSciBuf 属性列表中使用 ``NvSciBufGeneralAttrKey_GpuId`` 设置相应的 GPU id。应用程序可以选择指定以下属性：

- ``NvSciBufGeneralAttrKey_NeedCpuAccess`` ：指定缓冲区是否需要 CPU 访问
- ``NvSciBufRawBufferAttrKey_Align`` ：指定 ``NvSciBufType_RawBuffer`` 的对齐要求
- ``NvSciBufGeneralAttrKey_RequiredPerm`` ：可以为每个 NvSciBuf 内存对象实例针对不同的 UMD 配置不同的访问权限。例如，要为 GPU 提供对该缓冲区的只读访问权限，可以使用 ``NvSciBufObjDupWithReducePerm()`` 并以 ``NvSciBufAccessPerm_Readonly`` 作为输入参数创建一个重复的 NvSciBuf 对象。然后如下所示将这个新创建的、权限降低的重复对象导入到 CUDA 中
- ``NvSciBufGeneralAttrKey_EnableGpuCache`` ：用于控制 GPU L2 可缓存性
- ``NvSciBufGeneralAttrKey_EnableGpuCompression`` ：用于指定 GPU 压缩

.. note::

   有关这些属性及其有效输入选项的更多详细信息，请参阅 NvSciBuf 文档。

以下代码片段展示了它们的示例用法。

.. code-block:: c++

   NvSciBufObj createNvSciBufObject() {
      // Raw Buffer Attributes for CUDA
       NvSciBufType bufType = NvSciBufType_RawBuffer;
       uint64_t rawsize = SIZE;
       uint64_t align = 0;
       bool cpuaccess_flag = true;
       NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

       NvSciRmGpuId gpuid[] ={};
       CUuuid uuid;
       cuDeviceGetUuid(&uuid, dev);

       memcpy(&gpuid[0].bytes, &uuid.bytes, sizeof(uuid.bytes));
       // Disable cache on dev
       NvSciBufAttrValGpuCache gpuCache[] = {{gpuid[0], false}};
       NvSciBufAttrValGpuCompression gpuCompression[] = {{gpuid[0], NvSciBufCompressionType_GenericCompressible}};
       // Fill in values
       NvSciBufAttrKeyValuePair rawbuffattrs[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
            { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
            { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
            { NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid) },
            { NvSciBufGeneralAttrKey_EnableGpuCache, &gpuCache, sizeof(gpuCache) },
            { NvSciBufGeneralAttrKey_EnableGpuCompression, &gpuCompression, sizeof(gpuCompression) }
       };

       // Create list by setting attributes
       err = NvSciBufAttrListSetAttrs(attrListBuffer, rawbuffattrs,
               sizeof(rawbuffattrs)/sizeof(NvSciBufAttrKeyValuePair));

       NvSciBufAttrListCreate(NvSciBufModule, &attrListBuffer);

       // Reconcile And Allocate
       NvSciBufAttrListReconcile(&attrListBuffer, 1, &attrListReconciledBuffer,
                          &attrListConflictBuffer)
       NvSciBufObjAlloc(attrListReconciledBuffer, &bufferObjRaw);
       return bufferObjRaw;
   }

.. code-block:: c++

   NvSciBufObj bufferObjRo; // Readonly NvSciBuf memory obj
   // Create a duplicate handle to the same memory buffer with reduced permissions
   NvSciBufObjDupWithReducePerm(bufferObjRaw, NvSciBufAccessPerm_Readonly, &bufferObjRo);
   return bufferObjRo;

已分配的 NvSciBuf 内存对象可以使用 ``NvSciBufObj`` 句柄导入到 CUDA 中，如下所示。应用程序应查询已分配的 ``NvSciBufObj`` ，以获取填充 CUDA 外部内存描述符所需的属性。请注意，属性列表和 NvSciBuf 对象应由应用程序维护。如果导入到 CUDA 中的 NvSciBuf 对象也被其他驱动程序映射，则应用程序必须根据 ``NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency`` 输出属性值，使用 NvSciSync 对象（参阅 :ref:`导入同步对象 <importing-synchronization-objects-nvsci>` ）作为适当的屏障，以维护 CUDA 与其他驱动程序之间的一致性。

.. note::

   有关如何分配和维护 NvSciBuf 对象的更多详细信息，请参阅 `NvSciBuf API 文档 <https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-linux-sdk/common/topics/nvsci/NvStreams1.html>`_。

.. code-block:: c++

   cudaExternalMemory_t importNvSciBufObject (NvSciBufObj bufferObjRaw) {

       /*************** Query NvSciBuf Object **************/
       NvSciBufAttrKeyValuePair bufattrs[] = {
                   { NvSciBufRawBufferAttrKey_Size, NULL, 0 },
                   { NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, NULL, 0 },
                   { NvSciBufGeneralAttrKey_EnableGpuCompression, NULL, 0 }
       };
       NvSciBufAttrListGetAttrs(retList, bufattrs,
           sizeof(bufattrs)/sizeof(NvSciBufAttrKeyValuePair)));
                   ret_size = *(static_cast<const uint64_t*>(bufattrs[0].value));

       // Note cache and compression are per GPU attributes, so read values for specific gpu by comparing UUID
       // Read cacheability granted by NvSciBuf
       int numGpus = bufattrs[1].len / sizeof(NvSciBufAttrValGpuCache);
       NvSciBufAttrValGpuCache[] cacheVal = (NvSciBufAttrValGpuCache *)bufattrs[1].value;
       bool ret_cacheVal;
       for (int i = 0; i < numGpus; i++) {
           if (memcmp(gpuid[0].bytes, cacheVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
               ret_cacheVal = cacheVal[i].cacheability);
           }
       }

       // Read compression granted by NvSciBuf
       numGpus = bufattrs[2].len / sizeof(NvSciBufAttrValGpuCompression);
       NvSciBufAttrValGpuCompression[] compVal = (NvSciBufAttrValGpuCompression *)bufattrs[2].value;
       NvSciBufCompressionType ret_compVal;
       for (int i = 0; i < numGpus; i++) {
           if (memcmp(gpuid[0].bytes, compVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
               ret_compVal = compVal[i].compressionType);
           }
       }

       /*************** NvSciBuf Registration With CUDA **************/

       // Fill up CUDA_EXTERNAL_MEMORY_HANDLE_DESC
       cudaExternalMemoryHandleDesc memHandleDesc;
       memset(&memHandleDesc, 0, sizeof(memHandleDesc));
       memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
       memHandleDesc.handle.nvSciBufObject = bufferObjRaw;
       // Set the NvSciBuf object with required access permissions in this step
       memHandleDesc.handle.nvSciBufObject = bufferObjRo;
       memHandleDesc.size = ret_size;
       cudaImportExternalMemory(&extMemBuffer, &memHandleDesc);
       return extMemBuffer;
    }

.. _mapping-buffers-onto-imported-memory-objects-nvsci:

4.21.2.3.2. 将缓冲区映射到导入的内存对象上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以将设备指针映射到导入的内存对象上，如下所示。映射的偏移量和大小可以根据已分配的 ``NvSciBufObj`` 的属性填写。所有映射的设备指针都必须使用 ``cudaFree()`` 释放。

.. code-block:: c++

   void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
       void *ptr = NULL;
       cudaExternalMemoryBufferDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.offset = offset;
       desc.size = size;

       cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

       // Note: 'ptr' must eventually be freed using cudaFree()
       return ptr;
   }

.. _mapping-mipmapped-arrays-onto-imported-memory-objects-nvsci:

4.21.2.3.3. 将 mipmapped 数组映射到导入的内存对象上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以将 CUDA mipmapped 数组映射到导入的内存对象上，如下所示。偏移量、维度和格式可以根据已分配的 ``NvSciBufObj`` 的属性填写。所有映射的 mipmapped 数组都必须使用 ``cudaFreeMipmappedArray()`` 释放。以下代码示例展示了在将 mipmapped 数组映射到导入的内存对象上时，如何将 NvSciBuf 属性转换为相应的 CUDA 参数。

.. note::

   mip 层级数量必须为 1。

.. code-block:: c++

   cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
       cudaMipmappedArray_t mipmap = NULL;
       cudaExternalMemoryMipmappedArrayDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.offset = offset;
       desc.formatDesc = *formatDesc;
       desc.extent = *extent;
       desc.flags = flags;
       desc.numLevels = numLevels;

       // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
       cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

       return mipmap;
   }

.. _importing-synchronization-objects-nvsci:

4.21.2.3.4. 导入同步对象
~~~~~~~~~~~~~~~~~~~~~~~~

可以使用 ``cudaDeviceGetNvSciSyncAttributes()`` 生成与给定 CUDA 设备兼容的 NvSciSync 属性。返回的属性列表可用于创建保证与给定 CUDA 设备兼容的 ``NvSciSyncObj`` 。

.. code-block:: c++

   NvSciSyncObj createNvSciSyncObject() {
       NvSciSyncObj nvSciSyncObj
       int cudaDev0 = 0;
       int cudaDev1 = 1;
       NvSciSyncAttrList signalerAttrList = NULL;
       NvSciSyncAttrList waiterAttrList = NULL;
       NvSciSyncAttrList reconciledList = NULL;
       NvSciSyncAttrList newConflictList = NULL;

       NvSciSyncAttrListCreate(module, &signalerAttrList);
       NvSciSyncAttrListCreate(module, &waiterAttrList);
       NvSciSyncAttrList unreconciledList[2] = {NULL, NULL};
       unreconciledList[0] = signalerAttrList;
       unreconciledList[1] = waiterAttrList;

       cudaDeviceGetNvSciSyncAttributes(signalerAttrList, cudaDev0, CUDA_NVSCISYNC_ATTR_SIGNAL);
       cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDev1, CUDA_NVSCISYNC_ATTR_WAIT);

       NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList, &newConflictList);

       NvSciSyncObjAlloc(reconciledList, &nvSciSyncObj);

       return nvSciSyncObj;
   }

（如上创建的）NvSciSync 对象可以使用 ``NvSciSyncObj`` 句柄导入到 CUDA 中，如下所示。请注意，即使在导入之后， ``NvSciSyncObj`` 句柄的所有权仍归应用程序所有。

.. code-block:: c++

   cudaExternalSemaphore_t importNvSciSyncObject(void* nvSciSyncObj) {
       cudaExternalSemaphore_t extSem = NULL;
       cudaExternalSemaphoreHandleDesc desc = {};

       memset(&desc, 0, sizeof(desc));

       desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
       desc.handle.nvSciSyncObj = nvSciSyncObj;

       cudaImportExternalSemaphore(&extSem, &desc);

       // Deleting/Freeing the nvSciSyncObj beyond this point will lead to undefined behavior in CUDA

       return extSem;
   }

.. _signaling-waiting-on-imported-synchronization-objects-nvsci:

4.21.2.3.5. 对导入的同步对象发出信号/等待
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以对导入的 ``NvSciSyncObj`` 对象发出信号，如下所述。对由 NvSciSync 支持的信号量对象发出信号会初始化作为输入传递的围栏参数。与前述信号对应的等待操作会等待此围栏参数。此外，等待该信号的等待操作必须在此信号发出之后才能发出。如果将标志设置为 ``cudaExternalSemaphoreSignalSkipNvSciBufMemSync`` ，则会跳过默认作为信号操作一部分执行的内存同步操作（针对此进程中所有导入的 NvSciBuf）。当 ``NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency`` 为 FALSE 时，应设置此标志。

.. code-block:: c++

   void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
       cudaExternalSemaphoreSignalParams signalParams = {};

       memset(&signalParams, 0, sizeof(signalParams));

       signalParams.params.nvSciSync.fence = (void*)fence;
       signalParams.flags = 0; //OR cudaExternalSemaphoreSignalSkipNvSciBufMemSync

       cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);

   }

可以等待导入的 ``NvSciSyncObj`` 对象，如下所述。等待由 NvSciSync 支持的信号量对象会一直等到输入的围栏参数由相应的信号发送方发出信号。此外，必须先发出信号，然后才能发出等待。如果将标志设置为 ``cudaExternalSemaphoreWaitSkipNvSciBufMemSync`` ，则会跳过默认作为信号操作一部分执行的内存同步操作（针对此进程中所有导入的 NvSciBuf）。当 ``NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency`` 为 FALSE 时，应设置此标志。

.. code-block:: c++

   void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
        cudaExternalSemaphoreWaitParams waitParams = {};

       memset(&waitParams, 0, sizeof(waitParams));

       waitParams.params.nvSciSync.fence = (void*)fence;
       waitParams.flags = 0; //OR cudaExternalSemaphoreWaitSkipNvSciBufMemSync

       cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);
   }
