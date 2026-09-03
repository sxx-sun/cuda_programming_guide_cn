.. _cpp-language-support:

5.3. C++ 语言支持
=================

``nvcc`` 根据以下规范处理 CUDA 和设备代码：

- **C++03** (ISO/IEC 14882:2003)， ``--std=c++03`` 标志。

- **C++11** (ISO/IEC 14882:2011)， ``--std=c++11`` 标志。

- **C++14** (ISO/IEC 14882:2014)， ``--std=c++14`` 标志。

- **C++17** (ISO/IEC 14882:2017)， ``--std=c++17`` 标志。

- **C++20** (ISO/IEC 14882:2020)， ``--std=c++20`` 标志。

- **C++23** (ISO/IEC 14882:2024)， ``--std=c++23`` 标志。

传递 ``nvcc`` ``-std=c++<version>`` 标志会启用与指定版本相关的所有 C++ 特性，并以相应的 C++ 方言选项调用主机预处理器、编译器和链接器。

编译器支持所支持标准的所有语言特性，但以下章节中报告的限制除外。

.. _c-11-language-features:

5.3.1. C++11 语言特性
---------------------

.. list-table:: NVCC 设备代码支持的 C++11 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++11 提案
     - NVCC/CUDA Toolkit 7.x
   * - 右值引用
     - `N2118 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html>`__
     - ✅
   * - ``*this`` 的右值引用
     - `N2439 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm>`__
     - ✅
   * - 通过右值初始化类对象
     - `N1610 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html>`__
     - ✅
   * - 非静态数据成员初始化器
     - `N2756 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm>`__
     - ✅
   * - 可变参数模板
     - `N2242 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf>`__
     - ✅
   * - 扩展可变参数模板模板参数
     - `N2555 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf>`__
     - ✅
   * - 初始化列表
     - `N2672 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm>`__
     - ✅
   * - 静态断言
     - `N1720 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html>`__
     - ✅
   * - ``auto`` 类型变量
     - `N1984 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf>`__
     - ✅
   * - 多声明符 ``auto``
     - `N1737 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf>`__
     - ✅
   * - 移除 ``auto`` 作为存储类说明符
     - `N2546 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm>`__
     - ✅
   * - 新函数声明器语法
     - `N2541 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm>`__
     - ✅
   * - Lambda 表达式
     - `N2927 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf>`__
     - ✅
   * - 表达式的声明类型
     - `N2343 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf>`__
     - ✅
   * - 不完整返回类型
     - `N3276 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf>`__
     - ✅
   * - 右尖括号
     - `N1757 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html>`__
     - ✅
   * - 函数模板的默认模板参数
     - `DR226 <http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226>`__
     - ✅
   * - 解决表达式的 SFINAE 问题
     - `DR339 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html>`__
     - ✅
   * - 别名模板
     - `N2258 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf>`__
     - ✅
   * - 外部模板
     - `N1987 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm>`__
     - ✅
   * - 空指针常量
     - `N2431 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf>`__
     - ✅
   * - 强类型枚举
     - `N2347 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf>`__
     - ✅
   * - 枚举的前向声明
     - | `N2764 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf>`__
       | `DR1206 <http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206>`__
     - ✅
   * - 标准化属性语法
     - `N2761 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf>`__
     - ✅
   * - 广义常量表达式
     - `N2235 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf>`__
     - ✅
   * - 对齐支持
     - `N2341 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf>`__
     - ✅
   * - 条件支持的行为
     - `N1627 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf>`__
     - ✅
   * - 将未定义行为改为可诊断错误
     - `N1727 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf>`__
     - ✅
   * - 委托构造函数
     - `N1986 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf>`__
     - ✅
   * - 继承构造函数
     - `N2540 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm>`__
     - ✅
   * - 显式转换运算符
     - `N2437 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf>`__
     - ✅
   * - 新字符类型
     - `N2249 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html>`__
     - ✅
   * - Unicode 字符串字面量
     - `N2442 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm>`__
     - ✅
   * - 原始字符串字面量
     - `N2442 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm>`__
     - ✅
   * - 字面量中的通用字符名
     - `N2170 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html>`__
     - ✅
   * - 用户定义字面量
     - `N2765 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf>`__
     - ✅
   * - 标准布局类型
     - `N2342 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm>`__
     - ✅
   * - 默认函数
     - `N2346 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm>`__
     - ✅
   * - 删除函数
     - `N2346 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm>`__
     - ✅
   * - 扩展友元声明
     - `N1791 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf>`__
     - ✅
   * - 扩展 ``sizeof``
     - | `N2253 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html>`__
       | `DR850 <http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850>`__
     - ✅
   * - 内联命名空间
     - `N2535 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm>`__
     - ✅
   * - 无限制联合
     - `N2544 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf>`__
     - ✅
   * - 局部和未命名类型作为模板参数
     - `N2657 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm>`__
     - ✅
   * - 基于范围的 for 循环
     - `N2930 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html>`__
     - ✅
   * - 显式 ``virtual`` 覆盖
     - | `N2928 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm>`__
       | `N3206 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm>`__
       | `N3272 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm>`__
     - ✅
   * - 垃圾回收和基于可达性的泄漏检测的最低支持
     - `N2670 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm>`__
     - ❌
   * - 允许移动构造函数抛出异常 [noexcept]
     - `N3050 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html>`__
     - ✅
   * - 定义移动特殊成员函数
     - `N3053 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html>`__
     - ✅

**并发**

.. list-table:: NVCC 设备代码支持的 C++11 并发特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++11 提案
     - NVCC/CUDA Toolkit 7.x
   * - 序列点
     - `N2239 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html>`__
     - ❌
   * - 原子操作
     - `N2427 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html>`__
     - ❌
   * - 强比较并交换
     - `N2748 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html>`__
     - ❌
   * - 双向栅栏
     - `N2752 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm>`__
     - ❌
   * - 内存模型
     - `N2429 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm>`__
     - ❌
   * - 数据依赖排序：原子和内存模型
     - `N2664 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm>`__
     - ❌
   * - 传播异常
     - `N2179 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html>`__
     - ❌
   * - 允许在信号处理程序中使用原子
     - `N2547 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm>`__
     - ❌
   * - 线程局部存储
     - `N2659 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm>`__
     - ❌
   * - 并发动态初始化和销毁
     - `N2660 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm>`__
     - ❌

**C99 在 C++11 中的特性**

.. list-table:: NVCC 设备代码支持的 C99 特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++11 提案
     - NVCC/CUDA Toolkit 7.x
   * - ``__func__`` 预定义标识符
     - `N2340 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm>`__
     - ✅
   * - C99 预处理器
     - `N1653 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm>`__
     - ✅
   * - ``long long``
     - `N1811 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf>`__
     - ✅
   * - 扩展整数类型
     - `N1988 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf>`__
     - ❌

.. _c-14-language-features:

5.3.2. C++14 语言特性
---------------------

.. list-table:: NVCC 设备代码支持的 C++14 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++14 提案
     - NVCC/CUDA Toolkit 9.x
   * - 某些 C++ 上下文转换的调整
     - `N3323 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf>`__
     - ✅
   * - 二进制字面量
     - `N3472 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf>`__
     - ✅
   * - 推导返回类型的函数
     - `N3638 <https://isocpp.org/files/papers/N3638.html>`__
     - ✅
   * - 广义 lambda 捕获（init-capture）
     - `N3648 <https://isocpp.org/files/papers/N3648.html>`__
     - ✅
   * - 泛型（多态）lambda 表达式
     - `N3649 <https://isocpp.org/files/papers/N3649.html>`__
     - ✅
   * - 变量模板
     - `N3651 <https://isocpp.org/files/papers/N3651.pdf>`__
     - ✅
   * - 放宽 constexpr 函数的要求
     - `N3652 <https://isocpp.org/files/papers/N3652.html>`__
     - ✅
   * - 成员初始化器和聚合
     - `N3653 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html>`__
     - ✅
   * - 澄清内存分配
     - `N3664 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html>`__
     - ❌
   * - 带大小的释放
     - `N3778 <https://isocpp.org/files/papers/n3778.html>`__
     - ❌
   * - ``[[deprecated]]`` 属性
     - `N3760 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html>`__
     - ✅
   * - 单引号作为数字分隔符
     - `N3781 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf>`__
     - ✅

.. _c-17-language-features:

5.3.3. C++17 语言特性
---------------------

.. list-table:: NVCC 设备代码支持的 C++17 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++17 提案
     - NVCC/CUDA Toolkit 11.x
   * - 移除三字符组
     - `N4086 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4086.html>`__
     - ✅
   * - ``u8`` 字符字面量
     - `N4267 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4267.html>`__
     - ✅
   * - 折叠表达式
     - `N4295 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4295.html>`__
     - ✅
   * - 命名空间和枚举器的属性
     - `N4266 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4266.html>`__
     - ✅
   * - 嵌套命名空间定义
     - `N4230 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html>`__
     - ✅
   * - 允许所有非类型模板参数的常量求值
     - `N4268 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4268.html>`__
     - ✅
   * - 扩展 ``static_assert``
     - `N3928 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3928.pdf>`__
     - ✅
   * - ``auto`` 从花括号初始化列表推导的新规则
     - `N3922 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html>`__
     - ✅
   * - ``[[fallthrough]]`` 属性
     - `P0188R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0188r1.pdf>`__
     - ✅
   * - ``[[nodiscard]]`` 属性
     - `P0189R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0189r1.pdf>`__
     - ✅
   * - ``[[maybe_unused]]`` 属性
     - `P0212R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0212r1.pdf>`__
     - ✅
   * - 聚合初始化扩展
     - `P0017R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html>`__
     - ✅
   * - ``constexpr`` lambda 的措辞
     - `P0170R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0170r1.pdf>`__
     - ✅
   * - 一元折叠和空参数包
     - `P0036R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0036r0.pdf>`__
     - ✅
   * - 泛化基于范围的 for 循环
     - `P0184R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0184r0.html>`__
     - ✅
   * - Lambda 按值捕获 ``*this``
     - `P0018R3 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html>`__
     - ✅
   * - ``enum class`` 变量的构造规则
     - `P0138R2 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0138r2.pdf>`__
     - ✅
   * - C++ 十六进制浮点字面量
     - `P0245R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0245r1.html>`__
     - ✅
   * - 保证复制省略
     - `P0135R1 <https://wg21.link/p0135>`__
     - ✅
   * - ``constexpr if``
     - `P0292R2 <https://wg21.link/p0292>`__
     - ✅
   * - 带初始化器的选择语句
     - `P0305R1 <https://wg21.link/p0305>`__
     - ✅
   * - 类模板参数推导
     - `P0091R3 <https://wg21.link/p0091>`__
     - ✅
   * - 使用 ``auto`` 声明非类型模板参数
     - `P0127R2 <https://wg21.link/p0127>`__
     - ✅
   * - 结构化绑定
     - `P0217R3 <https://wg21.link/p0217>`__
     - ✅
   * - 内联变量
     - `P0386R2 <https://wg21.link/p0386r2>`__
     - ✅
   * - 允许在模板模板参数中使用 ``typename``
     - `N4051 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4051.html>`__
     - ✅
   * - 过度对齐数据的动态内存分配
     - `P0035R4 <https://wg21.link/p0035>`__
     - ✅
   * - 改进惯用 C++ 的表达式求值顺序
     - `P0145R3 <https://wg21.link/p0145>`__
     - ✅
   * - 类模板参数推导（补充）
     - `P0512R0 <https://wg21.link/p0512r0>`__
     - ✅
   * - 使用属性命名空间时无需重复
     - `P0028R4 <https://wg21.link/p0028>`__
     - ✅
   * - 忽略不支持的非标准属性
     - `P0283R2 <https://wg21.link/p0283>`__
     - ✅
   * - 移除 ``register`` 关键字的弃用用法
     - `P0001R1 <https://wg21.link/p0001>`__
     - ✅
   * - 移除弃用的 ``operator++(bool)``
     - `P0002R1 <https://wg21.link/p0002>`__
     - ✅
   * - 使异常规范成为类型系统的一部分
     - `P0012R1 <https://wg21.link/p0012>`__
     - ✅
   * - C++17 的 ``__has_include``
     - `P0061R1 <https://wg21.link/p0061>`__
     - ✅
   * - 重新措辞继承构造函数（核心问题 1941 等）
     - `P0136R1 <https://wg21.link/p0136>`__
     - ✅
   * - DR 150，模板模板参数的匹配
     - `P0522R0 <https://wg21.link/p0522r0>`__
     - ✅
   * - 移除动态异常规范
     - `P0003R5 <https://wg21.link/p0003r5>`__
     - ✅
   * - using 声明中的包扩展
     - `P0195R2 <https://wg21.link/p0195r2>`__
     - ✅
   * - ``byte`` 类型定义
     - `P0298R0 <https://wg21.link/p0298r0>`__
     - ✅
   * - DR 727，类内显式实例化
     - `CWG727 <https://cplusplus.github.io/CWG/issues/727.html>`__
     - ✅

.. _c-20-language-features:

5.3.4. C++20 语言特性
---------------------

需要 GCC 版本 ≥ 10.0、Clang 版本 ≥ 10.0、Microsoft Visual Studio ≥ 2022 和 nvc++ 版本 ≥ 20.7。

.. note::

   以 "DR:" 为前缀的条目是缺陷报告（Defect Report）的解决方案。它们修正了标准，并同样适用于较早的 C++ 标准模式（例如 C++17）；为完整性起见在此列出，并非 C++20 特有。

.. list-table:: NVCC 设备代码支持的 C++20 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++20 提案
     - NVCC/CUDA Toolkit 12.x
   * - 位域的默认成员初始化器
     - `P0683R1 <https://wg21.link/p0683r1>`__
     - ✅
   * - 修复 ``const`` 限定成员指针
     - `P0704R1 <https://wg21.link/p0704r1>`__
     - ✅
   * - 允许 lambda 捕获 ``[=, this]``
     - `P0409R2 <https://wg21.link/p0409r2>`__
     - ✅
   * - 预处理逗号省略的 ``__VA_OPT__``
     - `P0306R4 <https://wg21.link/p0306r4>`__
     - ✅
   * - 指定初始化器
     - `P0329R4 <https://wg21.link/p0329r4>`__
     - ✅
   * - 泛型 lambda 的熟悉模板语法
     - `P0428R2 <https://wg21.link/p0428r2>`__
     - ✅
   * - 概念
     - `P0734R0 <https://wg21.link/p0734r0>`__
     - ✅
   * - 一致比较（ ``operator<=>`` ）
     - `P0515R3 <https://wg21.link/p0515r3>`__
     - ✅
   * - 立即函数（ ``consteval`` ）
     - `P1073R3 <https://wg21.link/p1073r3>`__
     - ✅
   * - ``std::is_constant_evaluated``
     - `P0595R2 <https://wg21.link/p0595r2>`__
     - ✅
   * - constexpr 限制放宽
     - `P1002R1 <https://wg21.link/p1002r1>`__
     - ✅
   * - 特性测试宏
     - `P0941R2 <https://wg21.link/p0941r2>`__
     - ✅
   * - 模块
     - `P1103R3 <https://wg21.link/p1103r3>`__
     - ❌
   * - 协程
     - `P0912R5 <https://wg21.link/p0912r5>`__
     - ❌
   * - ``constinit``
     - `P1143R2 <https://wg21.link/p1143r2>`__
     - ✅
   * - 向量的列表推导
     - `P0702R1 <https://wg21.link/p0702r1>`__
     - ✅
   * - 带初始化器的基于范围的 for 语句
     - `P0614R1 <https://wg21.link/p0614r1>`__
     - ✅
   * - 简化隐式 lambda 捕获
     - `P0588R1 <https://wg21.link/p0588r1>`__
     - ✅
   * - ADL 和不可见的函数模板
     - `P0846R0 <https://wg21.link/p0846r0>`__
     - ✅
   * - 默认拷贝构造函数的 ``const`` 不匹配
     - `P0641R2 <https://wg21.link/p0641r2>`__
     - ✅
   * - 减少 ``constexpr`` 函数的急切实例化
     - `P0859R0 <https://wg21.link/p0859r0>`__
     - ✅
   * - 特化的访问检查
     - `P0692R1 <https://wg21.link/p0692r1>`__
     - ✅
   * - 无状态 lambda 的默认可构造和可赋值
     - `P0624R2 <https://wg21.link/p0624r2>`__
     - ✅
   * - 未求值上下文中的 Lambda
     - `P0315R4 <https://wg21.link/p0315r4>`__
     - ✅
   * - 空对象的语言支持
     - `P0840R2 <https://wg21.link/p0840r2>`__
     - ✅
   * - 放宽 range-for 循环自定义点查找规则
     - `P0962R1 <https://wg21.link/p0962r1>`__
     - ✅
   * - 允许结构化绑定访问可访问成员
     - `P0969R0 <https://wg21.link/p0969r0>`__
     - ✅
   * - 放宽结构化绑定自定义点查找规则
     - `P0961R1 <https://wg21.link/p0961r1>`__
     - ✅
   * - 去除 ``typename`` 的过度使用
     - `P0634R3 <https://wg21.link/p0634r3>`__
     - ✅
   * - 允许在 lambda init-capture 中展开包
     - `P0780R2 <https://wg21.link/p0780r2>`__ ， `P2095R0 <https://wg21.link/p2095r0>`__
     - ✅
   * - ``likely`` 和 ``unlikely`` 属性的建议措辞
     - `P0479R5 <https://wg21.link/p0479r5>`__
     - ✅
   * - 弃用通过 ``[=]`` 隐式捕获 ``this``
     - `P0806R2 <https://wg21.link/p0806r2>`__
     - ✅
   * - 非类型模板参数中的类类型
     - `P0732R2 <https://wg21.link/p0732r2>`__
     - ✅
   * - 非类型模板参数的不一致性
     - `P1907R1 <https://wg21.link/p1907r1>`__
     - ✅
   * - 带填充位的原子比较交换
     - `P0528R3 <https://wg21.link/p0528r3>`__
     - ✅
   * - 可变大小类的高效带大小 ``delete``
     - `P0722R3 <https://wg21.link/p0722r3>`__
     - ✅
   * - 允许在常量表达式中调用虚函数
     - `P1064R0 <https://wg21.link/p1064r0>`__
     - ✅
   * - 禁止用户声明构造函数的聚合
     - `P1008R1 <https://wg21.link/p1008r1>`__
     - ✅
   * - ``explicit(bool)``
     - `P0892R2 <https://wg21.link/p0892r2>`__
     - ✅
   * - 有符号整数为二进制补码
     - `P1236R1 <https://wg21.link/p1236r1>`__
     - ✅
   * - ``char8_t``
     - `P0482R6 <https://wg21.link/p0482r6>`__
     - ✅
   * - 嵌套 ``inline`` 命名空间
     - `P1094R2 <https://wg21.link/p1094r2>`__
     - ✅
   * - 聚合的括号初始化
     - `P0960R3 <https://wg21.link/p0960r3>`__ ， `P1975R0 <https://wg21.link/p1975r0>`__
     - ✅
   * - DR：new 表达式中的数组大小推导
     - `P1009R2 <https://wg21.link/p1009r2>`__
     - ✅
   * - DR：从 ``T*`` 到 ``bool`` 的转换应视为窄化
     - `P1957R2 <https://wg21.link/p1957r2>`__
     - ✅
   * - 更强的 Unicode 要求
     - `P1041R4 <https://wg21.link/p1041r4>`__ ， `P1139R2 <https://wg21.link/p1139r2>`__
     - ✅
   * - 结构化绑定扩展
     - `P1091R3 <https://wg21.link/p1091r3>`__ ， `P1381R1 <https://wg21.link/p1381r1>`__
     - ✅
   * - 弃用 ``a[b,c]``
     - `P1161R3 <https://wg21.link/p1161r3>`__
     - ✅
   * - 弃用 ``volatile`` 的某些用法
     - `P1152R4 <https://wg21.link/p1152r4>`__
     - ✅
   * - ``[[nodiscard("with reason")]]``
     - `P1301R4 <https://wg21.link/p1301r4>`__
     - ✅
   * - ``using enum``
     - `P1099R5 <https://wg21.link/p1099r5>`__
     - ✅
   * - 聚合的类模板参数推导
     - `P1816R0 <https://wg21.link/p1816r0>`__ ， `P2082R1 <https://wg21.link/p2082r1>`__
     - ✅
   * - 别名模板的类模板参数推导
     - `P1814R0 <https://wg21.link/p1814r0>`__
     - ✅
   * - 允许转换为未知边界的数组
     - `P0388R4 <https://wg21.link/p0388r4>`__
     - ✅
   * - 布局兼容性和指针可互换性特性
     - `P0466R5 <https://wg21.link/p0466r5>`__
     - ✅
   * - DR：检查抽象类类型
     - `P0929R2 <https://wg21.link/p0929r2>`__
     - ✅
   * - DR：更多隐式移动
     - `P1825R0 <https://wg21.link/p1825r0>`__
     - ✅
   * - DR：伪析构符结束对象生命周期
     - `P0593R6 <https://wg21.link/p0593r6>`__
     - ✅

.. _c-23-language-features:

5.3.5. C++23 语言特性
---------------------

需要 GCC 版本 ≥ 14.0、Clang 版本 ≥ 18.0、Microsoft Visual Studio（不支持）和 nvc++ 版本 ≥ 24.3。

.. note::

   以 "DR:" 为前缀的条目是缺陷报告（Defect Report）的解决方案。它们修正了标准，并同样适用于较早的 C++ 标准模式（例如 C++17、C++20）；为完整性起见在此列出，并非 C++23 特有。

.. note::

   NVCC 列中的 **N/A** 表示该特性不适用于设备代码（例如，移除未使用的标准措辞，如垃圾回收支持或主机定义的行为）。

.. list-table:: NVCC 设备代码支持的 C++23 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++23 提案
     - NVCC/CUDA Toolkit
   * - | 核心问题 411、1656 和 2333 的建议解决方案；
       | 字符和字符串字面量中的数字和通用字符转义
     - `P2029R4 <https://wg21.link/p2029r4>`__
     - ✅
   * - （有符号） ``size_t`` 的字面量后缀
     - `P0330R8 <https://wg21.link/p0330r8>`__
     - ✅
   * - 使 lambda 的 ``()`` 更可选（去除 ``()`` ！）
     - `P1102R2 <https://wg21.link/p1102r2>`__
     - ✅
   * - ``if consteval``
     - `P1938R3 <https://wg21.link/p1938r3>`__
     - ✅
   * - 移除垃圾回收支持
     - `P2186R2 <https://wg21.link/p2186r2>`__
     - N/A
   * - DR：使用 Unicode 标准附件 31 的 C++ 标识符语法
     - `P1949R7 <https://wg21.link/p1949r7>`__
     - ✅
   * - DR：允许重复属性
     - `P2156R1 <https://wg21.link/p2156r1>`__
     - ✅
   * - 到 ``bool`` 的窄化上下文转换
     - `P1401R5 <https://wg21.link/p1401r5>`__
     - ❌
   * - 行拼接前修剪空白
     - `P2223R2 <https://wg21.link/p2223r2>`__
     - ✅
   * - 强制声明顺序布局
     - `P1847R4 <https://wg21.link/p1847r4>`__
     - ✅
   * - 混合字符串字面量拼接
     - `P2201R1 <https://wg21.link/p2201r1>`__
     - N/A
   * - ``constexpr`` 函数中的非字面量变量（以及标签和 goto）
     - `P2242R3 <https://wg21.link/p2242r3>`__
     - ✅
   * - 推导 ``this``
     - `P0847R7 <https://wg21.link/p0847r7>`__
     - ✅
   * - 一致的字符字面量编码
     - `P2316R2 <https://wg21.link/p2316r2>`__
     - ✅
   * - 添加对预处理指令 ``elifdef`` 和 ``elifndef`` 的支持
     - `P2334R1 <https://wg21.link/p2334r1>`__
     - ✅
   * - 诊断文本的字符编码
     - `P2246R1 <https://wg21.link/p2246r1>`__
     - ✅
   * - 扩展 init-statement 以允许别名声明
     - `P2360R0 <https://wg21.link/p2360r0>`__
     - ✅
   * - 更改 lambda 尾随返回类型的作用域
     - `P2036R3 <https://wg21.link/p2036r3>`__
     - ✅
   * - 多维下标运算符
     - `P2128R6 <https://wg21.link/p2128r6>`__
     - ✅
   * - 字符集和编码
     - `P2314R4 <https://wg21.link/p2314r4>`__
     - ✅
   * - ``auto(x)`` 和 ``auto {x}``
     - `P0849R8 <https://wg21.link/p0849r8>`__
     - ✅
   * - C++20 核心论文缺失的特性测试宏
     - `P2493R0 <https://wg21.link/p2493r0>`__
     - ✅
   * - Lambda 表达式上的属性
     - `P2173R1 <https://wg21.link/p2173r1>`__
     - ✅
   * - 对 ``#warning`` 的支持
     - `P2437R1 <https://wg21.link/p2437r1>`__
     - ✅
   * - 移除不可编码的宽字符字面量和多字符宽字符字面量
     - `P2362R3 <https://wg21.link/p2362r3>`__
     - ✅
   * - 复合语句末尾的标签（C 兼容性）
     - `P2324R2 <https://wg21.link/p2324r2>`__
     - ✅
   * - 分隔的转义序列
     - `P2290R3 <https://wg21.link/p2290r3>`__
     - ✅
   * - 放宽某些 ``constexpr`` 限制
     - `P2448R2 <https://wg21.link/p2448r2>`__
     - ❌
   * - 更简单的隐式移动
     - `P2266R3 <https://wg21.link/p2266r3>`__
     - ✅
   * - 命名的通用字符转义
     - `P2071R2 <https://wg21.link/p2071r2>`__
     - ✅
   * - ``static operator()``
     - `P1169R4 <https://wg21.link/p1169r4>`__
     - ✅
   * - ``static operator[]``
     - `P2589R1 <https://wg21.link/p2589r1>`__
     - ✅
   * - 扩展浮点类型和标准名称
     - `P1467R9 <https://wg21.link/p1467r9>`__
     - ✅
   * - 可移植的假设 ``[[assume]]``
     - `P1774R8 <https://wg21.link/p1774r8>`__
     - ✅
   * - 支持 UTF-8 作为可移植源文件编码
     - `P2295R6 <https://wg21.link/p2295r6>`__
     - ✅
   * - DR： ``char8_t`` 兼容性和可移植性修复
     - `P2513R4 <https://wg21.link/p2513r4>`__
     - ✅
   * - DR：取消弃用 ``volatile`` 位运算复合赋值操作
     - `P2327R1 <https://wg21.link/p2327r1>`__
     - ❌
   * - DR：放宽 ``wchar_t`` 要求以匹配现有实践
     - `P2460R2 <https://wg21.link/p2460r2>`__
     - N/A
   * - DR：在常量表达式中使用未知指针和引用
     - `P2280R4 <https://wg21.link/p2280r4>`__
     - ❌
   * - DR：你正在寻找的相等运算符
     - `P2468R2 <https://wg21.link/p2468r2>`__
     - ❌
   * - 允许 ``constexpr`` 函数中的 ``static constexpr`` 变量
     - `P2647R1 <https://wg21.link/p2647r1>`__
     - ✅
   * - 延长基于范围的 for 循环初始化器中临时对象的生命周期
     - | `P2644R1 <https://wg21.link/p2644r1>`__
       | `P2718R0 <https://wg21.link/p2718r0>`__
     - ✅
   * - DR： ``consteval`` 需要向上传播
     - `P2564R3 <https://wg21.link/p2564r3>`__
     - ✅

.. _cuda-c-standard-library:

5.3.6. CUDA C++ 标准库
----------------------

CUDA 提供了 C++ 标准库 (STL) 的实现，称为 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/standard_api.html>`__。该库具有以下优势：

- 功能在主机和设备上都可用。

- 与 CUDA Toolkit 支持的所有 `Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id59>`__ 和 `Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#id2>`__ 平台兼容。

- 与 CUDA Toolkit 最近两个主要版本支持的所有 `GPU 架构 <https://developer.nvidia.com/cuda-gpus>`__ 兼容。

- 与当前和以前主要版本的所有 `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`__ 兼容。

- 提供最新标准版本（包括 C++20、C++23 和 C++26）中可用的 C++ 标准库特性的 C++17 后向移植。

- 支持扩展数据类型，如 128 位整数（ ``__int128`` ）、半精度浮点（ ``__half`` ）、Bfloat16（ ``__nv_bfloat16`` ）和四精度浮点（ ``__float128`` ）。

- 针对设备代码进行了高度优化。

此外， ``libcu++`` 还提供了 C++ 标准库中不可用的 `扩展功能 <https://nvidia.github.io/cccl/libcudacxx/extended_api.html>`__，以提高生产力和应用程序性能。这些功能包括数学函数、内存操作、同步原语、容器扩展、CUDA 内建函数的高级抽象、C++ PTX 包装器等。

``libcu++`` 作为 `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`__ 的一部分提供，也是开源 `CCCL <https://nvidia.github.io/cccl/>`__ 仓库的一部分。

.. _c-standard-library-functions:

5.3.7. C 标准库函数
-------------------

.. _clock-and-clock64:

5.3.7.1. ``clock()`` 和 ``clock64()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ clock_t   clock();
   __device__          long long clock64();

在设备代码中执行时，返回每个多处理器的计数器值，该计数器每个时钟周期递增一次。在核函数开始和结束时采样此计数器，减去两个值，核函数中的并为每个线程记录结果，可以估算设备执行该线程所花费的时钟周期数。但是，此值并不代表设备执行该线程指令所花费的实际时钟周期数。前者大于后者，因为线程是时间片轮转的。

.. hint::

   在 ``<cuda/std/ctime>`` 头文件中提供了相应的 `CUDA C++ 函数 <https://en.cppreference.com/w/cpp/chrono/c/clock.html>`__ ``cuda::std::clock()`` 。

   在 ``<cuda/std/chrono>`` `头文件 <https://nvidia.github.io/cccl/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time>`__ 中也提供了可移植的 `C++ <https://en.cppreference.com/w/cpp/header/chrono>`__ ``<chrono>`` 实现，用于类似目的。

.. _printf:

5.3.7.2. ``printf()``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   int printf(const char* format[, arg, ...]);

该函数将核函数中的格式化输出打印到主机端输出流。

核函数中的 ``printf()`` 函数行为类似于标准 C 库的 ``printf()`` 函数。用户应参考其主机系统的手册页面以获取 ``printf()`` 行为的完整描述。本质上，作为 ``format`` 传入的字符串被输出到主机上的流。

``printf()`` 命令像任何其他设备端函数一样执行：每个线程执行，并在调用线程的上下文中执行。在多线程核函数中，对 ``printf()`` 的简单调用将由每个线程使用该线程指定的数据执行。因此，主机流中会出现多个版本的输出字符串，每个对应于遇到 ``printf()`` 的线程。

与返回打印字符数的 C 标准 ``printf()`` 不同，CUDA 的 ``printf()`` 返回解析的参数数量。如果格式字符串后没有参数，则返回 0。如果格式字符串为 ``NULL`` ，则返回 -1。如果发生内部错误，则返回 -2。

在内部， ``printf()`` 使用共享数据结构，因此调用 ``printf()`` 可能会改变线程的执行顺序。特别是，调用 ``printf()`` 的线程可能比不调用 ``printf()`` 的线程执行路径更长，该路径的长度取决于 ``printf()`` 的参数。但是，请注意，CUDA 不保证线程执行顺序，除非在显式的 ``__syncthreads()`` 屏障处。因此，无法判断执行顺序是否被 ``printf()`` 或硬件中的其他调度行为修改。

**格式说明符**

与标准 ``printf()`` 一样，格式说明符采用以下形式： ``%[flags][width][.precision][size]type``

支持以下字段。有关所有行为的完整描述，请参阅广泛可用的文档。

- 标志： ``#`` 、 ``' '`` 、 ``0`` 、 ``+`` 、 ``-``
- 宽度： ``*`` 、 ``0-9``
- 精度： ``0-9``
- 大小： ``h`` 、 ``l`` 、 ``ll``
- 类型： ``%cdiouxXpeEfgGaAs``

**限制**

``printf()`` 输出的最终格式化在主机系统上进行。这意味着格式字符串必须被主机系统的编译器和 C 库理解。虽然已尽力确保 CUDA 的 ``printf()`` 函数支持的格式说明符是最常见主机编译器支持的通用子集，但确切的行为将取决于主机操作系统。

``printf()`` 接受所有有效的标志和类型组合。这是因为它无法确定在最终输出格式化的主机系统上什么有效和什么无效。因此，如果程序发出包含无效组合的格式字符串，输出可能是未定义的。

``printf()`` 函数最多可以接受 32 个参数，此外还有格式字符串。任何额外的参数将被忽略，格式说明符将按原样输出。

由于 Windows 平台（32 位）和 Linux 平台（64 位）上 ``long`` 类型的不同大小，在 Linux 机器上编译然后在 Windows 机器上运行的核函数将产生包含 ``%ld`` 的所有格式字符串的损坏输出。为确保安全，建议编译和执行平台匹配。

**主机端缓冲区**

``printf()`` 的输出缓冲区在核函数启动前设置为固定大小。缓冲区是循环的，因此如果在核函数执行期间产生的输出多于缓冲区可以容纳的内容，则会覆盖较旧的输出。仅在执行以下操作之一时刷新缓冲区：

- 通过 ``<<< >>>`` 或 ``cuLaunchKernel()`` 启动核函数：在启动开始时，以及如果 ``CUDA_LAUNCH_BLOCKING`` 环境变量设置为 1，则在启动结束时也会刷新，
- 通过 ``cudaDeviceSynchronize()`` 、 ``cuCtxSynchronize()`` 、 ``cudaStreamSynchronize()`` 、 ``cuStreamSynchronize()`` 、 ``cudaEventSynchronize()`` 或 ``cuEventSynchronize()`` 进行同步，
- 通过任何阻塞版本的 ``cudaMemcpy*()`` 或 ``cuMemcpy*()`` 进行内存复制，
- 通过 ``cuModuleLoad()`` 或 ``cuModuleUnload()`` 加载/卸载模块，
- 通过 ``cudaDeviceReset()`` 或 ``cuCtxDestroy()`` 销毁上下文。
- 在执行通过 ``cudaLaunchHostFunc()`` 或 ``cuLaunchHostFunc()`` 添加的流回调之前。

请注意，程序退出时缓冲区不会自动刷新。

以下 API 函数设置和检索用于将 ``printf()`` 参数和内部元数据传输到主机的缓冲区大小。默认大小为 1 MB。

- ``cudaDeviceGetLimit(size_t* size, cudaLimitPrintfFifoSize)``
- ``cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)``

**示例**

以下代码示例：

.. code-block:: c++

   #include <stdio.h>

   __global__ void helloCUDA(float value) {
       printf("Hello thread %d, value=%f\n", threadIdx.x, value);
   }

   int main() {
       helloCUDA<<<1, 5>>>(1.2345f);
       cudaDeviceSynchronize();
       return 0;
   }

将输出：

.. code-block:: text

   Hello thread 2, value=1.2345
   Hello thread 1, value=1.2345
   Hello thread 4, value=1.2345
   Hello thread 0, value=1.2345
   Hello thread 3, value=1.2345

注意每个线程都遇到 ``printf()`` 命令。因此，输出行数与网格中的线程数相同。

.. _memcpy-and-memset:

5.3.7.3. ``memcpy()`` 和 ``memset()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ void* memcpy(void* dest, const void* src, size_t size);

该函数从 ``src`` 指向的内存位置复制 ``size`` 字节到 ``dest`` 指向的内存位置。

.. code-block:: c++

   __host__ __device__ void* memset(void* ptr, int value, size_t size);

该函数将 ``ptr`` 指向的内存块的 ``size`` 字节设置为 ``value`` ，解释为 ``unsigned char`` 。

.. hint::

   建议使用 ``<cuda/std/cstring>`` `头文件 <https://nvidia.github.io/cccl/libcudacxx/standard_api/c_library/cstring.html#libcudacxx-standard-api-cstring>`__ 中提供的 ``cuda::std::memcpy()`` 和 ``cuda::std::memset()`` 函数作为 ``memcpy`` 和 ``memset`` 的更安全版本。

.. _malloc-and-free:

5.3.7.4. ``malloc()`` 和 ``free()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ void* malloc(size_t size);
   // or cuda::std::malloc(), cuda::std::calloc() in the <cuda/std/cstdlib> header

函数 ``malloc()`` （设备端）、 ``cuda::std::malloc()`` 和 ``cuda::std::calloc()`` 从设备堆中分配至少 ``size`` 字节，并返回指向已分配内存的指针。
如果内存不足以满足请求，则返回 ``NULL`` 。返回的指针保证对齐到 16 字节边界。

.. code-block:: c++

   __device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
   // or cuda::std::aligned_alloc() in the <cuda/std/cstdlib> header

函数 ``__nv_aligned_device_malloc()`` 和 `C++ <https://en.cppreference.com/w/cpp/memory/c/aligned_alloc>`__ ``cuda::std::aligned_alloc()`` 从设备堆中分配至少 ``size`` 字节，并返回指向已分配内存的指针。如果内存不足以满足请求的大小或对齐，则返回 ``NULL`` 。已分配内存的地址是 ``align`` 的倍数。 ``align`` 必须是非零的 2 的幂。

.. code-block:: c++

   __host__ __device__ void free(void* ptr);
   // or cuda::std::free() in the <cuda/std/cstdlib> header

设备端函数 ``free()`` 和 ``cuda::std::free()`` 释放 ``ptr`` 指向的内存，该内存必须由先前对 ``malloc()`` 、 ``cuda::std::malloc()`` 、 ``cuda::std::calloc()`` 、 ``__nv_aligned_device_malloc()`` 或 ``cuda::std::aligned_alloc()`` 的调用返回。
如果 ``ptr`` 为 ``NULL`` ，则忽略对 ``free()`` 或 ``cuda::std::free()`` 的调用。
使用相同的 ``ptr`` 重复调用 ``free()`` 或 ``cuda::std::free()`` 具有未定义行为。

通过 ``malloc()`` 、 ``cuda::std::malloc()`` 、 ``cuda::std::calloc()`` 、 ``__nv_aligned_device_malloc()`` 或 ``cuda::std::aligned_alloc()`` 由给定 CUDA 线程分配的内存在 CUDA 上下文的生存期内保持分配状态，或直到通过调用 ``free()`` 或 ``cuda::std::free()`` 显式释放。
此内存可由其他 CUDA 线程使用，即使是来自后续核函数启动的线程。
任何 CUDA 线程都可以释放由另一个线程分配的内存；但是，应注意确保同一指针不被多次释放。

**堆内存 API**

设备堆内存的大小必须在任何在设备代码中分配或释放内存的操作之前指定，包括 ``new`` 和 ``delete`` 关键字。
如果未显式指定堆大小，则分配 8 MB 的默认堆。

以下 API 函数获取和设置堆大小：

- ``cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)``
- ``cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)``

授予的堆大小将至少为 ``size`` 字节。`cuCtxGetLimit() <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8>`__ 和 `cudaDeviceGetLimit() <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2>`__ 返回当前请求的堆大小。

堆的实际内存分配发生在模块加载到上下文时，无论是通过 CUDA 驱动程序 API 显式加载（参见 `模块 <../03-advanced/driver-api.html#driver-api-module>`__），还是通过 CUDA 运行时 API 隐式加载。如果内存分配失败，模块加载将生成 ``CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`` 错误。

堆大小在模块加载后无法更改，并且不会根据需要动态调整大小。

为设备堆保留的内存是通过主机端 CUDA API 调用（如 ``cudaMalloc()`` ）分配的内存之外的。

**与主机内存 API 的互操作性**

通过设备端函数 ``malloc()`` 、 ``cuda::std::malloc()`` 、 ``cuda::std::calloc()`` 、 ``__nv_aligned_device_malloc()`` 、 ``cuda::std::aligned_alloc()`` 或 ``new`` 关键字分配的内存不能与运行时或驱动程序 API 调用（如 ``cudaMalloc`` 、 ``cudaMemcpy`` 或 ``cudaMemset`` ）一起使用或释放。同样，通过主机运行时 API 分配的内存不能使用设备端函数 ``free()`` 、 ``cuda::std::free()`` 或 ``delete`` 关键字释放。

**每线程分配示例：**

.. code-block:: c++

   #include <stdlib.h>
   #include <stdio.h>

   __global__ void single_thread_allocation_kernel() {
       size_t size = 123;
       char*  ptr  = (char*) malloc(size);
       memset(ptr, 0, size);
       printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
       free(ptr);
   }

   int main() {
       // Set a heap size of 128 megabytes.
       // Note that this must be done before any kernel is launched.
       cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
       single_thread_allocation_kernel<<<1, 5>>>();
       cudaDeviceSynchronize();
       return 0;
   }

将输出：

.. code-block:: text

   Thread 0 got pointer: 0x20d5ffe20
   Thread 1 got pointer: 0x20d5ffec0
   Thread 2 got pointer: 0x20d5fff60
   Thread 3 got pointer: 0x20d5f97c0
   Thread 4 got pointer: 0x20d5f9720

注意每个线程如何遇到 ``malloc()`` 和 ``memset()`` 命令，因此接收并初始化自己的分配。

.. _alloca:

5.3.7.5. ``alloca()``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ void* alloca(size_t size);

``alloca()`` 函数在调用者的栈帧内分配 ``size`` 字节的内存。返回值是指向已分配内存的指针。
当从设备代码调用该函数时，内存起始地址按 16 字节对齐。
当调用 ``alloca()`` 的函数返回时，分配的内存会自动释放。

.. note::

   在 Windows 平台上，使用 ``alloca()`` 函数之前必须包含 ``<malloc.h>`` 头文件。调用 ``alloca()`` 可能导致栈溢出；用户需要相应地调整栈大小。

示例：

.. code-block:: c++

   __device__ void device_function(int num_items) {
       int4* ptr = (int4*) alloca(num_items * sizeof(int4));
       // use of ptr
       ...
   }  // ptr is freed on the return of device_function

.. _lambda-expressions:

5.3.8. Lambda 表达式
--------------------

编译器通过将 lambda 表达式或闭包类型（C++11）与最内层包围它的函数的执行空间相关联来确定其执行空间。如果没有封闭函数作用域，则执行空间指定为 ``__host__`` 。

执行空间也可以使用 :ref:`扩展 lambda 语法 <extended-lambdas>` 显式指定。

示例：

.. code-block:: c++

   auto global_lambda = [](){ return 0; }; // __host__

   void host_function() {
       auto lambda1 = [](){ return 1; };   // __host__
       [](){ return 3; };                  // __host__, closure type (body of a lambda expression)
   }

   __device__ void device_function() {
       auto lambda2 = [](){ return 2; };   // __device__
   }

   __global__ void kernel_function(void) {
       auto lambda3 = [](){ return 3; };   // __device__
   }

   __host__ __device__ void host_device_function() {
       auto lambda4 = [](){ return 4; };   // __host__ __device__
   }

   __tile__ void tile_function() {
       auto lambda5 = [](){ return 5; };   // __tile__
   }

   __tile__ __device__ void tile_device_function() {
       auto lambda6 = [](){ return 6; };   // __tile__ __device__
   }

   using function_ptr_t = int (*)();

   __device__ void device_function(float          value,
                                   function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}

在 `Compiler Explorer <https://godbolt.org/z/scv4vcczr>`__ 上查看示例。

.. _lambda-expressions-and-global-function-parameters:

5.3.8.1. Lambda 表达式和 ``__global__`` 函数参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

只有当 lambda 表达式或闭包类型的执行空间为 ``__device__`` 或 ``__host__ __device__`` 时，才能将其用作 ``__global__`` 函数的参数。全局或命名空间作用域的 lambda 表达式不能用作 ``__global__`` 函数的参数。

示例：

.. code-block:: c++

   template <typename T>
   __global__ void kernel(T input) {}

   __device__ void device_function() {
       // device kernel call requires separate compilation (-rdc=true flag)
       kernel<<<1, 1>>>([](){});
       kernel<<<1, 1>>>([] __device__() {});          // extended lambda
       kernel<<<1, 1>>>([] __host__ __device__() {}); // extended lambda
   }

   auto global_lambda = [] __host__ __device__() {};

   void host_function() {
       kernel<<<1, 1>>>([] __device__() {});          // CORRECT, extended lambda
       kernel<<<1, 1>>>([] __host__ __device__() {}); // CORRECT, extended lambda
   //  kernel<<<1, 1>>>([](){});                      // ERROR, closure type with host execution space
   //  kernel<<<1, 1>>>(global_lambda);               // ERROR, extended lambda, but at global scope
   }

在 `Compiler Explorer <https://godbolt.org/z/ajrsn5z5Y>`__ 上查看示例。

.. _extended-lambdas:

5.3.8.2. 扩展 Lambda
^^^^^^^^^^^^^^^^^^^^

``nvcc`` 标志 ``--extended-lambda`` 允许在 lambda 表达式中显式注释执行空间。这些注释应出现在 lambda 引导符之后和可选的 lambda 声明符之前。

当指定 ``--extended-lambda`` 标志时， ``nvcc`` 定义宏 ``__CUDACC_EXTENDED_LAMBDA__`` 。

- *扩展 lambda* 定义在 ``__host__`` 或 ``__host__ __device__`` 函数的直接或嵌套块作用域内。

- *扩展设备 lambda* 是用 ``__device__`` 关键字注释的 lambda 表达式。

- *扩展主机设备 lambda* 是用 ``__host__ __device__`` 关键字注释的 lambda 表达式。

与标准 lambda 表达式不同，扩展 lambda 可以用作 ``__global__`` 函数中的类型参数。

示例：

.. code-block:: c++

   void host_function() {
       auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
       auto lambda2 = [] __device__ {};           // extended lambda
       auto lambda3 = [] __host__ __device__ {};  // extended lambda
       auto lambda4 = [] __host__ {};             // NOT an extended lambda
   }

   __host__ __device__ void host_device_function() {
       auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
       auto lambda2 = [] __device__ {};           // extended lambda
       auto lambda3 = [] __host__ __device__ {};  // extended lambda
       auto lambda4 = [] __host__ {};             // NOT an extended lambda
   }

   __device__ void device_function() {
       // none of the lambdas within this function are extended lambdas,
       // because the enclosing function is not a __host__ or __host__ __device__  function.
       auto lambda1 = [] {};
       auto lambda2 = [] __device__ {};
       auto lambda3 = [] __host__ __device__ {};
       auto lambda4 = [] __host__ {};
   }

   auto global_lambda = [] __host__ __device__ { }; // NOT an extended lambda because it is not defined
                                                    // within a __host__ or __host__ __device__ function

.. _extended-lambda-type-traits:

5.3.8.3. 扩展 Lambda 类型特性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

编译器提供类型特性来在编译时检测扩展 lambda 的闭包类型。

.. code-block:: c++

   bool __nv_is_extended_device_lambda_closure_type(type);

如果 ``type`` 是为扩展 ``__device__`` lambda 创建的闭包类，则函数返回 ``true`` ，否则返回 ``false`` 。

.. code-block:: c++

   bool __nv_is_extended_device_lambda_with_preserved_return_type(type);

如果 ``type`` 是为扩展 ``__device__`` lambda 创建的闭包类，并且 lambda 使用尾随返回类型定义，则函数返回 ``true`` ，否则返回 ``false`` 。如果尾随返回类型引用任何 lambda 参数名称，则返回类型不会被保留。

.. code-block:: c++

   bool __nv_is_extended_host_device_lambda_closure_type(type);

如果 ``type`` 是为扩展 ``__host__ __device__`` lambda 创建的闭包类，则函数返回 ``true`` ，否则返回 ``false`` 。

lambda 类型特性可在所有编译模式下使用，无论是否启用了 lambda 或扩展 lambda。如果扩展 lambda 模式未激活，这些特性将始终返回 ``false`` 。

示例：

.. code-block:: c++

   auto lambda0 = [] __host__ __device__ { };

   void host_function() {
       auto lambda1 = [] { };
       auto lambda2 = [] __device__ { };
       auto lambda3 = [] __host__ __device__ { };
       auto lambda4 = [] __device__ () -> double { return 3.14; }
       auto lambda5 = [] __device__ (int x) -> decltype(&x) { return 0; }

       using lambda0_t = decltype(lambda0);
       using lambda1_t = decltype(lambda1);
       using lambda2_t = decltype(lambda2);
       using lambda3_t = decltype(lambda3);
       using lambda4_t = decltype(lambda4);
       using lambda5_t = decltype(lambda5);

       // 'lambda0' is not an extended lambda because it is defined outside function scope
       static_assert(!__nv_is_extended_device_lambda_closure_type(lambda0_t));
       static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda0_t));
       static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda0_t));

       // 'lambda1' is not an extended lambda because it has no execution space annotations
       static_assert(!__nv_is_extended_device_lambda_closure_type(lambda1_t));
       static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda1_t));
       static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda1_t));

       // 'lambda2' is an extended device-only lambda
       static_assert(__nv_is_extended_device_lambda_closure_type(lambda2_t));
       static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda2_t));
       static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda2_t));

       // 'lambda3' is an extended host-device lambda
       static_assert(!__nv_is_extended_device_lambda_closure_type(lambda3_t));
       static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda3_t));
       static_assert(__nv_is_extended_host_device_lambda_closure_type(lambda3_t));

       // 'lambda4' is an extended device-only lambda with preserved return type
       static_assert(__nv_is_extended_device_lambda_closure_type(lambda4_t));
       static_assert(__nv_is_extended_device_lambda_with_preserved_return_type(lambda4_t));
       static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda4_t));

       // 'lambda5' is not an extended device-only lambda with preserved return type
       // because it references the operator()'s parameter types in the trailing return type.
       static_assert(__nv_is_extended_device_lambda_closure_type(lambda5_t));
       static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda5_t));
       static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda5_t));
   }

.. _extended-lambda-restrictions:

5.3.8.4. 扩展 Lambda 限制
^^^^^^^^^^^^^^^^^^^^^^^^^

在调用主机编译器之前，CUDA 编译器将扩展 lambda 表达式替换为在命名空间作用域中定义的占位符类型的实例。占位符类型的模板参数需要获取包围原始扩展 lambda 表达式的函数的地址。这对于正确执行任何模板参数涉及扩展 lambda 闭包类型的 ``__global__`` 函数模板是必要的。封闭函数按如下方式计算。

根据定义，扩展 lambda 存在于 ``__host__`` 或 ``__host__ __device__`` 函数的直接或嵌套块作用域内。

- 如果该函数不是 lambda 表达式的 ``operator()``，则它被视为扩展 lambda 的封闭函数。

- 否则，扩展 lambda 定义在一个或多个封闭 lambda 表达式的 ``operator()`` 的直接或嵌套块作用域内。

  - 如果最外层 lambda 表达式定义在函数 ``F`` 的直接或嵌套块作用域内，则 ``F`` 是计算出的封闭函数。

  - 否则，封闭函数不存在。

示例：

.. code-block:: c++

   void host_function() {
       auto lambda1 = [] __device__ { }; // enclosing function for lambda1 is "host_function()"
       auto lambda2 = [] {
           auto lambda3 = [] {
               auto lambda4 = [] __host__ __device__ { }; // enclosing function for lambda4 is "host_function"
           };
       };
   }

   auto global_lambda = [] {
       auto lambda5 = [] __host__ __device__ { }; // enclosing function for lambda5 does not exist
   };

**扩展 Lambda 限制**

1. 扩展 lambda 不能定义在另一个扩展 lambda 表达式内部。示例：

   .. code-block:: c++

      void host_function() {
          auto lambda1 = [] __host__ __device__  {
               // ERROR, extended lambda defined within another extended lambda
              auto lambda2 = [] __host__ __device__ { };
          };
      }

2. 扩展 lambda 不能定义在泛型 lambda 表达式内部。示例：

   .. code-block:: c++

      void host_function() {
          auto lambda1 = [] (auto) {
               // ERROR, extended lambda defined within a generic lambda
              auto lambda2 = [] __host__ __device__ { };
          };
      }

3. 如果扩展 lambda 定义在一个或多个嵌套 lambda 表达式的直接或嵌套块作用域内，则最外层 lambda 表达式必须定义在函数的直接或嵌套块作用域内。示例：

   .. code-block:: c++

      auto lambda1 = []  {
          // ERROR, outer enclosing lambda is not defined within a non-lambda-operator() function
          auto lambda2 = [] __host__ __device__ { };
      };

4. 扩展 lambda 的封闭函数必须是具名的，并且其地址必须是可访问的。如果封闭函数是类成员，则必须满足以下条件：

   - 包围成员函数的所有类必须具有名称。

   - 成员函数在其父类中不能具有 private 或 protected 访问权限。

   - 所有封闭类在其各自的父类中不能具有 private 或 protected 访问权限。

   示例：

   .. code-block:: c++

      void host_function() {
          auto lambda1 = [] __device__ { return 0; }; // OK
          {
              auto lambda2 = [] __device__          { return 0; }; // OK
              auto lambda3 = [] __device__ __host__ { return 0; }; // OK
          }
      }

      struct MyStruct1 {
          MyStruct1() {
              auto lambda4 = [] __device__ { return 0; }; // ERROR, address of the enclosing function is not accessible
          }
      };

      class MyStruct2 {
          void foo() {
              auto temp1 = [] __device__ { return 10; }; // ERROR, enclosing function has private access in parent class
          }

          struct MyStruct3 {
              void foo() {
                  auto temp1 = [] __device__ { return 10; };  // ERROR, enclosing class MyStruct3 has private access in its parent class
              }
          };
      };

5. 在定义扩展 lambda 的位置，必须能够无歧义地获取封闭例程的地址。但是，这并不总是可行的，例如，当别名声明遮蔽了同名的模板类型参数时。示例：

   .. code-block:: c++

      template <typename T>
      struct A {
          using Bar = void;
          void test();
      };

      template<>
      struct A<void> { };

      template <typename Bar>
      void A<Bar>::test() {
          // In code sent to host compiler, nvcc will inject an address expression here, of the form:
          //   (void (A< Bar> ::*)(void))(&A::test))
          //  However, the class typedef 'Bar' (to void) shadows the template argument 'Bar',
          //  causing the address expression in A<int>::test to actually refer to:
          //    (void (A< void> ::*)(void))(&A::test))
          //  which doesn't take the address of the enclosing routine 'A<int>::test' correctly.
          auto lambda1 = [] __host__ __device__ { return 4; };
      }

      int main() {
          A<int> var;
          var.test();
      }

6. 扩展 lambda 不能定义在函数局部的类中。示例：

   .. code-block:: c++

      void host_function() {
          struct MyStruct {
              void bar() {
                  // ERROR, bar() is member of a class that is local to a function
                  auto lambda2 = [] __host__ __device__ { return 0; };
              }
          };
      }

7. 扩展 lambda 的封闭函数不能具有推导的返回类型。示例：

   .. code-block:: c++

      auto host_function() {
          // ERROR, the return type of host_function() is deduced
          auto lambda3 = [] __host__ __device__ { return 0; };
      }

8. 主机 - 设备扩展 lambda 不能是泛型 lambda，即具有 ``auto`` 参数类型的 lambda。示例：

   .. code-block:: c++

      void host_function() {
          // ERROR, __host__ __device__ extended lambdas cannot be a generic lambda
          auto lambda1 = [] __host__ __device__ (auto i) { return i; };

          // ERROR, a host-device extended lambda cannot be a generic lambda
          auto lambda2 = [] __host__ __device__ (auto... i) {
              return sizeof...(i);
          };
      }

9. 如果封闭函数是函数或成员模板的实例化，或者如果该函数是类模板的成员，则模板必须满足以下约束：

   - 模板最多只能有一个可变参数，并且它必须列在模板参数列表的最后。

   - 模板参数必须具名。

   - 模板实例化参数类型不能涉及函数局部的类型（扩展 lambda 的闭包类型除外），或者是 ``private`` 或 ``protected`` 类成员。

   示例 1：

   .. code-block:: c++

      template <template <typename...> class T,
                typename... P1,
                typename... P2>
      void bar1(const T<P1...>, const T<P2...>) {
          // ERROR, enclosing function has multiple parameter packs
          auto lambda = [] __device__ { return 10; };
      }

      template <template <typename...> class T,
                typename... P1,
                typename    T2>
      void bar2(const T<P1...>, T2) {
          // ERROR, for enclosing function, the parameter pack is not last in the template parameter list
          auto lambda = [] __device__ { return 10; };
      }

      template <typename T, T>
      void bar3() {
          // ERROR, for enclosing function, the second template parameter is not named
          auto lambda = [] __device__ { return 10; };
      }

   示例 2：

   .. code-block:: c++

      template <typename T>
      void bar4() {
          auto lambda1 = [] __device__ { return 10; };
      }

      class MyStruct {
          struct MyNestedStruct {};

          friend int main();
      };

      int main() {
          struct MyLocalStruct {};
          // ERROR, enclosing function for device lambda in bar4() is instantiated with a type local to main
          bar4<MyLocalStruct>();

          // ERROR, enclosing function for device lambda in bar4 is instantiated with a type
          //        that is a private member of a class
          bar4<MyStruct::MyNestedStruct>();
      }

10. 使用 Microsoft Visual Studio 主机编译器时，封闭函数必须具有外部链接。此限制存在的原因是主机编译器不支持将非外部链接函数的地址用作模板参数。CUDA 编译器转换需要这些地址来支持扩展 lambda。

11. 使用 Microsoft Visual Studio 主机编译器时，扩展 lambda 不应定义在 ``if constexpr`` 块的主体内。

12. 扩展 lambda 对捕获的变量有以下限制：

    - 变量可能先按值传递给发送到主机编译器的代码中的一系列辅助函数，然后才用于直接初始化表示扩展 lambda 闭包类型的类类型的字段。但是，C++ 标准规定捕获的变量应直接用于初始化闭包类型的字段。

    - 变量只能按值捕获。

    - 如果数组维数大于 7，则不能捕获数组类型的变量。

    - 对于数组类型的变量，闭包类型的数组字段首先进行默认初始化，然后在发送到主机编译器的代码中，每个数组元素从捕获的数组变量的相应元素进行复制赋值。因此，数组元素类型在主机代码中必须同时是默认可构造和可复制赋值的。

    - 不能捕获作为可变参数包元素的函数参数。

    - 捕获的变量类型不能是函数局部的类型（扩展 lambda 闭包类型除外），也不能是 ``private`` 或 ``protected`` 类成员。

    - 主机 - 设备扩展 lambda 不支持初始化捕获（init-capture）。但是，设备扩展 lambda 支持，除非初始化器是数组或 ``std::initializer_list`` 类型。

    - 扩展 lambda 的函数调用运算符不是 ``constexpr``。扩展 lambda 的闭包类型不是字面量类型。声明扩展 lambda 时不能使用 ``constexpr`` 和 ``consteval`` 说明符。

    - 除非变量已在 ``if-constexpr`` 块外部被隐式捕获，或出现在扩展 lambda 的显式捕获列表中，否则不能在词法嵌套于扩展 lambda 内部的 ``if-constexpr`` 块内隐式捕获变量。

    示例：

    .. code-block:: c++

       void host_function() {
           // CORRECT, an init-capture is allowed for an extended device-only lambda
           auto lambda1 = [x = 1] __device__ () { return x; };

           // ERROR, an init-capture is not allowed for an extended host-device lambda
           auto lambda2 = [x = 1] __host__ __device__ () { return x; };

           int a = 1;
           // ERROR, an extended __device__ lambda cannot capture variables by reference
           auto lambda3 = [&a] __device__ () { return a; };

           // ERROR, by-reference capture is not allowed for an extended device-only lambda
           auto lambda4 = [&x = a] __device__ () { return x; };

           struct MyStruct {};
           MyStruct s1;
           // ERROR, a type local to a function cannot be used in the type of a captured variable
           auto lambda6 = [s1] __device__ () { };

           // ERROR, an init-capture cannot be of type std::initializer_list
           auto lambda7 = [x = {11}] __device__ () { };

           std::initializer_list<int> b = {11,22,33};
           // ERROR, an init-capture cannot be of type std::initializer_list
           auto lambda8 = [x = b] __device__ () { };

           int  var     = 4;
           auto lambda9 = [=] __device__ {
               int result = 0;
               if constexpr(false) {
                   //ERROR, An extended device-only lambda cannot first-capture 'var' in if-constexpr context
                   result += var;
               }
               return result;
           };

           auto lambda10 = [var] __device__ {
               int result = 0;
               if constexpr(false) {
                   // CORRECT, 'var' already listed in explicit capture list for the extended lambda
                   result += var;
               }
               return result;
           };

           auto lambda11 = [=] __device__ {
               int result = var;
               if constexpr(false) {
                   // CORRECT, 'var' already implicit captured outside the 'if-constexpr' block
                   result += var;
               }
               return result;
           };
       }

13. 解析函数时，CUDA 编译器为函数中的每个扩展 lambda 分配一个计数器值。此计数器值用于传递给主机编译器的替换命名类型。因此，函数中扩展 lambda 的存在与否不应依赖于 ``__CUDA_ARCH__`` 的特定值，也不应依赖于 ``__CUDA_ARCH__`` 未定义。示例：

    .. code-block:: c++

       template <typename T>
       __global__ void kernel(T in) { in(); }

       __host__ __device__ void host_device_function() {
           // ERROR, the number and relative declaration order of
           //        extended lambdas depend on __CUDA_ARCH__
       #if defined(__CUDA_ARCH__)
           auto lambda1 = [] __device__ { return 0; };
           auto lambda2 = [] __host__ __device__ { return 10; };
       #endif
           auto lambda3 = [] __device__ { return 4; };
           kernel<<<1, 1>>>(lambda3);
       }

14. 如上所述，CUDA 编译器将定义在主机函数中的设备扩展 lambda 替换为在命名空间作用域中定义的占位符类型。除非特性 ``__nv_is_extended_device_lambda_with_preserved_return_type()`` 对扩展 lambda 的闭包类型返回 ``true``，否则占位符类型不定义与原始 lambda 声明等效的 ``operator()`` 函数。因此，尝试确定此类 lambda 的 ``operator()`` 函数的返回类型或参数类型在主机代码中可能无法正常工作，因为主机编译器处理的代码与 CUDA 编译器处理的输入代码在语义上不同。但是，在设备代码中内省 ``operator()`` 函数的返回类型或参数类型是可以接受的。请注意，此限制不适用于特性 ``__nv_is_extended_device_lambda_with_preserved_return_type()`` 返回 ``true`` 的主机或设备扩展 lambda。示例：

    .. code-block:: c++

       #include <cuda/std/type_traits>

       const char& getRef(const char* p) { return *p; }

       void foo() {
           auto lambda1 = [] __device__ { return "10"; };

           // ERROR, attempt to extract the return type of a device lambda in host code
           cuda::std::result_of<decltype(lambda1)()>::type xx1 = "abc";

           auto lambda2 = [] __host__ __device__ { return "10"; };

           // CORRECT, lambda2 represents a host-device extended lambda
           cuda::std::result_of<decltype(lambda2)()>::type xx2 = "abc";

           auto lambda3 = [] __device__ () -> const char* { return "10"; };

           // CORRECT, lambda3 represents a device extended lambda with preserved return type
           cuda::std::result_of<decltype(lambda3)()>::type xx2 = "abc";
           static_assert(cuda::std::is_same_v<cuda::std::result_of<decltype(lambda3)()>::type, const char*>);

           auto lambda4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
           // lambda4's return type is not preserved because it references the operator()'s
           // parameter types in the trailing return type.
           static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(decltype(lambda4)));
       }

15. 对于仅设备扩展 lambda：

    - 对 ``operator()`` 参数类型的内省仅在设备代码中受支持。

    - 对 ``operator()`` 返回类型的内省仅在设备代码中受支持，除非特性函数 ``__nv_is_extended_device_lambda_with_preserved_return_type()`` 返回 ``true``。

16. 如果扩展 lambda 作为参数从主机代码传递到设备代码（例如传递给 ``__global__`` 函数），则无论 ``__CUDA_ARCH__`` 宏是否定义及其值如何，lambda 主体中任何捕获变量的表达式都必须保持不变。此限制产生的原因是，lambda 的闭包类布局取决于编译器处理 lambda 表达式时遇到捕获变量的顺序。如果闭包类布局在设备编译和主机编译之间不同，程序可能无法正确执行。示例：

    .. code-block:: c++

       __device__ int result;

       template <typename T>
       __global__ void kernel(T in) { result = in(); }

       void foo(void) {
           int x1 = 1;
           // ERROR, "x1" is only captured when __CUDA_ARCH__ is defined.
           auto lambda1 = [=] __host__ __device__ {
       #ifdef __CUDA_ARCH__
               return x1 + 1;
       #else
               return 10;
       #endif
           };
           kernel<<<1, 1>>>(lambda1);
       }

17. 如前所述，CUDA 编译器在发送到主机编译器的代码中将仅设备扩展 lambda 表达式替换为占位符类型实例。占位符类型在主机代码中不定义到函数指针的转换运算符；但是，该转换运算符在设备代码中提供。请注意，此限制不适用于主机 - 设备扩展 lambda。示例：

    .. code-block:: c++

       template <typename T>
       __global__ void kernel(T in) {
           int (*fp)(double) = in;
           fp(0); // CORRECT, conversion in device code is supported
           auto lambda1 = [](double) { return 1; };
       }

       void foo() {
           auto lambda_device      = [] __device__ (double) { return 1; };
           auto lambda_host_device = [] __host__ __device__ (double) { return 1; };
           kernel<<<1, 1>>>(lambda_device);
           kernel<<<1, 1>>>(lambda_host_device);

           // CORRECT, conversion for a __host__ __device__ lambda is supported in host code
           int (*fp1)(double) = lambda_host_device;

           // ERROR, conversion for a device lambda is not supported in host code
           int (*fp2)(double) = lambda_device;
       }

18. 如前所述，CUDA 编译器在发送到主机编译器的代码中将仅设备或主机 - 设备扩展 lambda 表达式替换为占位符类型实例。此占位符类型可能定义 C++ 特殊成员函数，例如构造函数和析构函数。因此，某些标准 C++ 类型特性对于扩展 lambda 的闭包类型，在 CUDA 前端编译器中可能产生与主机编译器中不同的结果。受影响的类型特性有：``std::is_trivially_copyable``、``std::is_trivially_constructible``、``std::is_trivially_copy_constructible``、``std::is_trivially_move_constructible``、``std::is_trivially_destructible``。必须小心确保这些特性的结果不用于 ``__global__``、``__device__``、``__constant__`` 或 ``__managed__`` 函数或变量模板的实例化。示例：

    .. code-block:: c++

       #include <cstdio>
       #include <type_traits>

       template <bool b>
       void __global__ kernel() { printf("hi"); }

       template <typename T>
       void kernel_launch() {
           // ERROR, this kernel launch may fail, because CUDA frontend compiler and host compiler
           //        may disagree on the result of std::is_trivially_copyable_v trait on the
           //        closure type of the extended lambda
           kernel<std::is_trivially_copyable_v<T>><<<1,1>>>();
           cudaDeviceSynchronize();
       }

       int main() {
           int  x       = 0;
           auto lambda1 = [=] __host__ __device__ () { return x; };
           kernel_launch<decltype(lambda1)>();
       }

CUDA 编译器将为 ``1-12`` 中描述的部分情况生成编译器诊断信息；对于情况 ``13-17`` 不会生成诊断信息，但主机编译器可能无法编译生成的代码。

.. _host-device-lambda-optimization:

5.3.8.5. 主机 - 设备 Lambda 优化注意事项
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与仅设备 lambda 不同，主机 - 设备 lambda 可以从主机代码调用。如前所述，CUDA 编译器将定义在主机代码中的扩展 lambda 表达式替换为命名占位符类型的实例。扩展主机 - 设备 lambda 的占位符类型通过间接函数调用调用原始 lambda 的 ``operator()``。如果扩展 lambda 模式未激活，这些特性将始终返回 ``false``。

间接函数调用的存在可能导致主机编译器对扩展主机 - 设备 lambda 的优化不如对隐式或显式仅 ``__host__`` 的 lambda 的优化。在后一种情况下，主机编译器可以很容易地将 lambda 主体内联到调用上下文中。但是，当遇到扩展主机 - 设备 lambda 时，主机编译器可能无法轻易内联原始 lambda 主体。

.. _this-capture-by-value:

5.3.8.6. ``*this`` 按值捕获
^^^^^^^^^^^^^^^^^^^^^^^^^^^

根据 C++11/C++14 规则，当 lambda 定义在非 ``static`` 类成员函数内，且 lambda 主体引用了类成员变量时，必须按值捕获类的 ``this`` 指针，而不是被引用的成员变量。如果该 lambda 是定义在主机函数中并在 GPU 上执行的扩展仅设备或主机 - 设备 lambda，且 ``this`` 指针指向主机内存，则在 GPU 上访问被引用的成员变量将导致运行时错误。

示例：

.. code-block:: c++

   #include <cstdio>

   template <typename T>
   __global__ void foo(T in) { printf("value = %d\n", in()); }

   struct MyStruct {
       int var;

       __host__ __device__ MyStruct() : var(10) {};

       void run() {
           auto lambda1 = [=] __device__ {
               // reference to "var" causes the 'this' pointer (MyStruct*) to be captured by value
               return var + 1;
           };
           // Kernel launch fails at run time because 'this->var' is not accessible from the GPU
           foo<<<1, 1>>>(lambda1);
           cudaDeviceSynchronize();
       }
   };

   int main() {
       MyStruct s1;
       s1.run();
   }

C++17 通过引入新的 ``*this`` 捕获模式解决了这个问题。在此模式下，编译器复制 ``*this`` 所表示的对象，而不是按值捕获 ``this`` 指针。``*this`` 捕获模式在 `P0018R3 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html>`__ 中有更详细的描述。

当使用 ``--extended-lambda`` 标志时，CUDA 编译器支持对定义在 ``__device__`` 和 ``__global__`` 函数内的 lambda 以及定义在主机代码中的扩展仅设备 lambda 使用 ``*this`` 捕获模式。

以下是修改为使用 ``*this`` 捕获模式的上述示例：

.. code-block:: c++

   #include <cstdio>

   template <typename T>
   __global__ void foo(T in) { printf("\n value = %d", in()); }

   struct MyStruct {
       int var;
       __host__ __device__ MyStruct() : var(10) { };

       void run() {
           // note the "*this" capture specification
           auto lambda1 = [=, *this] __device__ {
               // reference to "var" causes the object denoted by '*this' to be captured by
               // value, and the GPU code will access 'copy_of_star_this->var'
               return var + 1;
           };
           // Kernel launch succeeds
           foo<<<1, 1>>>(lambda1);
           cudaDeviceSynchronize();
       }
   };

   int main() {
       MyStruct s1;
       s1.run();
   }

除非所选语言方言启用了 ``*this`` 捕获，否则不允许对定义在主机代码中的无注解 lambda 或扩展主机 - 设备 lambda 使用 ``*this`` 捕获模式。以下是支持和不支持用法的示例：

.. code-block:: c++

   struct MyStruct {
       int var;
       __host__ __device__ MyStruct() : var(10) { };

       void host_function() {
           // CORRECT, use in an extended device-only lambda
           auto lambda1 = [=, *this] __device__ { return var; };

           // Use in an extended host-device lambda
           // Error if *this capture not enabled by language dialect
           auto lambda2 = [=, *this] __host__ __device__ { return var; };

           // Use in an non-annotated lambda in host function
           // Error if *this capture not enabled by language dialect
           auto lambda3 = [=, *this]  { return var; };
       }

       __device__ void device_function() {
           // CORRECT, use in a lambda defined in a device-only function
           auto lambda1 = [=, *this] __device__ { return var; };

           // CORRECT, use in a lambda defined in a device-only function
           auto lambda2 = [=, *this] __host__ __device__ { return var; };

           // CORRECT, use in a lambda defined in a device-only function
           auto lambda3 = [=, *this]  { return var; };
       }

       __host__ __device__ void host_device_function() {
           // CORRECT, use in an extended device-only lambda
           auto lambda1 = [=, *this] __device__ { return var; };

           // Use in an extended host-device lambda
           // Error if *this capture not enabled by language dialect
           auto lambda2 = [=, *this] __host__ __device__ { return var; };

           // Use in an unannotated lambda in a host-device function
           // Error if *this capture not enabled by language dialect
           auto lambda3 = [=, *this]  { return var; };
       }
   };

.. _adl-with-extended-lambdas:

5.3.8.7. 参数依赖查找 (ADL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

如前所述，CUDA 编译器在调用主机编译器之前将扩展 lambda 表达式替换为占位符类型。占位符类型的一个模板参数使用包围原始 lambda 表达式的函数的地址。这可能导致额外的命名空间参与任何参数类型涉及扩展 lambda 表达式闭包类型的主机函数调用的 `参数依赖查找 (ADL) <https://en.cppreference.com/w/cpp/language/adl.html>`__。因此，主机编译器可能选择了错误的函数。

示例：

.. code-block:: c++

   namespace N1 {

   struct MyStruct {};

   template <typename T>
   void my_function(T);

   }; // namespace N1

   namespace N2 {

   template <typename T>
   int my_function(T);

   template <typename T>
   void run(T in) { my_function(in); }

   } // namespace N2

   void bar(N1::MyStruct in) {
       // For extended device-only lambda, the code sent to the host compiler is replaced with
       // the placeholder type instantiation expression
       //    ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::MyStruct in),(&bar),1> > { }'
       //
       // As a result, the namespace 'N1' participates in ADL lookup of the
       // call to "my_function()" in the body of N2::run, causing ambiguity.
       auto lambda1 = [=] __device__ { };
       N2::run(lambda1);
   }

在上面的示例中，CUDA 编译器将扩展 lambda 替换为涉及 ``N1`` 命名空间的占位符类型。因此，``N1`` 命名空间参与了 ``N2::run()`` 主体中 ``my_function(in)`` 的 ADL 查找，由于发现多个重载候选（``N1::my_function`` 和 ``N2::my_function``）而导致主机编译失败。

.. _polymorphic-function-wrappers:

5.3.9. 多态函数包装器
---------------------

``nvfunctional`` 头文件提供了多态函数包装器类模板 ``nvstd::function``。此类模板的实例可以存储、复制和调用任何可调用目标，例如 lambda 表达式。``nvstd::function`` 可在主机和设备代码中使用。

示例：

.. code-block:: c++

   #include <nvfunctional>

   __host__            int host_function()        { return 1; }
   __device__          int device_function()      { return 2; }
   __host__ __device__ int host_device_function() { return 3; }

   __global__ void kernel(int* result) {
       nvstd::function<int()> fn1 = device_function;
       nvstd::function<int()> fn2 = host_device_function;
       nvstd::function<int()> fn3 = [](){ return 10; };
       *result                    = fn1() + fn2() + fn3();
   }

   __host__ __device__ void host_device_test(int* result) {
       nvstd::function<int()> fn1 = host_device_function;
       nvstd::function<int()> fn2 = [](){ return 10; };
       *result                    = fn1() + fn2();
   }

   __host__ void host_test(int* result) {
       nvstd::function<int()> fn1 = host_function;
       nvstd::function<int()> fn2 = host_device_function;
       nvstd::function<int()> fn3 = [](){ return 10; };
       *result                    = fn1() + fn2() + fn3();
   }

**无效情况：**

- 主机代码中的 ``nvstd::function`` 实例不能用 ``__device__`` 函数的地址初始化，也不能用 ``operator()`` 是 ``__device__`` 函数的函数对象初始化。

- 类似地，设备代码中的 ``nvstd::function`` 实例不能用 ``__host__`` 函数的地址初始化，也不能用 ``operator()`` 是 ``__host__`` 函数的函数对象初始化。

- ``nvstd::function`` 实例不能在运行时从主机代码传递到设备代码（或反之）。

- 如果 ``__global__`` 函数从主机代码启动，则 ``nvstd::function`` 不能用于该 ``__global__`` 函数的参数类型。

无效情况示例：

.. code-block:: c++

   #include <nvfunctional>

   __device__ int device_function() { return 1; }
   __host__   int host_function() { return 3; }
   auto       lambda_host  = [] { return 0; };

   __global__ void k() {
       nvstd::function<int()> fn1 = host_function; // ERROR, initialized with address of __host__ function
       nvstd::function<int()> fn2 = lambda_host;   // ERROR, initialized with address of functor with
                                                   //        __host__ operator() function
   }

   __global__ void kernel(nvstd::function<int()> f1) {}

   void foo(void) {
       auto lambda_device = [=] __device__ { return 1; };

       nvstd::function<int()> fn1 = device_function; // ERROR, initialized with address of __device__ function
       nvstd::function<int()> fn2 = lambda_device;   // ERROR, initialized with address of functor with
                                                     //        __device__ operator() function
       kernel<<<1, 1>>>(fn2);                        // ERROR, passing nvstd::function from host to device
   }

``nvstd::function`` 在 ``nvfunctional`` 头文件中定义如下：

.. code-block:: c++

   namespace nvstd {

   template <typename RetType, typename ...ArgTypes>
   class function<RetType(ArgTypes...)> {
   public:
       // constructors
       __device__ __host__ function() noexcept;
       __device__ __host__ function(nullptr_t) noexcept;
       __device__ __host__ function(const function&);
       __device__ __host__ function(function&&);

       template<typename F>
       __device__ __host__ function(F);

       // destructor
       __device__ __host__ ~function();

       // assignment operators
       __device__ __host__ function& operator=(const function&);
       __device__ __host__ function& operator=(function&&);
       __device__ __host__ function& operator=(nullptr_t);
       template<typename F>
       __device__ __host__ function& operator=(F&&);

       // swap
       __device__ __host__ void swap(function&) noexcept;

       // function capacity
       __device__ __host__ explicit operator bool() const noexcept;

       // function invocation
       __device__ RetType operator()(ArgTypes...) const;
   };

   // null pointer comparisons
   template <typename R, typename... ArgTypes>
   __device__ __host__
   bool operator==(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

   template <typename R, typename... ArgTypes>
   __device__ __host__
   bool operator==(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

   template <typename R, typename... ArgTypes>
   __device__ __host__
   bool operator!=(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

   template <typename R, typename... ArgTypes>
   __device__ __host__
   bool operator!=(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

   // specialized algorithms
   template <typename R, typename... ArgTypes>
   __device__ __host__
   void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&);

   } // namespace nvstd

.. _c-c-language-restrictions:

5.3.10. C/C++ 语言限制
----------------------

.. _unsupported-features:

5.3.10.1. 不支持的特性
^^^^^^^^^^^^^^^^^^^^^^

- 运行时类型信息（RTTI）和异常在设备代码中不受支持：

  - ``typeid`` 关键字

  - ``dynamic_cast`` 关键字

  - ``try/catch/throw`` 关键字

- ``long double`` 在设备代码中不受支持。

- 三字符组在任何平台上都不受支持。双字符组在 Windows 上不受支持。

- 用户定义的 ``operator new``、``operator new[]``、``operator delete`` 或 ``operator delete[]`` 不能用于替换编译器提供的相应内建运算符，在主机和设备上均被视为未定义行为。

.. _namespace-reservations:

5.3.10.2. 命名空间保留
^^^^^^^^^^^^^^^^^^^^^^

除非另有说明，向顶级命名空间 ``cuda::``、``nv::`` 或 ``cooperative_groups::``，或其内部的任何嵌套命名空间添加定义均为未定义行为。我们允许将 ``cuda::`` 作为子命名空间使用，如下所示：

示例：

.. code-block:: c++

   namespace cuda {   // same for "nv" and "cooperative_groups" namespaces

   struct foo;        // ERROR, class declaration in the "cuda" namespace

   void bar();        // ERROR, function declaration in the "cuda" namespace

   namespace utils {} // ERROR, namespace declaration in the "cuda" namespace

   } // namespace cuda

   namespace utils {
   namespace cuda {

   // CORRECT, namespace "cuda" may be used nested within a non-reserved namespace
   void bar();

   } // namespace cuda
   } // namespace utils

   // ERROR, Equivalent to adding symbols to namespace "cuda" at global scope
   using namespace utils;

.. _pointers-and-memory-addresses:

5.3.10.3. 指针和内存地址
^^^^^^^^^^^^^^^^^^^^^^^^

指针解引用（``*pointer``、``pointer->member``、``pointer[0]``）仅允许在相关内存所在的同一执行空间中进行。以下情况会导致未定义行为，通常是段错误和应用程序终止。

- 在主机上解引用指向全局内存、共享内存或常量内存的指针。

- 在设备代码中解引用指向主机内存的指针。

以下限制适用于函数：

- 不允许在主机代码中获取 ``__device__`` 函数的地址。

- 在主机代码中获取的 ``__global__`` 函数地址不能用于设备代码。类似地，在设备代码中获取的 ``__global__`` 函数地址不能用于主机代码。

如内存空间说明符一节所述，通过 ``cudaGetSymbolAddress()`` 获取的 ``__device__`` 或 ``__constant__`` 变量的地址只能用于主机代码。

.. _variables:

5.3.10.4. 变量
^^^^^^^^^^^^^^

.. _local-variables:

5.3.10.4.1. 局部变量
""""""""""""""""""""

在主机上执行的函数内，非 ``extern`` 变量声明不允许使用 ``__device__``、``__tile__``、``__shared__``、``__managed__`` 和 ``__constant__`` 内存空间说明符。

示例：

.. code-block:: c++

   __host__ void host_function() {
       int x;                   // CORRECT, __host__ variable
       __device__   int y;      // ERROR,   __device__ variable declaration within a host function
       __tile__     int z;      // ERROR,   __tile__ variable declaration within a host function
       __shared__   int w;      // ERROR,   __shared__ variable declaration within a host function
       __managed__  int h;      // ERROR,   __managed__ variable  declaration within a host function
       __constant__ int i;      // ERROR,   __constant__ variable declaration within a host function
       extern __device__ int j; // CORRECT, extern __device__ variable
   }

在设备上执行的函数内，既非 ``extern`` 也非 ``static`` 的变量声明不允许使用 ``__device__``、``__tile__``、``__constant__`` 和 ``__managed__`` 内存空间说明符。

.. code-block:: c++

   __device__ void device_function() {
       int x;                   // CORRECT, __device__ variable
       __constant__      int y; // ERROR,   __constant__ variable declaration within a device function
       __managed__       int z; // ERROR,   __managed__ variable  declaration within a device function
       extern __device__ int k; // CORRECT, extern __device__ variable
   }

另请参见 ``static`` 变量一节。

.. _const-qualified-variables:

5.3.10.4.2. ``const`` 限定变量
""""""""""""""""""""""""""""""

在全局、命名空间或类作用域声明的、没有内存空间注解（``__device__``、``__tile__`` 或 ``__constant__``）的 ``const`` 限定变量被视为主机变量。
设备代码不能包含对该变量的引用或获取其地址。

如果满足以下条件，该变量可以直接在设备代码中使用：

- 在使用点之前已用常量表达式初始化，

- 类型不是 ``volatile`` 限定的，并且

- 它具有以下类型之一：

  - 内建整数类型，或

  - 内建浮点类型，但主机编译器是 Microsoft Visual Studio 时除外。

从 C++14 开始，建议使用 ``constexpr`` 或 ``inline constexpr`` （C++17）变量，而不是 ``const`` 限定变量。
``constexpr`` 变量不受相同的类型限制，可以直接在设备代码中使用。

``__managed__`` 变量不支持 ``const`` 限定类型。

示例：

.. code-block:: c++

   const            int   ConstVar          = 10;
   const            float ConstFloatVar     = 5.0f;
   inline constexpr float ConstexprFloatVar = 5.0f; // C++17

   struct MyStruct {
       static const            int   ConstVar          = 20;
   //  static const            float ConstFloatVar     = 5.0f; // ERROR, static const variables cannot be float
       static inline constexpr float ConstexprFloatVar = 5.0f; // CORRECT
   };

   extern const int ExternVar;

   __device__ void foo() {
       int array1[ConstVar];                     // CORRECT
       int array2[MyStruct::ConstVar];           // CORRECT

       const     float var1 = ConstFloatVar;     // CORRECT, except when the host compiler is Microsoft Visual Studio.
       constexpr float var2 = ConstexprFloatVar; // CORRECT
   //  int             var3 = ExternVar;          // ERROR, "ExternVar" is not initialized with a constant expression
   //  int&            var4 = ConstVar;           // ERROR, reference to host variable
   //  int*            var5 = &ConstVar;          // ERROR, address of host variable
   }

在 `Compiler Explorer <https://godbolt.org/z/eWG8KxK94>`__ 上查看示例。

.. _volatile-qualified-variables:

5.3.10.4.3. ``volatile`` 限定变量
"""""""""""""""""""""""""""""""""

.. note::

   支持 ``volatile`` 关键字是为了保持与 ISO C++ 的兼容性。但是，其剩余的非弃用用法（如果有的话）几乎都不适用于 GPU。

对 ``volatile`` 限定对象的读写不是原子的，会被编译为一条或多条不保证以下情况的 volatile 指令：

- 内存操作的顺序，或

- 硬件执行的内存操作数量与 PTX 指令数量匹配。

在 tile 代码中，``volatile`` 关键字对内存访问的行为没有影响。

CUDA C++ ``volatile`` 不适用于：

**线程间同步**：请改用通过 ``cuda::atomic_ref``、``cuda::atomic`` 或原子函数提供的原子操作。

原子内存操作提供线程间同步保证，并且比 ``volatile`` 操作具有更好的性能。但是，CUDA C++ ``volatile`` 操作不提供任何线程间同步保证，因此不适用于此目的。以下示例展示了如何使用原子操作在两个线程之间传递消息。

.. tab-set::

   .. tab-item:: ``cuda::atomic_ref``

      .. code-block:: c++

         #include <cuda/atomic>

         __global__ void kernel(int* flag, int* data) {
             cuda::atomic_ref<int, cuda::thread_scope_device> atomic_ref{*flag};
             if (threadIdx.x == 0) {
                 // Consumer: blocks until flag is set by producer, then reads data
                 while(atomic_ref.load(cuda::memory_order_acquire) == 0)
                     ;
                 if (*data != 42)
                     __trap(); // Errors if wrong data read
             }
             else if (threadIdx.x == 1) {
                 // Producer: writes data then sets flag
                 *data = 42;
                 atomic_ref.store(1, cuda::memory_order_release);
             }
         }

   .. tab-item:: ``cuda::atomic``

      .. code-block:: c++

         #include <cuda/atomic>

         __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {
             if (threadIdx.x == 0) {
                 // Consumer: blocks until flag is set by producer, then reads data
                 while(flag->load(cuda::memory_order_acquire) == 0)
                     ;
                 if (*data != 42)
                     __trap(); // Errors if wrong data read
             }
             else if (threadIdx.x == 1) {
                 // Producer: writes data then sets flag
                 *data = 42;
                 flag->store(1, cuda::memory_order_release);
             }
         }

   .. tab-item:: 原子函数（ ``atomicAdd`` 和 ``atomicExch`` ）

      .. code-block:: c++

         __global__ void kernel(int* flag, int* data) {
             if (threadIdx.x == 0) {
                 // Consumer: blocks until flag is set by producer, then reads data
                 while(atomicAdd(flag, 0) == 0)
                     ;                // Load with Relaxed Read-Modify-Write
                 __threadfence();     // SequentiallyConsistent fence
                 if (*data != 42)
                     __trap();        // Errors if wrong data read
             } else if (threadIdx.x == 1) {
                 // Producer: writes data then sets flag
                 *data = 42;
                 __threadfence();     // SequentiallyConsistent fence
                 atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write
             }
         }

**内存映射 IO** （MMIO）：请改用通过内联 PTX 提供的 PTX MMIO 操作。

PTX MMIO 操作严格保留执行的内存访问次数。但是，CUDA C++ ``volatile`` 操作不保留执行的内存访问次数，可能以不确定的方式执行比请求更多或更少的访问。这使它们不适用于 MMIO。以下示例展示了如何使用 PTX MMIO 操作读取和写入寄存器。

.. code-block:: c++

   __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
       // Write to MMIO register:
       int value = 13;
       asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
           :
           : "l"(mmio_reg0), "r"(value) : "memory");

       // Read MMIO register:
       asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
           : "=r"(value)
           : "l"(mmio_reg1) : "memory");

       if (value != 42)
           __trap(); // Errors if wrong data read
   }

.. _static-variables:

5.3.10.4.4. ``static`` 变量
"""""""""""""""""""""""""""

设备函数中允许使用 ``static`` 局部变量。

封闭函数的每个执行空间使用不同的静态变量。例如，

- ``__host__ __device__`` 函数具有一份用于主机执行的静态变量副本和一份用于设备执行的副本。

- ``__tile__ __device__`` 函数具有一份用于 tile 执行的副本和一份用于 SIMT 执行的副本。

如果函数具有 ``__host__`` 执行空间说明符，则仅当定义了 ``__CUDA_ARCH__`` 时才允许使用具有显式内存空间的 ``static`` 变量，例如 ``static __device__/__tile__/__constant__/__shared__/__managed__``。

下面展示了函数作用域 ``static`` 变量的合法和非法用法示例。

.. code-block:: c++

   struct TrivialStruct {
       int x;
   };

   struct NonTrivialStruct {
       __device__ NonTrivialStruct(int x) {}
   };

   __device__ void device_function(int x) {
       static int v1;              // CORRECT, implicit __device__ memory space specifier
       static int v2 = 11;         // CORRECT, implicit __device__ memory space specifier
   //  static int v3 = x;           // ERROR, dynamic initialization is not allowed

       static __managed__  int v4; // CORRECT, explicit
       static __device__   int v5; // CORRECT, explicit
       static __constant__ int v6; // CORRECT, explicit
       static __shared__   int v7; // CORRECT, explicit

       static TrivialStruct    s1;     // CORRECT, implicit __device__ memory space specifier
       static TrivialStruct    s2{22}; // CORRECT, implicit __device__ memory space specifier
   //  static TrivialStruct    s3{x};   // ERROR, dynamic initialization is not allowed
   //  static NonTrivialStruct s4{3};   // ERROR, dynamic initialization is not allowed
   }

在 `Compiler Explorer <https://godbolt.org/z/TdYKaTq3f>`__ 上查看示例。

.. code-block:: c++

   __host__ __device__ void host_device_function() {
       static            int v1; // CORRECT, implicit __device__ memory space specifier
   //  static __device__ int v2;  // ERROR, __device__-only variable inside a host-device function
   #ifdef __CUDA_ARCH__
       static __device__ int v3; // CORRECT, declaration is only visible during device compilation
   #else
       static int v4;            // CORRECT, declaration is only visible during host compilation
   #endif
   }

在 `Compiler Explorer <https://godbolt.org/z/18qhjn8P1>`__ 上查看示例。

.. code-block:: c++

   #include <cassert>

   __host__ __device__ int host_device_function() {
       static int v = 0;
       v++;
       return v;
   }

   __global__ void kernel() {
       int ret = host_device_function(); // v = 1
       assert(ret == 4);                 // FAIL
   }

   int main() {
       host_device_function();           // v = 1
       host_device_function();           // v = 2
       int ret = host_device_function(); // v = 3
       assert(ret == 3);                 // OK
       kernel<<<1, 1>>>();
       cudaDeviceSynchronize();
   }

在 `Compiler Explorer <https://godbolt.org/z/Wqo9WjvYY>`__ 上查看示例。

.. _extern-variables:

5.3.10.4.5. ``extern`` 变量
"""""""""""""""""""""""""""

在整个程序编译模式下编译时，不能使用 ``extern`` 关键字定义具有外部链接的 ``__device__``、``__tile__``、``__shared__``、``__managed__`` 和 ``__constant__`` 变量。对于 ``__tile__`` 变量，此限制在分离编译模式下也适用。

唯一的例外是动态共享内存分配一节中描述的动态分配的 ``__shared__`` 变量。

.. code-block:: c++

   __device__        int x; // OK
   extern __device__ int y; // ERROR in whole program compilation mode
   extern __shared__ int z; // OK

.. _functions:

5.3.10.5. 函数
^^^^^^^^^^^^^^

.. _recursion-restrictions:

5.3.10.5.1. 递归
""""""""""""""""

``__global__``、``__tile_global__`` 和 ``__tile__`` 函数不支持递归，而 ``__device__`` 和 ``__host__ __device__`` 函数没有此限制。

.. _external-linkage:

5.3.10.5.2. 外部链接
""""""""""""""""""""

具有外部链接的设备变量或函数需要跨多个翻译单元的分离编译模式。

在分离编译模式下，如果 ``__device__`` 或 ``__global__`` 函数定义需要存在于特定翻译单元中，则该函数的参数和返回类型在该翻译单元中必须是完整的。此概念也称为单定义规则使用（One Definition Rule-use），即 ODR-use。

示例：

.. code-block:: c++

   //first.cu:
   struct S;                   // forward declaration
   __device__ void foo(S);     // ERROR, type 'S' is an incomplete type
   __device__ auto* ptr = foo; // ODR-use, address taken

   int main() {}

   //second.cu:
   struct S {};               // struct definition
   __device__ void foo(S) {}  // function definition

.. code-block:: text

   # compiler invocation
   $ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
   nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                    first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
   nvlink fatal   : merge_elf failed

.. _formal-parameters:

5.3.10.5.3. 形参
""""""""""""""""

``__device__``、``__tile__``、``__shared__``、``__managed__`` 和 ``__constant__`` 内存空间说明符不允许用于形参。

.. code-block:: c++

   void device_function1(__device__ int x) { } // ERROR, __device__ parameter
   void device_function2(__shared__ int x) { } // ERROR, __shared__ parameter

.. _global-function-parameters:

5.3.10.5.4. ``__global__`` 函数参数
""""""""""""""""""""""""""""""""""""

``__global__`` 或 ``__tile_global__`` 函数有以下限制：

- 它不能具有可变数量的参数，即 C 省略号语法 ``...`` 和 ``va_list`` 类型。C++11 可变参数模板是允许的，但须遵守 ``__global__`` 可变参数模板一节中描述的限制。

- 函数参数通过常量内存传递给设备，其总大小限制为 32,764 字节。

- 函数参数不能是 ``std::initializer_list`` 类型。

- 多态类参数（``virtual``）被视为未定义行为。

- Lambda 表达式和闭包类型是允许的，但须遵守 Lambda 表达式和 ``__global__`` 函数参数一节中描述的限制。

- 对于 ``__tile_global__`` 函数，函数参数不能是按值传递的类、结构体或联合体。

.. _global-function-arguments:

5.3.10.5.5. ``__global__`` 函数参数传递
""""""""""""""""""""""""""""""""""""""""

从设备代码启动 ``__global__`` 函数时，每个参数必须是可平凡复制（trivially copyable）且可平凡析构（trivially destructible）的。

从主机代码启动 ``__global__`` 函数时，每个参数类型可以是非平凡可复制或非平凡可析构的。但是，对这些类型的处理不遵循标准 C++ 模型，如下所述。用户代码必须确保此工作流不影响程序正确性。该工作流在两个方面与标准 C++ 存在差异：

1. **原始内存复制代替拷贝构造函数调用**

   CUDA Runtime 通过复制原始内存内容将内核参数传递给 ``__global__`` 函数，最终使用 ``memcpy``。如果参数是非平凡可复制的并提供了用户定义的拷贝构造函数，则在主机到设备的复制过程中会跳过该调用的操作和副作用。

   示例：

   .. code-block:: c++

      #include <cassert>

      struct MyStruct {
          int  value = 1;
          int* ptr;

          MyStruct() = default;

          __host__ __device__ MyStruct(const MyStruct&) { ptr = &value; }
      };

      __global__ void device_function(MyStruct my_struct) {
          // this assert fails because "my_struct" is obtained by copying
          // the raw memory content and the copy constructor is skipped.
          assert(my_struct.ptr == &my_struct.value); // FAIL
      }

      void host_function(MyStruct my_struct) {
          assert(my_struct.ptr == &my_struct.value); // CORRECT
      }

      int main() {
          MyStruct my_struct;
          host_function(my_struct);
          device_function<<<1, 1>>>(my_struct); // copy constructor invoked in the host-side only
          cudaDeviceSynchronize();
      }

   在 `Compiler Explorer <https://godbolt.org/z/xhqe16dec>`__ 上查看示例。

2. **析构函数可能在 ``__global__`` 函数完成之前被调用**

   内核启动与主机执行是异步的。因此，如果 ``__global__`` 函数参数具有非平凡析构函数，析构函数可能在主机代码中甚至在 ``__global__`` 函数完成执行之前就已执行。这可能会破坏析构函数具有副作用的程序。

   示例：

   .. code-block:: c++

      #include <cassert>

      __managed__ int var = 0;

      struct MyStruct {
          __host__ __device__ ~MyStruct() { var = 3; }
      };

      __global__ void device_function(MyStruct my_struct) {
          assert(var == 0); // FAIL, MyStruct::~MyStruct() sets the value to 3
      }

      int main() {
          MyStruct my_struct;
          // GPU kernel execution is asynchronous with host execution.
          // As a result, MyStruct::~MyStruct() could be executed before
          // the kernel finishes executing.
          device_function<<<1, 1>>>(my_struct);
          cudaDeviceSynchronize();
      }

   在 `Compiler Explorer <https://godbolt.org/z/cn6Y5W6zs>`__ 上查看示例。

.. _classes:

5.3.10.6. 类
^^^^^^^^^^^^

.. _class-type-variables:

5.3.10.6.1. 类类型变量
""""""""""""""""""""""

具有 ``__device__``、``__tile__``、``__constant__``、``__managed__`` 或 ``__shared__`` 内存空间的变量定义不能具有带非空构造函数或非空析构函数的类类型。如果类类型的构造函数是平凡的（trivial），或在翻译单元中的某一点满足以下所有条件，则认为它是空的：

- 构造函数已被定义。

- 构造函数没有参数、空的初始化列表和空的复合语句函数体。

- 其类没有 ``virtual`` 函数、``virtual`` 基类或非 ``static`` 数据成员初始化器。

- 其所有基类的默认构造函数可被视为空的。

- 对于类的所有类类型（或其数组）的非 ``static`` 数据成员，默认构造函数可被视为空的。

如果类的析构函数是平凡的，或在翻译单元中的某一点满足以下所有条件，则认为它是空的：

- 析构函数已被定义。

- 析构函数体是空的复合语句。

- 其类没有 ``virtual`` 函数或 ``virtual`` 基类。

- 其所有基类的析构函数可被视为空的。

- 对于类的所有类类型（或其数组）的非 ``static`` 数据成员，析构函数可被视为空的。

.. _data-members:

5.3.10.6.2. 数据成员
""""""""""""""""""""

``__device__``、``__tile__``、``__shared__``、``__managed__`` 和 ``__constant__`` 内存空间说明符不允许用于 ``class``、``struct`` 和 ``union`` 数据成员。

仅支持在编译时求值的 ``static`` 数据成员，例如 const 限定和 ``constexpr`` 变量。

.. code-block:: c++

   struct MyStruct {
      static inline constexpr int value1 = 10; // C++17
      static constexpr        int value2 = 10; // C++11
      static const            int value3 = 10;
   // static                  int value4; // ERROR
   };

.. _function-members:

5.3.10.6.3. 函数成员
""""""""""""""""""""

``__global__`` 和 ``__tile_global__`` 函数不能是 ``struct``、``class`` 或 ``union`` 的成员。

``__global__`` 或 ``__tile_global__`` 函数允许出现在 ``friend`` 声明中，但不能被定义。

示例：

.. code-block:: c++

   struct MyStruct {
       friend __global__ void f();   // CORRECT, friend declaration only

   //  friend __global__ void g() {} // ERROR, friend definition
   };

在 `Compiler Explorer <https://godbolt.org/z/rv6cP3b9j>`__ 上查看示例。

.. _implicitly-defaulted-functions:

5.3.10.6.4. 隐式声明和非虚显式默认函数
""""""""""""""""""""""""""""""""""""""

隐式声明的特殊成员函数是当用户未声明时编译器为类声明的那些函数；显式默认函数是用户声明但标记为 ``= default`` 的函数。隐式声明或显式默认的特殊成员函数有默认构造函数、拷贝构造函数、移动构造函数、拷贝赋值运算符、移动赋值运算符和析构函数。

设 ``F`` 表示一个非 ``virtual`` 函数，它是隐式声明的或在其首次声明时显式默认的。``F`` 的执行空间说明符是调用它的所有函数的执行空间说明符的并集。请注意，在此分析中，``__global__`` 调用者将被视为 ``__device__`` 调用者。例如：

.. code-block:: c++

   class Base {
       int x;
   public:
       __host__ __device__ Base() : x(10) {}
   };

   class Derived : public Base {
       int y;
   };

   class Other: public Base {
       int z;
   };

   __device__ void foo() {
       Derived D1;
       Other D2;
   }

   __host__ void bar() {
       Other D3;
   }

在这种情况下，隐式声明的构造函数 ``Derived::Derived()`` 将被视为 ``__device__`` 函数，因为它仅从 ``__device__`` 函数 ``foo()`` 调用。隐式声明的构造函数 ``Other::Other()`` 将被视为 ``__host__ __device__`` 函数，因为它既从 ``__device__`` 函数 ``foo()`` 又从 ``__host__`` 函数 ``bar()`` 调用。

此外，如果 ``F`` 是隐式声明的 ``virtual`` 函数（例如 ``virtual`` 析构函数），则如果 ``D`` 不是隐式声明的，被 ``F`` 覆盖的每个虚函数 ``D`` 的执行空间将被添加到 ``F`` 的执行空间集合中。

例如：

.. code-block:: c++

   struct Base1 {
       virtual __host__ __device__ ~Base1() {}
   };

   struct Derived1 : Base1 {}; // implicitly-declared virtual destructor
                               // ~Derived1() has __host__ __device__  execution space specifiers

   struct Base2 {
       virtual __device__ ~Base2() = default;
   };

   struct Derived2 : Base2 {}; // implicitly-declared virtual destructor
                               // ~Derived2() has __device__ execution space specifiers

.. _polymorphic-classes:

5.3.10.6.5. 多态类
""""""""""""""""""

多态类，即具有 ``virtual`` 函数、从其他多态类派生或具有多态数据成员的类，受以下限制约束：

- 将多态对象从设备复制到主机或从主机复制到设备（包括 ``__global__`` 函数参数）是未定义行为。

- 被覆盖的 ``virtual`` 函数的执行空间必须与基类中函数的执行空间匹配。

示例：

.. code-block:: c++

   struct MyClass {
       virtual __host__ __device__ void f() {}
   };

   __global__ void kernel(MyClass my_class) {
       my_class.f(); // undefined behavior
   }

   int main() {
       MyClass my_class;
       kernel<<<1, 1>>>(my_class);
       cudaDeviceSynchronize();
   }

在 `Compiler Explorer <https://godbolt.org/z/To39sGTrW>`__ 上查看示例。

.. code-block:: c++

   struct BaseClass {
       virtual __host__ __device__ void f() {}
   };

   struct DerivedClass : BaseClass {
       __device__ void f() override {} // ERROR
   };

在 `Compiler Explorer <https://godbolt.org/z/xfKhEGfdG>`__ 上查看示例。

.. _windows-class-layout:

5.3.10.6.6. Windows 特定类布局
""""""""""""""""""""""""""""""

CUDA 编译器遵循 IA64 ABI 进行类布局，而 Microsoft Visual Studio 则不然。这阻碍了特殊对象在主机和设备代码之间的按位复制，如下所述。

设 ``T`` 表示指向成员类型的指针，或满足以下任一条件的类类型：

- ``T`` 是多态类

- ``T`` 具有多重继承，且有多个直接或间接的空基类。

- 所有直接和间接基类 ``B`` 都是空的，且 ``T`` 的第一个字段 ``F`` 的类型在其定义中使用了 ``B``，使得 ``B`` 在 ``F`` 的定义中布局在偏移 0 处。

使用 Microsoft Visual Studio 编译时，类型为 ``T`` 的类、具有类型 ``T`` 基类的类或具有类型 ``T`` 数据成员的类，在主机和设备之间可能具有不同的类布局和大小。

将此类对象从设备复制到主机或从主机复制到设备（包括 ``__global__`` 函数参数）是未定义行为。

.. _templates:

5.3.10.7. 模板
^^^^^^^^^^^^^^

如果出现以下任一情况，则类型不能用作 ``__global__`` 函数或 ``__device__/__constant__`` 变量（C++14）的模板参数：

- 该类型定义在 ``__host__`` 或 ``__host__ __device__`` 函数作用域内。

- 该类型是未命名的，例如匿名结构体或 lambda 表达式，除非该类型是 ``__device__`` 或 ``__global__`` 函数的局部类型。

- 该类型是具有 ``private`` 或 ``protected`` 的类成员，除非该类是 ``__device__`` 或 ``__global__`` 函数的局部类型。

- 该类型由上述任何类型组合而成。

示例：

.. code-block:: c++

   template <typename T>
   __global__ void kernel() {}

   template <typename T>
   __device__ int device_var; // C++14

   struct {
       int v;
   } unnamed_struct;

   void host_function() {
       struct LocalStruct {};
   //  kernel<LocalStruct><<<1, 1>>>(); // ERROR, LocalStruct is defined within a host function
       int data = 4;
   //  cudaMemcpyToSymbol(device_var<LocalStruct>, &data, sizeof(data)); // ERROR, same as above

       auto lambda = [](){};
   //  kernel<decltype(lambda)><<<1, 1>>>();         // ERROR, unnamed type
   //  kernel<decltype(unnamed_struct)><<<1, 1>>>(); // ERROR, unnamed type
   }

   class MyClass {
   private:
       struct PrivateStruct {};
   public:
       static void launch() {
   //      kernel<PrivateStruct><<<1, 1>>>(); // ERROR, private type
       }
   };

在 `Compiler Explorer <https://godbolt.org/z/EhTn3GT3z>`__ 上查看示例。

.. _restrictions-in-tile-code:

5.3.10.8. Tile 代码中的限制
^^^^^^^^^^^^^^^^^^^^^^^^^^^

用 ``__tile__`` 或 ``__tile_global__`` 注解的函数有以下额外限制：

- 以下语言结构在 tile 代码中不受支持：

  - ``do``、``while`` 或 ``for`` 循环内的返回语句。

  - 虚函数调用。

  - Goto 语句。

  - Switch 语句。

  - 产生函数指针、函数引用、指向成员变量的指针或指向成员函数的指针的表达式。

  - 函数指针和指向成员函数的指针的调用。

  - 指向成员变量的指针的访问。

  - 128 位整数或浮点类型。

  - 包含位域的类型。

  - 大小超过 16 MB 的类型。

  - 具有虚基类或虚函数的类型。

  - 使用非 placement 的 ``new`` 或 ``delete`` 运算符进行动态内存分配或释放。

- ``__tile__`` 或 ``__tile_global__`` 函数必须在其声明所在的同一翻译单元中具有函数体。

- 不支持用 ``__tile__`` 注解虚函数。

- ``__tile_global__`` 或 ``__tile__`` 函数不能使用 C 省略号语法 ``...`` 具有可变数量的参数。

- ``__tile_global__`` 或 ``__tile__`` 函数不能直接或间接递归。

- 对于 ``__tile_global__`` 函数，函数参数不能是按值传递的类、结构体或联合体。

- Tile 代码不能执行设备端内核启动，且 tile 内核不能从设备端内核调用启动。

- 在 tile 代码中不支持直接访问 ``__half``、``__nv_bfloat16`` 及相关扩展浮点类型的 ``__x`` 成员变量。

.. _c-11-restrictions:

5.3.11. C++11 限制
------------------

.. _inline-namespaces-restrictions:

5.3.11.1. ``inline`` 命名空间
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当另一个同名且类型签名相同的实体定义在封闭命名空间中时，不允许在 ``inline`` 命名空间内定义以下实体之一：

- ``__global__`` 或 ``__tile_global__`` 函数。

- ``__device__``、``__tile__``、``__constant__``、``__managed__``、``__shared__`` 变量。

- 具有表面或纹理类型的变量，例如 ``cudaSurfaceObject_t`` 或 ``cudaTextureObject_t``。

示例：

.. code-block:: c++

   __device__ int my_var; // global scope

   inline namespace NS {

   __device__ int my_var; // namespace scope

   } // namespace NS

.. _inline-unnamed-namespaces-restrictions:

5.3.11.2. ``inline`` 未命名命名空间
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下实体不能在 ``inline`` 未命名命名空间内的命名空间作用域中声明：

- ``__global__`` 或 ``__tile_global__`` 函数。

- ``__device__``、``__tile__``、``__constant__``、``__managed__``、``__shared__`` 变量。

- 具有表面或纹理类型的变量，例如 ``cudaSurfaceObject_t`` 或 ``cudaTextureObject_t``。

.. _constexpr-functions-restrictions:

5.3.11.3. ``constexpr`` 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``__global__`` 函数不能声明为 ``constexpr``。默认情况下，``constexpr`` 函数不能从具有不兼容执行空间的函数调用，与标准函数相同。在本节列出的示例代码中，``UB`` 代表"未定义行为"（undefined behavior）。

- 在主机编译阶段（``__CUDA_ARCH__`` 宏未定义时），从主机函数调用没有显式或隐式 ``__host__`` 注解的 ``constexpr`` 函数是未定义行为。示例：

  .. code-block:: c++

     constexpr __device__             int  device_func() { return 0; }
     constexpr __tile__               int  tile_func()   { return 0; }

     constexpr __device__ __host__   int host_device_func() { return 0; }

     int main() {
         constexpr int x1 = device_func(); // UB: calling a __device__-only constexpr function from host code
         constexpr int x2 = tile_func();   // UB: calling a __tile__-only constexpr function from host code
         constexpr int x3 = host_device_func(); // OK

     }

- 在设备编译阶段（``__CUDA_ARCH__`` 宏已定义时），从 ``__device__`` 或 ``__global__`` 函数调用没有显式或隐式 ``__device__`` 注解的 ``constexpr`` 函数是未定义行为。示例：

  .. code-block:: c++

     constexpr  int host_func() { return 0; }

     __device__ void dmain()
     {
         int x = host_func();  // UB: calling a host-only constexpr function from device code
     }

- 在设备编译阶段（``__CUDA_ARCH__`` 宏已定义时），从 ``__tile__`` 或 ``__tile_global__`` 函数调用没有显式或隐式 ``__tile__`` 注解的 ``constexpr`` 函数是未定义行为。示例：

  .. code-block:: c++

     constexpr  int host_func() { return 0; }

     __tile__ void dmain()
     {
         int x = host_func();  // UB: calling a host-only constexpr function from tile code
     }

请注意，即使相应的模板函数标记了 ``constexpr`` 关键字，函数模板特化也可能不是 ``constexpr`` 函数。

**放宽的 constexpr 函数支持**

实验性的 ``nvcc`` 标志 ``--expt-relaxed-constexpr`` 可用于按下文所述放宽此约束。``nvcc`` 还会定义宏 ``__CUDACC_RELAXED_CONSTEXPR__``。在本节列出的示例代码中，``UB`` 代表"未定义行为"。

当指定 ``--expt-relaxed-constexpr`` 标志时，编译器将支持跨执行空间调用，如下所示：

1. 如果对 ``constexpr`` 函数的跨执行空间调用发生在需要常量求值的上下文中（例如 constexpr 变量的初始化器中），则支持该调用。示例：

   .. code-block:: c++

      constexpr __host__ int host_func(int x) { return x + 1; };

      __global__ void doit() {
           constexpr int val = host_func(1); // OK: call is in a context that
                                             // requires constant evaluation.
      }

      __tile_global__ void tile_doit() {
           constexpr int val = host_func(1); // OK: call is in a context that
                                             // requires constant evaluation.
      }

      constexpr __device__ int device_func(int x) { return x + 1; }

      constexpr __tile__ int tile_func(int x) { return x + 2; }

      int main() {
      constexpr int val = device_func(1) + tile_func(1); // OK: call is in a context that
                                                         // requires constant evaluation.
      }

2. 否则：

   1. 从 Tile 代码对没有显式或隐式 ``__tile__`` 注解的 ``constexpr`` 函数的跨执行空间调用，在语言规则不要求常量折叠的上下文之外不受支持。示例：

      .. code-block:: c++

         constexpr __host__ int host_func(int x) { return x + 1; }

         __tile__ int doit(int in) {
             in = host_func(in); // UB: call occurs outside of a context that requires
                                 // constant evaluation.
             constexpr int other = host_func(10); // OK with -expt-relaxed-constexpr:
                                                  // call is  required to be evaluated at compile time
         }

   2. 在 SIMT 设备代码生成期间，会为仅主机的 ``constexpr`` 函数 ``host_func`` 的主体生成设备代码，除非 ``host_func`` 未被使用或仅在常量求值上下文中被调用。示例：

      .. code-block:: c++

         // NOTE: "host_func" is emitted in generated device code because it is
         // called from device code in a non-constexpr context
         constexpr __host__ int host_func(int x) { return x + 1; }

         __device__ int doit(int in) {
             in = host_func(in);  // OK, even though argument is not a constant expression
             return in;
         }

   3. 适用于 ``__device__`` 函数的所有代码限制也适用于从 SIMT 设备代码调用的 ``constexpr`` 仅主机函数 ``H``。但是，编译器可能不会针对这些限制为 ``H`` 发出任何构建时诊断。原因是诊断通常在解析期间生成，而 ``H`` 可能在翻译单元后面遇到从设备代码对 ``H`` 的调用之前就已经被解析了。

      例如，以下代码模式在 ``H`` 的主体中不受支持（与任何 ``__device__`` 函数一样），但可能不会生成编译器诊断：

      - 对主机变量或仅主机非 ``constexpr`` 函数的 ODR-use。示例：

        .. code-block:: c++

           int host_var1, host_var2;

           constexpr __host__ int* host_func(bool b) { return b ? &host_var1 : &host_var2; };

           __device__ int doit(bool flag) {
               int *ptr;
               ptr = host_func(flag); // UB: host_func() attempts to refer to host variables 'host_var1' and 'host_var2'.
                            // code will compile, but will NOT execute correctly.
               return *ptr;
           }

      - 使用异常（``throw/catch``）和 RTTI（``typeid``、``dynamic_cast``）。示例：

        .. code-block:: c++

           struct Base { };
           struct Derived : public Base { };

           // NOTE: "host_func" is emitted in generated device code
           constexpr int host_func(bool b, Base *ptr) {
             if (b) {
               return 1;
             } else if (typeid(ptr) == typeid(Derived)) { // UB: use of typeid in code executing on the GPU
               return 2;
             } else {
               throw int{4}; // UB: use of throw in code executing on the GPU
             }
           }

           __device__ void doit(bool flag) {
               int val;
               Derived d;
               val = host_func(flag, &d); //UB: host_func() attempts use typeid and throw(), which are not allowed in code that executes on the GPU
           }

   4. 在主机代码生成期间，``constexpr`` 非主机函数 ``F`` 的主体被保留在发送到主机编译器的代码中。如果 ``F`` 的主体试图 ODR-use 命名空间作用域的设备或 ``__tile__`` 变量，或非主机非 ``constexpr`` 函数，则从主机代码对 ``F`` 的调用不受支持（代码可能在无编译器诊断的情况下构建，但运行时可能行为不正确）。示例：

      .. code-block:: c++

         __device__ int device_var1, device_var2;

         constexpr __device__ int* device_func(bool b) { return b ? &device_var1 : &device_var2; };

         __tile__ int tile_var1, tile_var2;

         constexpr __tile__ int* tile_func(bool b) { return b ? &tile_var1 : &tile_var2; };

         int doit1(bool flag) {
             int *ptr;
             ptr = device_func(flag); // UB: device_func() attempts to refer to device variables 'device_var1' and 'device_var2'
                                      // code will compile, but will NOT execute correctly.
             return *ptr;
         }

         int doit2(bool flag) {
             int *ptr;
             ptr = tile_func(flag); // UB: tile_func() attempts to refer to __tile__ variables 'tile_var1' and 'tile_var2'
                                    // code will compile, but will NOT execute correctly.
             return *ptr;
         }

.. warning::

   由于上述限制以及缺乏针对错误用法的编译器诊断，建议避免从设备代码调用标准 C++ 头文件 ``std::`` 中的函数。此类函数的实现因主机平台而异。相反，强烈建议调用 CUDA C++ 标准库 libcu++ 中 ``cuda::std::`` 命名空间内的等效功能。

.. _constexpr-variables:

5.3.11.4. ``constexpr`` 变量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认情况下，``constexpr`` 变量不能用于具有不兼容执行空间的函数中，与标准变量相同。

``constexpr`` 变量可以在以下情况下直接在设备代码中使用：

- C++ 标量类型，不包括指针和指向成员类型：

  - ``nullptr_t``。

  - ``bool``。

  - 整数类型：``char``、``signed char``、``unsigned``、``long long`` 等。

  - 浮点类型：``float``、``double``。

  - 枚举：``enum`` 和 ``enum class``。

- 类类型：具有 ``constexpr`` 构造函数的 ``class``、``struct`` 和 ``union``。

- 上述类型的原始数组，例如 ``int[]``，仅当它们在 ``constexpr`` ``__device__`` 或 ``__host__ __device__`` 函数内使用时。

不允许 ``constexpr __managed__`` 和 ``constexpr __shared__`` 变量。

示例：

.. code-block:: c++

   constexpr int ConstexprVar = 4; // scalar type

   struct MyStruct {
       static constexpr int ConstexprVar = 100;
   };

   constexpr MyStruct my_struct = MyStruct{}; // class type

   constexpr int array[] = {1, 2, 3};

   __device__ constexpr int get_value(int idx) {
       return array[idx];                      // CORRECT
   }

   __device__ void foo(int idx) {
       int        v1 = ConstexprVar;           // CORRECT
       int        v2 = MyStruct::ConstexprVar; // CORRECT
   //  const int &v3 = ConstexprVar1;          // ERROR, reference to host constexpr variable
   //  const int *v4 = &ConstexprVar1;         // ERROR, address of host constexpr variable
       int        v5 = get_value(2);           // CORRECT, 'get_value(2)' is a constant expression.
   //  int        v6 = get_value(idx);         // ERROR, 'get_value(idx)' is not a constant expression
   //  int        v7 = array[2];               // ERROR, 'array' is not scalar type.
       MyStruct   v8 = my_struct;              // CORRECT
   }

在 `Compiler Explorer <https://godbolt.org/z/MWa1o3c9z>`__ 上查看示例。

.. _global-variadic-template:

5.3.11.5. ``__global__`` 可变参数模板
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可变参数 ``__global__`` 或 ``__tile_global__`` 函数模板有以下限制：

- 仅允许单个包参数。

- 包参数必须列在模板参数列表的最后。

示例：

.. code-block:: c++

   template <typename... Pack>
   __global__ void kernel1(); // CORRECT

   // template <typename... Pack, template T>
   // __global__ void kernel2(); // ERROR, parameter pack is not the last parameter

   template <typename... TArgs>
   struct MyStruct {};

   // template <typename... Pack1, typename... Pack2>
   // __global__ void kernel3(MyStruct<Pack1...>, MyStruct<Pack2...>); // ERROR, more than one parameter pack

在 `Compiler Explorer <https://godbolt.org/z/x48KnPbbY>`__ 上查看示例。

.. _defaulted-functions:

5.3.11.6. 默认函数 ``= default``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 编译器按 :ref:`隐式声明和显式默认函数 <implicitly-defaulted-functions>` 中所述推断显式默认成员函数的执行空间。

显式默认函数上的执行空间说明符会被编译器忽略，除非该函数是外联定义的或是 ``virtual`` 函数。

示例：

.. code-block:: c++

   struct MyStruct1 {
       MyStruct1() = default;
   };

   void host_function() {
       MyStruct1 my_struct; // __host__ __device__ constructor
   }

   __device__ void device_function() {
       MyStruct1 my_struct; // __host__ __device__ constructor
   }

   struct MyStruct2 {
       __device__ MyStruct2() = default; // WARNING: __device__ annotation is ignored
   };

   struct MyStruct3 {
       __host__ MyStruct3();
   };
   MyStruct3::MyStruct3() = default; // out-of-line definition, not ignored

   __device__ void device_function2() {
   //  MyStruct3 my_struct; // ERROR, __host__ constructor
   }

   struct MyStruct4 {
       //  MyStruct4::~MyStruct4 has host execution space, not ignored because virtual
       virtual __host__ ~MyStruct4() = default;
   };

   __device__ void device_function3() {
       MyStruct4 my_struct4;
       // implicit destructor call for 'my_struct4':
       //    ERROR: call from a __device__ function 'device_function3' to a
       //    __host__ function 'MyStruct4::~MyStruct4'
   }

在 `Compiler Explorer <https://godbolt.org/z/q1M4j8YYf>`__ 上查看示例。

.. _initializer-list-restrictions:

5.3.11.7. ``[cuda::]std::initializer_list``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认情况下，CUDA 编译器隐式地认为 ``[cuda::]std::initializer_list`` 的成员函数具有 ``__host__ __device__ __tile__`` 执行空间说明符，因此它们可以直接从设备代码调用。

``nvcc`` 标志 ``--no-host-device-initializer-list`` 禁用此行为；``[cuda::]std::initializer_list`` 的成员函数届时将被视为 ``__host__`` 函数，不能直接从设备代码调用。

``__global__`` 或 ``__tile_global__`` 函数不能具有 ``[cuda::]std::initializer_list`` 类型的参数。

示例：

.. code-block:: c++

   #include <initializer_list>

   __device__ void foo(std::initializer_list<int> in) {}

   __device__ void bar() {
       foo({4,5,6}); // (a) initializer list containing only constant expressions.
       int i = 4;
       foo({i,5,6}); // (b) initializer list with at least one  non-constant element.
                     // This form may have better performance than (a).
   }

在 `Compiler Explorer <https://godbolt.org/z/xeah7r44T>`__ 上查看示例。

.. _move-forward-restrictions:

5.3.11.8. ``[cuda::]std::move``、``[cuda::]std::forward``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认情况下，CUDA 编译器隐式地认为 ``std::move`` 和 ``std::forward`` 函数模板具有 ``__host__ __device__ __tile__`` 执行空间说明符，因此它们可以直接从设备代码调用。``nvcc`` 标志 ``--no-host-device-move-forward`` 禁用此行为；``std::move`` 和 ``std::forward`` 届时将被视为 ``__host__`` 函数，不能直接从设备代码调用。

.. hint::

   相反，``cuda::std::move`` 和 ``cuda::std::forward`` 始终具有 ``__host__ __device__`` 执行空间。

.. _c-14-restrictions:

5.3.12. C++14 限制
------------------

.. _functions-with-deduced-return-type:

5.3.12.1. 推导返回类型的函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``__global__`` 函数不能有推导返回类型 ``auto``
- 返回类型内省不允许在主机代码中使用

.. _variable-templates-restrictions:

5.3.12.2. 变量模板
^^^^^^^^^^^^^^^^^^

``__device__`` 或 ``__constant__`` 变量模板在使用 Microsoft 编译器时不能是 ``const`` 限定的。

.. _c-17-restrictions:

5.3.13. C++17 限制
------------------

.. _inline-variables-restrictions:

5.3.13.1. ``inline`` 变量
^^^^^^^^^^^^^^^^^^^^^^^^^

仅在分离编译模式下或对于具有内部链接的变量允许。

.. _structured-binding-restrictions:

5.3.13.2. 结构化绑定
^^^^^^^^^^^^^^^^^^^^^

不能用内存空间说明符（ ``__device__`` 、 ``__shared__`` 等）声明。

.. _c-20-restrictions:

5.3.14. C++20 限制
------------------

.. _three-way-comparison-operator:

5.3.14.1. 三向比较运算符（ ``<=>`` ）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在设备代码中支持，但可能需要 ``--expt-relaxed-constexpr`` 标志和主机实现兼容性。

.. code-block:: c++

   struct S {
       int x, y;
       auto operator<=>(const S&) const = default;
   };

.. _consteval-functions:

5.3.14.2. ``consteval`` 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以从主机和设备代码独立调用，无论其执行空间如何：

.. code-block:: c++

   consteval int host_consteval() { return 10; }

   __device__ int device_function() {
       return host_consteval();  // 正确
   }

.. note::

   有关 C++ 语言支持的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html>`_。
