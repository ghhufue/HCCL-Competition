/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ops_hccl {
CcuResult CcuKernel(CcuKernelArg arg)
{
    // TODO: 编写 CCU Kenrel 函数实现

    // 1. 初始化资源

    // 2. 加载参数

    // 3. 前同步：交换output、token和cclBuffer地址变量

    // 4. 执行算子数据面传输

    // 5. 后同步：确保所有rank数据传输完成

    return CCU_SUCCESS;
}

// TODO: 可编写多个 CCU Kernel 函数，以最大化性能
// CcuResult CcuKernel2(CcuKernelArg arg)
// {
//     return CCU_SUCCESS;
// }

} // namespace ops_hccl
