/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    // TODO: 根据单次通信量约束（256MB）、HCCL Buffer大小约束（400MB），完成数据切分和 CCU Kernel 下发
    // 调用 HcommCcuKernelLaunch() 等接口完成 CCU Kernel 下发
    // NOTE: 如果有多个 CCU Kernel，则需要分别下发

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
