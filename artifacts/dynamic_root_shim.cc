#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>

#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>

extern "C" HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t, HcclComm comm, aclrtStream stream)
{
    using BroadcastFn = HcclResult (*)(
        void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    static auto next = reinterpret_cast<BroadcastFn>(dlsym(RTLD_NEXT, "HcclBroadcast"));
    static std::atomic<uint32_t> callIndex{0};
    const char *alternateText = std::getenv("HCCL_TEST_ALTERNATE_ROOT");
    const uint32_t alternateRoot = alternateText == nullptr ? 1U :
        static_cast<uint32_t>(std::strtoul(alternateText, nullptr, 10));
    const uint32_t sequence[] = {0U, alternateRoot, 0U};
    const uint32_t index = callIndex.fetch_add(1, std::memory_order_relaxed);
    const uint32_t root = index < 3 ? sequence[index] : 0U;
    return next == nullptr ? HCCL_E_INTERNAL : next(buf, count, dataType, root, comm, stream);
}
