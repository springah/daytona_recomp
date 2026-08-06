# Research brief — Daytona USA XBLA native renderer

Date: 2026-08-06. Every claim is tagged **VERIFIED** (read in source/header/official doc),
**INFERRED** (my reasoning from verified facts), or **HEARSAY** (forum/secondary).

---

## Q1 — Texture cache invalidation

### 1.1 Headline

**Our vendored SDK already contains Xenia's complete physical write-watch machinery, it is
public, and nothing in this project is currently using it.** You do not need to invent a
hashing cadence; you need to register a callback.

### 1.2 How Xenia does it (three layers)

Xenia's invalidation is a three-layer chain. The SDK is a fork of it, so the local files below
are the *same code* (they even retain the upstream `TODO(Triang3l)` / `TODO(benvanik)`
comments) — I read the local copies, which is why these are VERIFIED rather than skimmed.

**Layer 1 — host page protection + access-violation handler (`Memory`).** VERIFIED.

| Symbol | Local path |
|---|---|
| `Memory::EnablePhysicalMemoryAccessCallbacks(phys_addr, length, notify, providers)` | `thirdparty/rexglue-sdk/src/system/xmemory.cpp:663` |
| `PhysicalHeap::EnableAccessCallbacks(...)` | `.../src/system/xmemory.cpp:1847` |
| `Memory::AccessViolationCallback(lock, host_address, is_write)` | `.../src/system/xmemory.cpp:516` |
| `PhysicalHeap::TriggerCallbacks(lock, va, len, is_write, unwatch_exact_range, unprotect)` | `.../src/system/xmemory.cpp:1956` |
| `Memory::RegisterPhysicalMemoryInvalidationCallback(cb, ctx)` / `Unregister...` | `.../src/system/xmemory.cpp:640` / `:648` |
| Declarations + the long design comment | `thirdparty/rexglue-sdk/include/rex/system/xmemory.h:398-471` |

Mechanism: `EnableAccessCallbacks` walks the affected **system pages**, sets a
`notify_on_invalidation` bit per page (packed 64-to-a-word so a bit-scan extracts ranges), and
calls `rex::memory::Protect(..., kReadOnly)` on the host mapping
(`xmemory.cpp:1880-1953`). A guest write then faults; the MMIO handler installed at
`xmemory.cpp:221-224` routes it to `Memory::AccessViolationCallback`, which finds the
`PhysicalHeap` and calls `TriggerCallbacks`, which fires **every** registered
`PhysicalMemoryInvalidationCallback` with `(physical_address_start, length, exact_range)`.

Watches are enabled on all three guest-physical aliases at once —
`heaps_.vA0000000`, `heaps_.vC0000000`, `heaps_.vE0000000` (`xmemory.cpp:666-671`). This
matters because `src/dt_guest_mem.cpp:36` shows Daytona's GPU-visible memory is read through
exactly `0xA0000000 / 0xC0000000 / 0xE0000000 / 0x80000000`. The first three are covered; the
cached title window `0x80000000` is **not** a `kGuestPhysical` heap and is **not** watched
(INFERRED from `AccessViolationCallback`'s `heap_type() != kGuestPhysical → return false`,
`xmemory.cpp:530`).

**Layer 2 — GPU shared memory (page-bitmap + range watches).** VERIFIED.
`SharedMemory` registers exactly **one** global invalidation callback with `Memory`
(`.../src/graphics/shared_memory.cpp:44`), then maintains its own 4 KB-page valid bitmaps and
a bucketed watch structure so subsystems can watch sub-ranges:
`RegisterGlobalWatch`, `WatchMemoryRange`, `UnwatchMemoryRange`, `FireWatches`
(`include/rex/graphics/shared_memory.h:38-80, 204-260`). Upstream doc comment (VERIFIED via
<https://raw.githubusercontent.com/xenia-project/xenia/master/src/xenia/gpu/shared_memory.h>):
range watches are **one-shot** — "The watch for the callback is cancelled after the callback -
the handle becomes invalid."

**Layer 3 — texture cache.** VERIFIED. Each `Texture` holds
`base_watch_handle_` / `mips_watch_handle_` (`include/rex/graphics/pipeline/texture/cache.h:274-275`),
installed by `Texture::MakeUpToDateAndWatch` (`:229`); `Texture::WatchCallback` (`:231`) marks
the texture outdated and it is re-uploaded on next bind. Textures are keyed by a packed
`TextureKey` (address + format + swizzle + dims) hashed with `rex::XXHasher` (`:134-179`) — the
hash is over the *key*, never over the pixel bytes. **Xenia does not content-hash textures.**

### 1.3 Granularity, cost, and the 256 KB heuristic

VERIFIED, from the SDK's own copy of Xenia's comment at
`thirdparty/rexglue-sdk/src/graphics/shared_memory.cpp:613-621`, verbatim:

> "Check if a somewhat wider range (up to 256 KB with 4 KB pages) can be invalidated - if no
> GPU-written data nearby that was not intended to be invalidated since it's not in sync with
> CPU memory and can't be reuploaded. It's a lot cheaper to upload some excess data than to
> catch access violations - with 4 KB callbacks, 58410824 (being a software-rendered game) runs
> at 4 FPS on Intel Core i7-3770, with 64 KB, the CPU game code takes 3 ms to run per frame,
> but with 256 KB, it's 0.7 ms."

Corroborated by the Xenia dev blog (Triang3l), <https://xenia.jp/updates/2021/04/27/leaving-no-pixel-behind-new-render-target-cache-3x3-resolution-scaling.html>:
smallest watchable region is 4 KB (host page); the AV handler is "a slow operation, requiring
multiple switches between the user space and the kernel space, mutex locking, and lookups";
256 KB was chosen as the optimal over-invalidation window.

So: **granularity = host page (4 KB). Cost = one page fault per newly-dirtied region, not per
write** (the page is unprotected on the first fault and stays writable until re-watched).
Steady-state cost is therefore proportional to *how many distinct regions get rewritten*, not
to bytes written.

### 1.4 Xenia's documented pitfalls (what to expect)

All VERIFIED from the header comment `include/rex/system/xmemory.h:398-443` and the blog:

- **Callbacks fire spuriously.** "callbacks may be triggered spuriously, and handlers should
  properly ignore pages they don't care about."
- **One-shot per page.** "Once callbacks are triggered for a page, the page is not watched
  anymore until requested again later." You must re-arm after each upload.
- **No per-callback removal.** "there's no way to disable only a specific callback for a page."
- **Invalidate on Release, not Alloc.** The header warns that if invalidation is done at Alloc
  time, "Alloc won't be aware of callbacks enabled in other heaps, thus callback handlers will
  keep considering this range valid forever."
- **Never blanket-invalidate GPU-written data.** Data the GPU produced (resolve destinations,
  memexport) has no CPU-side copy to re-upload from; the 256 KB widening must be clipped
  against it. That is precisely what `SharedMemory::MemoryInvalidationCallback`'s
  `system_page_flags_valid_and_gpu_written_` scan does (`shared_memory.cpp:598-640`).
  Your existing `g_resolve_written` stamp (`src/dt_plume.cpp:1826`) is the same idea and should
  keep taking precedence over any watch-driven path.
- **Read watches are not implemented** — `TriggerCallbacks` opens with
  `assert_true(is_write)` (`xmemory.cpp:1959-1963`). Write-only.
- **Data providers are not implemented** — `assert_false(enable_data_providers)`
  (`xmemory.cpp:1851`). Pass `enable_data_providers = false`.

### 1.5 Can our renderer, sitting outside the SDK's GPU module, register watches?

**Yes. VERIFIED, with no caveats about module boundaries.**

- `Memory::RegisterPhysicalMemoryInvalidationCallback` and
  `EnablePhysicalMemoryAccessCallbacks` are **public** methods on `rex::memory::Memory`
  (`include/rex/system/xmemory.h:452, 460`), reachable from anywhere.
- The instance is globally reachable: `rex::system::kernel_memory()`
  (`include/rex/system/kernel_state.h:405-408`), which the project already uses as
  `REX_KERNEL_MEMORY()` in `src/dt_guest_mem.cpp:15`.
- **The slot is free.** The only in-SDK registrants are `SharedMemory::Initialize`
  (`src/graphics/shared_memory.cpp:44`) and `primitive_processor.cpp:1373` — both live in the
  SDK GraphicsSystem, which this project **replaces wholesale** with `PlumeGraphicsSystem`
  (`src/gpu_plume/plugin.cpp:90, 1014`). INFERRED but strongly: no SDK component is arming
  watches today, so `Memory`'s callback list is empty and the pages are unprotected.

**Recipe (INFERRED from the verified API, not tested):**

1. At renderer init: `handle = kernel_memory()->RegisterPhysicalMemoryInvalidationCallback(&Thunk, this);`
   Signature is `std::pair<uint32_t,uint32_t>(*)(void* ctx, uint32_t phys_start, uint32_t length, bool exact_range)`.
2. After each successful texture upload, arm the watch over the source bytes:
   `kernel_memory()->EnablePhysicalMemoryAccessCallbacks(t.base_address, source_bytes, /*invalidation*/true, /*providers*/false);`
   Note it takes a **physical** address (which is what `DecodedTexture::base_address` already
   is, per `dt_plume.cpp:1349`), and it snaps to system page boundaries.
3. In the callback: mark every `g_texture_index` entry whose `[base_address, base_address+source_bytes)`
   intersects `[phys_start, phys_start+length)` as dirty. Return the safe-to-unwatch range —
   `(0, UINT32_MAX)` means "don't care"; returning something narrower is how you protect
   GPU-written regions from being unwatched. The runtime intersects the returns of all
   callbacks (`xmemory.cpp:2025-2035`).
4. On next bind, a dirty entry re-uploads and re-arms.

**Hard constraints on the callback body (VERIFIED from call sites):**
- It runs **on the faulting guest thread**, inside the global critical region, with the guest
  mid-instruction. Do no GPU work, take no renderer mutex that a render thread can hold while
  touching guest memory, allocate nothing that can block. Set a flag; do the work at bind time.
- The upstream doc explicitly says do not watch/unwatch from inside a watch callback.
- Only guest-physical heaps are covered. Anything read through the `0x80000000` alias is not.
- Pages the *guest* has marked read-only are deliberately skipped so real access violations
  still occur (`xmemory.cpp:1920-1935`) — a texture in a read-only page is never watched, but
  it also can't be rewritten, so that is benign.

### 1.6 If watches turn out not to work — ranked alternatives

1. **Use the guest's own signal: `D3DDevice_InvalidateGpuCache`.** VERIFIED from the real XDK
   header at `C:/tmp/xdk20209/extracted/XDK/include/xbox/d3d9.h:2286`:
   `void D3DDevice_InvalidateGpuCache(D3DDevice*, void* pBaseAddress, DWORD Size, DWORD Flags)`
   plus `D3DDevice_InvalidateResourceGpuCache(D3DDevice*, D3DResource*, DWORD)` (`:2287`).
   On real hardware a title **must** call this after the CPU rewrites texture memory, or the
   GPU's own texture cache serves stale texels. This is an *exact* (address, size) invalidation
   signal delivered by the game itself, with zero polling cost.
   **This project already knows where it is**: `src/dt_d3d9_hooks.cpp:129` maps
   `sub_82221818` → `D3DDevice_InvalidateGpuCache`, but it is only `DT_D3D9_TRACE`'d, with no
   semantic hook. *Cheapest possible next experiment: check the existing trace for hits on
   slot 38 and, if it fires, wire it to the cache.* If it fires with the right addresses it is
   strictly better than write-watches — no page faults, no over-invalidation, exact ranges.
2. **Lock/Unlock bracketing** — UnleashedRecomp's approach (see Q3); only works if Daytona
   updates textures through `LockRect`/`UnlockRect` rather than raw stores.
3. **Full content hashing** (XXH3 over the whole source, or the current sampled probe widened).
   The current sampled probe — 32 windows × 16 bytes = 512 bytes (`src/dt_plume.cpp:1901-1909`)
   — is honest about its own failure mode: "It cannot see a change confined to bytes it does
   not sample." A palette or a small UI atlas patch is exactly that shape of change.
4. **Frame-count TTL — do not.** Xenia's experience argues against it in both directions: too
   short and you re-upload 8 MB resolve destinations every frame (the blog's whole point is
   that avoiding needless re-upload is what made the new cache fast); too long and you get the
   stale-art defect back non-deterministically, which is worse than a deterministic one because
   it will reproduce differently on every machine. No source I found endorses TTLs.

---

## Q2 — Xbox 360 Clear / EDRAM / backbuffer persistence semantics

Primary source throughout: the actual XDK 20209.3 headers on this machine —
`C:/tmp/xdk20209/extracted/XDK/include/xbox/d3d9.h` and `d3d9types.h`. All quotes are verbatim
from those files (VERIFIED).

### 2.1 The Resolve-with-clear question — CONFIRMED, and it is real

`d3d9types.h:222-226`, verbatim:

```
#define D3DRESOLVE_CLEARRENDERTARGET        0x0100  /* Clear the specified render target after resolving */
#define D3DRESOLVE_CLEARDEPTHSTENCIL        0x0200  /* Clear the depth/stencil buffer after resolving */
#define D3DRESOLVE_EXPONENTBIAS_SHIFT       26
#define D3DRESOLVE_EXPONENTBIAS(Bias)       ((Bias) << D3DRESOLVE_EXPONENTBIAS_SHIFT)
```

The clear values ride on `Resolve` itself (`d3d9.h:2216`):

```
void WINAPI D3DDevice_Resolve(D3DDevice* pDevice, DWORD Flags, CONST D3DRECT *pSourceRect,
                              D3DBaseTexture *pDestTexture, CONST D3DPOINT *pDestPoint,
                              UINT DestLevel, UINT DestSliceOrFace,
                              CONST D3DVECTOR4* pClearColor, float ClearZ, DWORD ClearStencil,
                              CONST D3DRESOLVE_PARAMETERS* pParameters);
```

**So yes: a title can clear EDRAM entirely through `Resolve` and never call `Clear` once.**
`pClearColor` / `ClearZ` / `ClearStencil` are otherwise unused parameters that only take effect
when bits `0x0100` / `0x0200` are set. `pClearColor` is a `D3DVECTOR4*` (float RGBA), *not* a
packed `D3DCOLOR`.

Full `Flags` layout (VERIFIED, `d3d9types.h:207-226`):

| Bits | Meaning |
|---|---|
| `0x0007` | Source select: `RENDERTARGET0`=0x0, `1`=0x1, `2`=0x2, `3`=0x3, `DEPTHSTENCIL`=0x4 |
| `0x0070` | Fragment select: `ALLFRAGMENTS`=0x00, `FRAGMENT0..3`=0x10/0x20/0x30/0x40, `FRAGMENTS01`=0x50 (weight 0.5), `FRAGMENTS23`=0x60, `FRAGMENTS0123`=0x70 (weight 0.25) |
| `0x0100` | Clear render target after resolve |
| `0x0200` | Clear depth/stencil after resolve |
| `31:26` | Exponent bias, −32..31, applied before formatting |

Trap: `D3DRESOLVE_RENDERTARGET0 == 0x0000`, so "no source bits" means RT0, not "unset".

`D3DDevice_EndTiling` carries the identical payload (`d3d9.h:2234`) — same clear-on-resolve
semantics at the end of a tiling bracket:

```
HRESULT WINAPI D3DDevice_EndTiling(D3DDevice*, DWORD ResolveFlags, CONST D3DRECT* pResolveRects,
                                   D3DBaseTexture* pDestTexture, CONST D3DVECTOR4* pClearColor,
                                   float ClearZ, DWORD ClearStencil,
                                   CONST D3DRESOLVE_PARAMETERS* pParameters);
```

**Action:** the project already hooks `D3DDevice_Resolve`, `D3DDevice_ResolveEx` and
`D3DDevice_EndTiling` (`src/dt_d3d9_hooks.cpp`). Decode `Flags & 0x0300` in all three and treat
those bits as a Clear of the *current EDRAM surface*, issued **after** the copy-out. If that is
not already done, a title that clears this way will look exactly like "the frame target never
gets cleared" — which is the failure mode you just introduced by persisting across Swap.

### 2.2 `D3DDevice_Clear` has an undocumented-in-public `EDRAMClear` parameter

VERIFIED, `d3d9.h:2175`:

```
void WINAPI D3DDevice_Clear(D3DDevice *pDevice, DWORD Count, CONST D3DRECT *pRects, DWORD Flags,
                            D3DCOLOR Color, float Z, DWORD Stencil, BOOL EDRAMClear);
```

The portable D3D9 inline wrappers hard-code it to `FALSE`
(`d3d9.h:3313`, `d3d9.h:3629`), so `EDRAMClear == TRUE` only appears if the title calls the
underscore form directly. What `TRUE` changes is **not stated in the header** and I found no
public documentation for it — treat as unknown; log it if the hook ever sees a non-zero value.

There is also `D3DDevice_ClearF(pDevice, Flags, pRect, pColor /*D3DVECTOR4*/, Z, Stencil)`
(`d3d9.h:3314`) — float colour, single rect, no `EDRAMClear`. Already hooked.

Clear flags (VERIFIED, `d3d9types.h:190-202`):

```
D3DCLEAR_TARGET0        0x01   D3DCLEAR_TARGET1  0x02
D3DCLEAR_TARGET2        0x04   D3DCLEAR_TARGET3  0x08
D3DCLEAR_ALLTARGETS     0x0F   (== D3DCLEAR_TARGET)
D3DCLEAR_ZBUFFER        0x10   D3DCLEAR_STENCIL  0x20
D3DCLEAR_HISTENCIL_CULL 0x40   D3DCLEAR_HISTENCIL_PASS 0x80
```

Two traps: (a) `D3DCLEAR_TARGET` is *all four* MRTs, so a naive `Flags & D3DCLEAR_TARGET` test
is fine but `Flags == D3DCLEAR_TARGET0` is a single-RT clear you must not apply to RT1..3;
(b) `HISTENCIL_CULL`/`PASS` clear the **hierarchical stencil**, an occlusion accelerator with
no host analogue — a Clear whose flags are `0x40` alone must be a **no-op**, not a colour clear.
Related: `D3DDevice_FlushHiZStencil(D3DFHZS_ASYNCHRONOUS|D3DFHZS_SYNCHRONOUS)` (`d3d9.h:2288`,
`d3d9types.h:3008-3012`) — also a no-op for us.

### 2.3 What Clear compiles to on real hardware

VERIFIED (Xenia dev blog, link above): the Xenos has **no dedicated clear packet**. Clears are
emitted by the D3D9 runtime as *draws into EDRAM*, exploiting MSAA replication — "to clear a
1280x720 single-sampled render target, games can draw a 640x360 rectangle with 4x MSAA at the
same location in eDRAM"; for depth, "draw a rectangle parallel to the 'view' to a 4x MSAA depth
render target… with the low 8 bits of the needed packed 32-bit value written via stencil."

**Consequence for us: mostly irrelevant, with one sting.** Because we intercept at the D3D9
API, we get a clean `Clear` call and never see the PM4. But the same reasoning means **a title
can also hand-roll a clear as an ordinary fullscreen draw**, which we will never recognise as a
clear. INFERRED: if you ever see "the target isn't clearing" with no `Clear` and no
`0x0300` resolve flags, look for a fullscreen untextured quad at the top of the frame.

### 2.4 Persistence across Swap — your change is the authentic behaviour

VERIFIED that the presented image is a *resolved texture in main RAM*, not EDRAM:
`void WINAPI D3DDevice_Swap(D3DDevice*, D3DBaseTexture* pFrontBuffer, CONST D3DVIDEO_SCALER_PARAMETERS*)`
(`d3d9.h:2228`). EDRAM is never scanned out directly; the frame reaches the screen only via
Resolve → texture → Swap.

INFERRED (from that plus the absence of any system-side EDRAM clear in the API): EDRAM contents
persist across Swap, and a title that draws a full-frame skybox each frame legitimately never
clears colour. **Persisting the frame target and clearing only on guest Clear is correct.**
UnleashedRecomp does the same (Q3 §3.2).

### 2.5 Edge cases to expect, in the order I would guard them

1. **Resolve-with-clear (§2.1).** Highest probability, and it exactly mimics "never clears".
2. **Predicated tiling replays the clear per tile.** VERIFIED (Microsoft, archived XNA GS docs,
   <https://learn.microsoft.com/en-us/previous-versions/windows/xna/bb464139(v=xnagamestudio.41)>):
   "the commands issued in the **Draw** method are recorded before execution. The recorded
   commands… are then executed for each tile… Once the primitives for a tile are fully
   rendered, the tile is then resolved into the texture that is used for the front buffer. Each
   successive tile is handled the same way and is resolved into the same texture." So inside a
   `BeginTiling`/`EndTiling` bracket **one guest Clear = N hardware clears**, each scissored to
   its tile, and there are **N resolves into different sub-rects of one destination**. Your
   `D3DDevice_Resolve` handler must therefore not assume one resolve per frame, and
   `pDestPoint` matters. Also VERIFIED from that page: "Resources that are referenced in the
   command buffer cannot be modified by the CPU" during tiling.
3. **`BeginTiling` carries its own clear.** VERIFIED, `d3d9.h:2233`:
   `D3DDevice_BeginTiling(D3DDevice*, DWORD Flags, DWORD Count, CONST D3DRECT* pTileRects, CONST D3DVECTOR4* pClearColor, float ClearZ, DWORD ClearStencil)`
   — i.e. **entering a tiling bracket clears each tile**, before any guest `Clear` is seen.
   Modified by (`d3d9types.h:243-245`):
   `D3DTILING_SKIP_FIRST_TILE_CLEAR 0x01`, `D3DTILING_ONE_PASS_ZPASS 0x02`,
   `D3DTILING_FIRST_TILE_INHERITS_DEPTH_BUFFER 0x04`. If `pClearColor` is NULL, INFERRED: no
   colour clear. This is the second-most-likely explanation for "the game clears but we see no
   Clear call" — it is already hooked (`dt_d3d9_hooks.cpp`), so check whether the handler
   currently honours the clear arguments.
4. **Depth-only clears.** `Flags == D3DCLEAR_ZBUFFER|D3DCLEAR_STENCIL` with no target bits is
   normal and must not clear colour. Conversely a colour-only clear must not touch depth.
   With a persisted target this asymmetry becomes visible for the first time.
5. **Rect-limited clears.** `Count`/`pRects` are real; a persisted target makes a partial clear
   observable where a full-clear-every-frame renderer hid it.
6. **EDRAM aliasing / multiple surfaces.** `D3DSURFACE_PARAMETERS { DWORD Base; DWORD HierarchicalZBase; INT ColorExpBias; D3DHIZFUNC HiZFunc; }`
   (`d3d9types.h:2295-2302`) lets the title place several surfaces at overlapping EDRAM offsets
   and reuse the space within a frame. "The frame target" is only well-defined if Daytona uses
   one; if `Base` varies, persistence must be per-surface-identity, not per-renderer.
7. **`ColorExpBias`.** Present on both the surface (`d3d9types.h:2299`) and the resolve
   (`D3DRESOLVE_EXPONENTBIAS`, bits 31:26). A clear colour written into a biased surface is not
   the same number as the same clear colour written into an unbiased one.
8. **EDRAM-only formats.** `D3DFMT_A2B10G10R10F_EDRAM`, `D3DFMT_G16R16_EDRAM`,
   `D3DFMT_A16B16G16R16_EDRAM` (`d3d9types.h:2061-2065`) exist only as render targets and
   convert on resolve; a clear value for these is not a plain RGBA8.
9. **`D3DDevice_SetPredication` / `D3DPRED_TILE_Z(n)` / `D3DPRED_TILE_RENDER(n)`**
   (`d3d9types.h:266-267`) — already hooked; per-tile predication is how the tile replay is
   gated, and if it is ignored, tiled frames double-draw.

---

## Q3 — Cousin projects' native-renderer patterns

Researched by a delegated agent that shallow-cloned `hedge-dev/UnleashedRecomp` and `rt64/rt64`
and read the sources directly; line numbers below are from those clones (VERIFIED by that
agent's direct read, not re-verified by me).

### 3.1 EDRAM / render targets — no emulation at all
`grep -i edram` over `UnleashedRecomp/gpu/video.cpp` (7,882 lines) returns **zero hits**.
`SetRenderTarget` (`video.cpp:3324`, guest `sub_82BDD9F0`) enqueues a command;
`ProcSetRenderTarget` (`:3334`) swaps a `GuestSurface*` pointer whose backing is a 1:1 host
`RenderTexture` from `g_device->createTexture()` (`CreateSurface`, `:3184-3204`). No tiling, no
EDRAM offsets, no scanout-memory step. **Transfers directly** — the D3D9 API layer already
abstracted EDRAM away before our hook ever sees a call.

### 3.2 Clear — 1:1, and no forced clear on Swap
`Clear()` `:3663` → `ProcClear` `:3676` → `commandList->clearColor(...)` / `clearDepth(...)`,
only for the flags the guest passed. Render targets persist across Swap unless the guest clears
or rebinds. **This is independent confirmation that the change you just made is the right
shape.** Transfers directly.

### 3.3 Texture cache — none, because the API is the dirty signal
There is **no address-keyed cache, no hashing, no dirty flags**. One host `RenderTexture` per
guest texture object at `CreateTexture` (`:3095`). The only write path is
`LockTextureRect`/`UnlockTextureRect` (`:2168-2206`): Unlock memcpys into an upload allocator
and issues `copyTextureRegion`. No `XXH`/`hash`/`dirty` symbols anywhere near texture code.

**Transfers *if* Daytona updates textures through Lock/Unlock.** If Daytona instead writes guest
memory directly and signals with `InvalidateGpuCache` (Q1 §1.6.1), that call is the equivalent
signal. Either way the lesson is the same: **at D3D9-hook level the guest tells you when memory
changed — find the call, don't guess with a hash.**

### 3.4 Frame pacing — the standout technique to copy
- Dedicated `g_renderThread` (`video.cpp:5249`) draining a lock-free MPSC queue
  (`g_renderQueue.wait_dequeue_bulk`, `:5260`); every D3D9 hook `enqueue()`s rather than
  recording inline.
- The kernel import `VdSwap` (`kernel/imports.cpp:794`) is a **complete no-op stub**. Pacing
  does not come from the guest's vblank path at all.
- Instead, the guest's own per-frame `CApplication::Update` (`app.cpp:45-51`) calls
  `Video::WaitOnSwapChain()` (`video.cpp:2773`) as its first line, which blocks on
  `g_swapChain->wait()` with `setVsyncEnabled(Config::VSync)` (`:1532/:1879`).
  `Video::Present()` (`:2791`) additionally does `g_executedCommandList.wait(false)`.

**Transfers directly**, and is worth copying: replace the guest's vblank wait with a host
swapchain wait at one hook point, and move command recording off the guest thread. Note the
interaction with Q1: if the invalidation callback runs on the guest thread while the render
thread is reading guest memory, that is the deadlock you have to design against.

### 3.5 MSAA — guest request reduced to a boolean
`CreateSurface` (`:3184`):
`desc.multisampling.sampleCount = multiSample != 0 && Config::AntiAliasing != None ? int32_t(Config::AntiAliasing.Value) : COUNT_1;`
The guest's `D3DMULTISAMPLE_TYPE` only gates *whether* to use MSAA; the level comes from a user
setting, capability-checked against `g_device->getSampleCountsSupported()` (`:1823-1833`).
Resolve picks hardware (`RESOLVE_SOURCE`/`RESOLVE_DEST`) or a shader fallback in
`PopulateBarriersForStretchRect` (`:3364`); MSAA surfaces sampled as textures queue through
`g_pendingMsaaResolves` (`:3818`). Mostly transfers; the resolve plumbing is reusable as-is if
we are also on plume.

### 3.6 RT64 (Zelda64Recomp) — deliberately does NOT transfer
`rt64/src/common/rt64_tmem_hasher.h` — `TMEMHasher::hash`, `CurrentHashVersion = 5`, XXH3 over
the emulated 4 KB TMEM per texture-bind microcode op, folding in width/height/tlut/line/siz/fmt,
and hashing only the rows actually addressed (`needsToHashRowsIndividually`). Cache in
`src/render/rt64_texture_cache.cpp` (`TextureMap::use/add/replace/evict`, `:217-375`).
Framebuffer/RDRAM aliasing handled by hashing raw RDRAM: `rt64_framebuffer_manager.cpp`
`storeRAM` `:817`, `checkRAM` `:828`, `hashTracking` `:894`.

**Does not transfer, and that is the point.** RT64 sits *below* an API — it interprets display
lists and raw RDRAM, so hashing is its only dirtiness signal. We sit *above* one. Porting RT64
hashing into a D3D9-hook renderer buys overhead, not correctness. Bounded-cost hashing is a
fallback for us (Q1 §1.6.3), not a design.

---

## Q4 — Daytona USA XBLA specifics

Researched by a delegated agent. Confidence is uneven; tags are per-claim.

### 4.1 Engine provenance — largely unresolved
- **VERIFIED** (<https://en.wikipedia.org/wiki/Daytona_USA>): the 2011 re-release includes "both
  the original arcade soundtrack and the arranged soundtrack from *Sega Racing Classic*". That
  is a documented **audio** link to the RingEdge/RingWide remake — **not** evidence of a shared
  code or render engine. Treat any "it's the SEGA Racing Classic codebase" claim as unproven.
- Developer: multiple secondary listings attribute the port to **Sega-AM2**; the MobyGames
  credits page returned 403/402 and could not be read, so this is **HEARSAY**. The "Sumo
  Digital" rumour is **not supported by any source found** — Sumo's confirmed Sega racing work
  is the *Sonic & Sega All-Stars Racing* line.
- Emulation vs source port vs rewrite: **not found.** No developer interview or Sega statement.
  Period marketing ("arcade-perfect… using original arcade assets") suggests a reimplementation
  using original asset data, but that is **INFERRED** from marketing copy.

### 4.2 Model 2 palette / colour-index heritage — good primary source exists
**VERIFIED** from MAME `src/mame/sega/model2_v.cpp` and `model2.cpp`
(<https://github.com/mamedev/mame/blob/master/src/mame/sega/model2_v.cpp>):
- Untextured polygon colour: `object.colorbase` → `m_palram[color + 0x1000]`, 5 bits per
  channel, then through `m_colorxlat[]` — **three separate 256-entry translation tables** for
  R/G/B at offsets `0x0000` / `0x4000` / `0x8000`. It is a LUT stage, not raw RGB.
- `palette_w()` converts the hardware's 15-bit `xGGGGGRRRRRBBBBB` word to 8-bit-per-channel via
  `m_colorxlat` (offsets `0x0080/0x4080/0x8080`), then applies `m_gamma_table`.
- Textured polygons carry a **luma base**: `extra.lumabase = (poly->texheader[1] & 0xff) << 7;`
  `extra.colorbase = (poly->texheader[3] >> 6) & 0x3ff;` `extra.luma = poly->luma;` —
  i.e. **8-bit indexed/luminance texels combined with a per-polygon colour+luma base**, which
  matches the k_8_8 "index in .x" decode this renderer already implements
  (`src/dt_plume.cpp:1841-1844`).
- Texture-sheet addressing: `texwidth = 32 << ((texheader[0]>>0)&7)`,
  `texheight = 32 << ((texheader[0]>>3)&7)`, `texx = 32*((texheader[2]>>0)&0x3f)`,
  `texy = 32*((texheader[2]>>6)&0x1f)` — 32-texel tiling with an LOD-select bit, plus 128×128
  "microtextures". UVs are 13.3 fixed point.
- Caveat: the reported memory-map addresses (texture RAM ~`0x12000000`, luma RAM ~`0x12800000`,
  palette RAM `0x01800000`, colorxlat `0x01810000`) came through a summarizer — **re-read the
  file directly before relying on them.**
- Whether the 2011 port reproduces this scheme in shaders: **not found** in any public source.
  Only readable from the port's own shader bytecode — which this project is already dumping.

### 4.3 Resolution / framerate
**60 fps and native 16:9 are corroborated** across period reviews (xblafans.com, RotoRob).
The "full 1080p" figure traces only to a marketing-derived listing — **HEARSAY**. No
XBLA-specific Digital Foundry face-off was found (the DF Retro Daytona episode appears to cover
Saturn/Dreamcast/PC/PS3). One review notes "bad pop-in of scenery on the Expert course" and
"moments of missing background entirely" — worth A/Bing against our LOD/culling.

### 4.4 Attract / ranking flow — the useful bit
**VERIFIED** (game-over-dex.fandom.com/wiki/Daytona_USA, arcade behaviour): after the ranking
screen concludes (or GAME OVER displays) the game **cuts directly to attract mode**, and when
rankings display, the background demo scenes **reset to their beginning — except the Three
Seven Speedway demo, which advances to its second scene instead**. So on arcade Daytona the
ranking screen and the attract demo are directly coupled in a loop, with per-course demo state.
**INFERRED**: if the XBLA port preserved that structure, letting attract run through a full
cycle should reach a ranking screen without playing a race — a cheap way to hit the screen you
still need. Not verified for the XBLA build.
Arcade manual (**HEARSAY**, snippet only): attract/ADVERTISE mode cycles view changes and
lights the View Change button lamp.
XBLA additions (**VERIFIED**, Wikipedia + reviews): Karaoke, Survival (own leaderboard),
Challenge (30 challenges, 10/track, sequential), 8-player online, Time Trial with online
leaderboards, three base tracks (six with mirrors).

⚠ **Anomaly worth your attention, not acted on:** the delegated agent reported that
<https://tcrf.net/Daytona_USA_(Arcade)> returned, in place of wiki content, text that read as
**instructions directed at an AI agent**. Nothing from that page was used and no instruction
from it was followed. I have no reliable read on what TCRF actually documents for this game —
if you want it, open it in a browser yourself.

---

## Q5 — Quick wins / known-tricky spots

1. **A sibling project exists and is public: `github.com/Subarasheese/daytona-xbla-recomp`**
   (VERIFIED, README read directly) — "A static recompilation of Daytona USA (Xbox 360 / XBLA,
   2011)", built on **ReXGlue**, using the **`daytonaxbla` branch** of the ReXGlue SDK. Ships
   `daytona_symbols.h` with "a complete symbol map from a working version". Its roadmap
   explicitly includes "native rendering backend to replace Xbox 360 GPU emulation with
   Vulkan/D3D12" — the same work. 9 open issues, mostly build/platform (VS2026, Wayland/X11,
   "CachyOS texture glitches", a menu softlock on achievements access). Corroborated by
   <https://timeextension.com/news/2026/06/native-pc-port-of-xbla-daytona-usa-is-here-if-youre-willing-to-do-some-leg-work>.
   **Its symbol map is worth diffing against ours**, especially for the D3D9 slots still marked
   "(unidentified)" in `src/dt_d3d9_hooks.cpp`.
2. **Xenia compatibility issue #1123, "58410B1D - DAYTONA USA"** (VERIFIED,
   <https://github.com/xenia-project/game-compatibility/issues/1123>), status *state-playable*,
   filed 2018-12-11 against commit `7698944`:
   - **Demo/attract mode is playable but has significant graphics corruption that drags the
     framerate down in specific track areas.** That is the same surface our renderer is working
     on — if Xenia (a PM4-level emulator) also corrupts there, the content is unusual, not our
     hook.
   - **The paid game freezes on the main menu and cannot create a save file** (kernel-level save
     errors). INFERRED, flagged as a hypothesis only: that symptom shape matches the
     kernel-stub class documented in this user's own `futari-kernel-stub-audit` notes — an
     import returning SUCCESS without writing its out-param. Worth checking Daytona's save-path
     imports before assuming a renderer cause for any menu hang.
   - Also notes vsync-off changes game speed — relevant to Q3 §3.4 pacing choices.
   - The `xenia-project/game-compatibility` repo is unmaintained; tracking moved to
     `xenia-canary/game-compatibility`, where no newer Daytona-specific issue was found.

---

## Recommended next actions (my ranking)

1. **Grep the existing D3D9 trace for slot 38 (`sub_82221818` = `D3DDevice_InvalidateGpuCache`).**
   Zero cost, and if it fires it is a better invalidation source than anything else in this brief.
2. **Decode `Flags & 0x0300` in the `Resolve` / `ResolveEx` / `EndTiling` hooks** and in
   `BeginTiling`'s `pClearColor`. If Daytona clears this way, the persist-across-Swap change
   will otherwise present as a regression.
3. **Register a `PhysicalMemoryInvalidationCallback`** per Q1 §1.5 as the exact replacement for
   the 512-byte sampled probe. Arm on upload, clear on bind, keep `g_resolve_written`
   authoritative for GPU-written ranges.
4. **Diff `Subarasheese/daytona-xbla-recomp`'s `daytona_symbols.h`** against our unidentified
   D3D9 slots.
